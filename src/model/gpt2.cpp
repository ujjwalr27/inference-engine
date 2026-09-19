#include "model/gpt2.h"

#include <stdexcept>

namespace gpt2 {
namespace {

class WeightReader {
 public:
  WeightReader(const SafeTensors& st, torch::Device device, torch::Dtype dtype)
      : st_(st), device_(device), dtype_(dtype) {}

  torch::Tensor take(const std::string& name, const std::vector<int64_t>& expected_shape) const {
    if (!st_.contains(name)) throw std::runtime_error("missing weight: " + name);
    if (st_.shape(name) != expected_shape) {
      throw std::runtime_error("shape mismatch for " + name + ": expected " +
                               c10::str(c10::ArrayRef<int64_t>(expected_shape)) + ", got " +
                               c10::str(c10::ArrayRef<int64_t>(st_.shape(name))));
    }
    return st_.load(name).to(device_, dtype_);
  }

  Linear linear(const std::string& prefix, int64_t in, int64_t out) const {
    return {take(prefix + ".weight", {out, in}), take(prefix + ".bias", {out})};
  }

  LayerNorm layer_norm(const std::string& prefix, int64_t dim, double eps) const {
    return {take(prefix + ".weight", {dim}), take(prefix + ".bias", {dim}), eps};
  }

 private:
  const SafeTensors& st_;
  torch::Device device_;
  torch::Dtype dtype_;
};

}  // namespace

GPT2Model GPT2Model::load(const std::string& dir, torch::Device device, torch::Dtype dtype) {
  const auto cfg = GPT2Config::from_json_file(dir + "/config.json");
  const auto weights = SafeTensors::open(dir + "/model.safetensors");
  return GPT2Model(cfg, weights, device, dtype);
}

GPT2Model::GPT2Model(const GPT2Config& config, const SafeTensors& weights, torch::Device device, torch::Dtype dtype)
    : cfg_(config), device_(device), dtype_(dtype) {
  cfg_.validate();
  if (!c10::isFloatingType(dtype)) {
    throw std::invalid_argument("model dtype must be floating point");
  }
  torch::InferenceMode guard;
  const WeightReader r(weights, device, dtype);
  const int64_t D = cfg_.n_embd;

  wte_ = r.take("wte", {cfg_.vocab_size, D});
  wpe_ = r.take("wpe", {cfg_.n_ctx, D});

  blocks_.reserve(static_cast<size_t>(cfg_.n_layer));
  for (int64_t i = 0; i < cfg_.n_layer; ++i) {
    const std::string p = "h." + std::to_string(i) + ".";
    Block b;
    b.ln_1 = r.layer_norm(p + "ln_1", D, cfg_.layer_norm_epsilon);
    b.attn.q_proj = r.linear(p + "attn.q_proj", D, D);
    b.attn.k_proj = r.linear(p + "attn.k_proj", D, D);
    b.attn.v_proj = r.linear(p + "attn.v_proj", D, D);
    b.attn.c_proj = r.linear(p + "attn.c_proj", D, D);
    b.attn.n_head = cfg_.n_head;
    b.ln_2 = r.layer_norm(p + "ln_2", D, cfg_.layer_norm_epsilon);
    b.mlp.c_fc = r.linear(p + "mlp.c_fc", D, 4 * D);
    b.mlp.c_proj = r.linear(p + "mlp.c_proj", 4 * D, D);
    blocks_.push_back(std::move(b));
  }
  ln_f_ = r.layer_norm("ln_f", D, cfg_.layer_norm_epsilon);
}

torch::Tensor GPT2Model::forward(const torch::Tensor& ids, const torch::Tensor& positions,
                                 const torch::Tensor& padding_mask) const {
  return run(ids, positions, padding_mask, nullptr);
}

std::vector<torch::Tensor> GPT2Model::hidden_states(const torch::Tensor& ids, const torch::Tensor& positions,
                                                    const torch::Tensor& padding_mask) const {
  std::vector<torch::Tensor> hidden;
  run(ids, positions, padding_mask, &hidden);
  return hidden;
}

torch::Tensor GPT2Model::prefill(const torch::Tensor& ids, int64_t slot, KVCache& cache, int64_t start_pos) const {
  TORCH_CHECK(ids.dim() == 2 && ids.size(0) == 1, "prefill expects ids [1, T], got ", ids.sizes());
  TORCH_CHECK(ids.scalar_type() == torch::kInt64, "ids must be int64");
  const int64_t T = ids.size(1);
  TORCH_CHECK(T >= 1 && start_pos + T <= cfg_.n_ctx, "prefill range [", start_pos, ", ", start_pos + T,
              ") outside context ", cfg_.n_ctx);

  torch::InferenceMode guard;
  const auto pos = torch::arange(start_pos, start_pos + T, torch::TensorOptions().dtype(torch::kInt64).device(device_));
  auto x = torch::embedding(wte_, ids.to(device_)) + torch::embedding(wpe_, pos).unsqueeze(0);

  // This pass attends only over the tokens it just produced, so it is correct only from position 0.
  // Chunked prefill (attending over already-cached keys) is a Phase 8 item.
  TORCH_CHECK(start_pos == 0, "chunked prefill (start_pos > 0) is not implemented yet");
  const auto allowed = build_attention_mask(T, {}, device_);
  for (int64_t layer = 0; layer < cfg_.n_layer; ++layer) {
    x = blocks_[static_cast<size_t>(layer)].forward_prefill(x, allowed, layer, slot, start_pos, cache);
  }
  x = ln_f_(x.slice(1, T - 1, T));  // only the last position feeds the LM head
  return torch::linear(x, wte_).squeeze(1);
}

torch::Tensor GPT2Model::prefill_batch(const torch::Tensor& ids, const torch::Tensor& positions,
                                       const torch::Tensor& padding_mask, KVCache& cache) const {
  TORCH_CHECK(ids.dim() == 2, "prefill_batch expects ids [B, T], got ", ids.sizes());
  TORCH_CHECK(ids.scalar_type() == torch::kInt64, "ids must be int64");
  TORCH_CHECK(positions.sizes() == ids.sizes() && padding_mask.sizes() == ids.sizes(),
              "positions and padding_mask must match ids shape");
  const int64_t B = ids.size(0);
  const int64_t T = ids.size(1);
  TORCH_CHECK(T >= 1 && T <= cfg_.n_ctx, "sequence length ", T, " outside [1, ", cfg_.n_ctx, "]");
  TORCH_CHECK(B <= cache.n_slots(), "batch ", B, " exceeds cache slots ", cache.n_slots());

  torch::InferenceMode guard;
  auto x = torch::embedding(wte_, ids.to(device_)) + torch::embedding(wpe_, positions.to(device_, torch::kInt64));
  const auto allowed = build_attention_mask(T, padding_mask, device_);
  for (int64_t layer = 0; layer < cfg_.n_layer; ++layer) {
    x = blocks_[static_cast<size_t>(layer)].forward_prefill_batch(x, allowed, layer, /*start_pos=*/0, cache);
  }
  x = ln_f_(x.slice(1, T - 1, T));
  return torch::linear(x, wte_).squeeze(1);
}

torch::Tensor GPT2Model::decode_padded(const torch::Tensor& ids, const torch::Tensor& positions,
                                       const torch::Tensor& cache_index, int64_t length, const torch::Tensor& key_mask,
                                       KVCache& cache) const {
  TORCH_CHECK(ids.dim() == 1, "decode_padded expects ids [B], got ", ids.sizes());
  TORCH_CHECK(positions.sizes() == ids.sizes() && cache_index.sizes() == ids.sizes(),
              "positions and cache_index must match ids shape");
  const int64_t B = ids.size(0);
  TORCH_CHECK(B > 0 && B <= cache.n_slots(), "batch ", B, " exceeds cache slots ", cache.n_slots());
  TORCH_CHECK(length >= 1 && length <= cfg_.n_ctx, "length ", length, " outside [1, ", cfg_.n_ctx, "]");
  TORCH_CHECK(key_mask.dim() == 2 && key_mask.size(0) == B && key_mask.size(1) == length,
              "key_mask must be [B, length], got ", key_mask.sizes());

  torch::InferenceMode guard;
  auto x = (torch::embedding(wte_, ids.to(device_)) + torch::embedding(wpe_, positions.to(device_, torch::kInt64)))
               .unsqueeze(1);
  const auto allowed = key_mask.to(device_, torch::kBool).view({B, 1, 1, length});
  const auto dev_index = cache_index.to(device_, torch::kInt64);
  for (int64_t layer = 0; layer < cfg_.n_layer; ++layer) {
    x = blocks_[static_cast<size_t>(layer)].forward_decode(x, dev_index, layer, length, allowed, cache);
  }
  return torch::linear(ln_f_(x), wte_).squeeze(1);
}

torch::Tensor GPT2Model::decode(const torch::Tensor& ids, const torch::Tensor& positions, int64_t length,
                                KVCache& cache) const {
  TORCH_CHECK(ids.dim() == 1, "decode expects ids [B], got ", ids.sizes());
  TORCH_CHECK(ids.scalar_type() == torch::kInt64 && positions.scalar_type() == torch::kInt64, "ids/positions int64");
  TORCH_CHECK(positions.sizes() == ids.sizes(), "positions must match ids shape");
  const int64_t B = ids.size(0);
  TORCH_CHECK(B > 0 && B <= cache.n_slots(), "batch ", B, " exceeds cache slots ", cache.n_slots());
  TORCH_CHECK(length >= 1 && length <= cfg_.n_ctx, "length ", length, " outside [1, ", cfg_.n_ctx, "]");

  torch::InferenceMode guard;
  const auto dev_ids = ids.to(device_);
  const auto dev_pos = positions.to(device_);
  auto x = (torch::embedding(wte_, dev_ids) + torch::embedding(wpe_, dev_pos)).unsqueeze(1);  // [B, 1, D]

  const auto allowed = build_decode_mask(dev_pos, length);
  for (int64_t layer = 0; layer < cfg_.n_layer; ++layer) {
    x = blocks_[static_cast<size_t>(layer)].forward_decode(x, dev_pos, layer, length, allowed, cache);
  }
  return torch::linear(ln_f_(x), wte_).squeeze(1);
}

torch::Tensor GPT2Model::run(const torch::Tensor& ids, const torch::Tensor& positions,
                             const torch::Tensor& padding_mask, std::vector<torch::Tensor>* hidden) const {
  TORCH_CHECK(ids.dim() == 2, "ids must be [B, T], got ", ids.sizes());
  TORCH_CHECK(ids.scalar_type() == torch::kInt64, "ids must be int64");
  const int64_t B = ids.size(0);
  const int64_t T = ids.size(1);
  TORCH_CHECK(T >= 1 && T <= cfg_.n_ctx, "sequence length ", T, " outside [1, ", cfg_.n_ctx, "]");

  torch::InferenceMode guard;
  const auto dev_ids = ids.to(device_);
  torch::Tensor pos;
  if (positions.defined()) {
    TORCH_CHECK(positions.sizes() == ids.sizes(), "positions must match ids shape");
    pos = positions.to(device_, torch::kInt64);
  } else {
    pos = torch::arange(T, torch::TensorOptions().dtype(torch::kInt64).device(device_)).expand({B, T});
  }
  if (padding_mask.defined()) {
    TORCH_CHECK(padding_mask.sizes() == ids.sizes(), "padding_mask must match ids shape");
  }

  auto x = torch::embedding(wte_, dev_ids) + torch::embedding(wpe_, pos);
  const auto allowed = build_attention_mask(T, padding_mask, device_);

  for (const auto& block : blocks_) {
    if (hidden) hidden->push_back(x);
    x = block.forward(x, allowed);
  }
  x = ln_f_(x);
  if (hidden) hidden->push_back(x);
  return torch::linear(x, wte_);
}

}  // namespace gpt2
