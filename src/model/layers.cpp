#include "model/layers.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace gpt2 {

torch::Tensor gelu_new(const torch::Tensor& x) {
  static const double kScale = std::sqrt(2.0 / 3.14159265358979323846);
  return 0.5 * x * (1.0 + torch::tanh(kScale * (x + 0.044715 * torch::pow(x, 3.0))));
}

double mask_fill_value(torch::Dtype dtype) {
  if (dtype == torch::kFloat32) return std::numeric_limits<float>::lowest();
  if (dtype == torch::kFloat64) return std::numeric_limits<double>::lowest();
  if (dtype == torch::kFloat16) return -65504.0;
  if (dtype == torch::kBFloat16) return -3.3895313892515355e38;
  throw std::invalid_argument("mask_fill_value: not a floating dtype");
}

torch::Tensor build_attention_mask(int64_t seq_len, const torch::Tensor& padding_mask, torch::Device device) {
  auto allowed = torch::ones({seq_len, seq_len}, torch::TensorOptions().dtype(torch::kBool).device(device))
                     .tril()
                     .view({1, 1, seq_len, seq_len});
  if (padding_mask.defined()) {
    TORCH_CHECK(padding_mask.dim() == 2 && padding_mask.size(1) == seq_len,
                "padding_mask must be [B, T] with T = ", seq_len);
    allowed = allowed & padding_mask.to(device, torch::kBool).view({padding_mask.size(0), 1, 1, seq_len});
  }
  return allowed;
}

torch::Tensor build_decode_mask(const torch::Tensor& positions, int64_t length) {
  TORCH_CHECK(positions.dim() == 1, "positions must be [batch]");
  if (positions.size(0) == 1) return {};  // the single row always covers the whole cached length
  const auto keys = torch::arange(length, positions.options());
  return (keys.view({1, 1, 1, length}) <= positions.view({positions.size(0), 1, 1, 1}));
}

torch::Tensor LayerNorm::operator()(const torch::Tensor& x) const {
  return torch::layer_norm(x, {weight.size(0)}, weight, bias, eps);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Attention::project(const torch::Tensor& x) const {
  const int64_t B = x.size(0);
  const int64_t T = x.size(1);
  const int64_t head_dim = x.size(2) / n_head;
  auto split_heads = [&](const Linear& proj) {
    return proj(x).view({B, T, n_head, head_dim}).transpose(1, 2);  // [B, H, T, hd]
  };
  return {split_heads(q_proj), split_heads(k_proj), split_heads(v_proj)};
}

torch::Tensor Attention::attend(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                                const torch::Tensor& allowed) const {
  const int64_t B = q.size(0);
  const int64_t T = q.size(2);
  const int64_t head_dim = q.size(3);

  auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<double>(head_dim));
  if (allowed.defined()) {
    scores.masked_fill_(allowed.logical_not(), mask_fill_value(scores.scalar_type()));
  }
  auto out = torch::matmul(torch::softmax(scores, -1), v);  // [B, H, T, hd]
  return c_proj(out.transpose(1, 2).reshape({B, T, n_head * head_dim}));
}

torch::Tensor Attention::forward(const torch::Tensor& x, const torch::Tensor& allowed) const {
  auto [q, k, v] = project(x);
  return attend(q, k, v, allowed);
}

torch::Tensor Attention::forward_prefill(const torch::Tensor& x, const torch::Tensor& allowed, int64_t layer,
                                         int64_t slot, int64_t start_pos, KVCache& cache) const {
  TORCH_CHECK(x.size(0) == 1, "prefill handles one sequence at a time");
  auto [q, k, v] = project(x);
  cache.write_prefill(layer, slot, start_pos, k, v);
  return attend(q, k, v, allowed);
}

torch::Tensor Attention::forward_prefill_batch(const torch::Tensor& x, const torch::Tensor& allowed, int64_t layer,
                                               int64_t start_pos, KVCache& cache) const {
  auto [q, k, v] = project(x);
  cache.write_prefill_batch(layer, x.size(0), start_pos, k, v);
  return attend(q, k, v, allowed);
}

torch::Tensor Attention::forward_decode(const torch::Tensor& x, const torch::Tensor& cache_index, int64_t layer,
                                        int64_t length, const torch::Tensor& allowed, KVCache& cache) const {
  TORCH_CHECK(x.size(1) == 1, "decode consumes exactly one token per row");
  const int64_t batch = x.size(0);
  auto [q, k, v] = project(x);
  cache.write_decode(layer, batch, cache_index, k, v);
  return attend(q, cache.keys(layer, batch, length), cache.values(layer, batch, length), allowed);
}

torch::Tensor MLP::forward(const torch::Tensor& x) const { return c_proj(gelu_new(c_fc(x))); }

torch::Tensor Block::forward(const torch::Tensor& x, const torch::Tensor& allowed) const {
  auto h = x + attn.forward(ln_1(x), allowed);
  return h + mlp.forward(ln_2(h));
}

torch::Tensor Block::forward_prefill(const torch::Tensor& x, const torch::Tensor& allowed, int64_t layer, int64_t slot,
                                     int64_t start_pos, KVCache& cache) const {
  auto h = x + attn.forward_prefill(ln_1(x), allowed, layer, slot, start_pos, cache);
  return h + mlp.forward(ln_2(h));
}

torch::Tensor Block::forward_prefill_batch(const torch::Tensor& x, const torch::Tensor& allowed, int64_t layer,
                                           int64_t start_pos, KVCache& cache) const {
  auto h = x + attn.forward_prefill_batch(ln_1(x), allowed, layer, start_pos, cache);
  return h + mlp.forward(ln_2(h));
}

torch::Tensor Block::forward_decode(const torch::Tensor& x, const torch::Tensor& cache_index, int64_t layer,
                                    int64_t length, const torch::Tensor& allowed, KVCache& cache) const {
  auto h = x + attn.forward_decode(ln_1(x), cache_index, layer, length, allowed, cache);
  return h + mlp.forward(ln_2(h));
}

}  // namespace gpt2
