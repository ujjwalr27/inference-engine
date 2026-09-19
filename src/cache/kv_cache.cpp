#include "cache/kv_cache.h"

namespace gpt2 {

using torch::indexing::Slice;

KVCache::KVCache(const GPT2Config& config, int64_t n_slots, torch::Device device, torch::Dtype dtype)
    : n_layer_(config.n_layer),
      n_slots_(n_slots),
      n_head_(config.n_head),
      n_ctx_(config.n_ctx),
      head_dim_(config.head_dim()) {
  TORCH_CHECK(n_slots_ > 0, "n_slots must be positive");
  const auto options = torch::TensorOptions().device(device).dtype(dtype);
  const std::vector<int64_t> shape{n_layer_, n_slots_, n_head_, n_ctx_, head_dim_};
  k_ = torch::zeros(shape, options);
  v_ = torch::zeros(shape, options);
}

int64_t KVCache::bytes() const { return 2 * k_.numel() * k_.element_size(); }

void KVCache::check(int64_t layer, int64_t batch_or_slot, int64_t length) const {
  TORCH_CHECK(layer >= 0 && layer < n_layer_, "layer ", layer, " out of range");
  TORCH_CHECK(batch_or_slot >= 0 && batch_or_slot <= n_slots_, "slot/batch ", batch_or_slot, " out of range");
  TORCH_CHECK(length >= 0 && length <= n_ctx_, "length ", length, " out of range");
}

void KVCache::write_prefill(int64_t layer, int64_t slot, int64_t start_pos, const torch::Tensor& k,
                            const torch::Tensor& v) {
  const int64_t T = k.size(2);
  check(layer, slot, start_pos + T);
  TORCH_CHECK(slot < n_slots_, "slot ", slot, " out of range");
  TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have the same shape");
  TORCH_CHECK(k.size(0) == 1 && k.size(1) == n_head_ && k.size(3) == head_dim_, "unexpected k shape ", k.sizes());

  k_[layer][slot].slice(1, start_pos, start_pos + T).copy_(k[0]);
  v_[layer][slot].slice(1, start_pos, start_pos + T).copy_(v[0]);
}

void KVCache::write_prefill_batch(int64_t layer, int64_t batch, int64_t start_pos, const torch::Tensor& k,
                                  const torch::Tensor& v) {
  const int64_t T = k.size(2);
  check(layer, batch, start_pos + T);
  TORCH_CHECK(batch > 0 && batch <= n_slots_, "batch ", batch, " exceeds slots ", n_slots_);
  TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have the same shape");
  TORCH_CHECK(k.size(0) == batch && k.size(1) == n_head_ && k.size(3) == head_dim_, "unexpected k shape ", k.sizes());

  k_[layer].slice(0, 0, batch).slice(2, start_pos, start_pos + T).copy_(k);
  v_[layer].slice(0, 0, batch).slice(2, start_pos, start_pos + T).copy_(v);
}

void KVCache::write_decode(int64_t layer, int64_t batch, const torch::Tensor& cache_index, const torch::Tensor& k,
                           const torch::Tensor& v) {
  check(layer, batch, 0);
  TORCH_CHECK(batch > 0, "batch must be positive");
  TORCH_CHECK(cache_index.dim() == 1 && cache_index.size(0) == batch, "cache_index must be [batch]");
  TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have the same shape");
  TORCH_CHECK(k.size(0) == batch && k.size(1) == n_head_ && k.size(2) == 1 && k.size(3) == head_dim_,
              "unexpected k shape ", k.sizes());

  // Advanced indexing with a slice between two index tensors puts the indexed dims first: [batch, H, head_dim].
  const auto slots = torch::arange(batch, cache_index.options());
  k_[layer].index_put_({slots, Slice(), cache_index}, k.squeeze(2));
  v_[layer].index_put_({slots, Slice(), cache_index}, v.squeeze(2));
}

torch::Tensor KVCache::keys(int64_t layer, int64_t batch, int64_t length) const {
  check(layer, batch, length);
  return k_[layer].slice(0, 0, batch).slice(2, 0, length);
}

torch::Tensor KVCache::values(int64_t layer, int64_t batch, int64_t length) const {
  check(layer, batch, length);
  return v_[layer].slice(0, 0, batch).slice(2, 0, length);
}

void KVCache::copy_slot(int64_t from, int64_t to, int64_t length) {
  check(0, from, length);
  TORCH_CHECK(from >= 0 && from < n_slots_ && to >= 0 && to < n_slots_, "slot out of range");
  if (from == to || length == 0) return;
  k_.select(1, to).slice(2, 0, length).copy_(k_.select(1, from).slice(2, 0, length));
  v_.select(1, to).slice(2, 0, length).copy_(v_.select(1, from).slice(2, 0, length));
}

void KVCache::clear() {
  k_.zero_();
  v_.zero_();
}

}  // namespace gpt2
