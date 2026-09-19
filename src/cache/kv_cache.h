#pragma once

#include <cstdint>

#include <torch/torch.h>

#include "model/config.h"

namespace gpt2 {

// Preallocated key/value cache: two tensors of [n_layer, n_slots, n_head, n_ctx, head_dim].
// Every active request owns one slot. Nothing is allocated after construction.
//
// Slot convention (kept from Phase 2 onward): a decode step covers slots 0..batch-1,
// so the scheduler compacts active requests to the front and reads a view, not a copy.
class KVCache {
 public:
  KVCache(const GPT2Config& config, int64_t n_slots, torch::Device device, torch::Dtype dtype);

  int64_t n_slots() const { return n_slots_; }
  int64_t n_ctx() const { return n_ctx_; }
  int64_t bytes() const;

  // Prefill: k, v are [1, H, T, head_dim] for one slot, written at positions [start_pos, start_pos + T).
  void write_prefill(int64_t layer, int64_t slot, int64_t start_pos, const torch::Tensor& k, const torch::Tensor& v);

  // Padded prefill (static batching): k, v are [batch, H, T, head_dim] for slots 0..batch-1.
  // Rows keep their padded layout, so padded columns occupy cache positions and must stay masked.
  void write_prefill_batch(int64_t layer, int64_t batch, int64_t start_pos, const torch::Tensor& k,
                           const torch::Tensor& v);

  // Decode: k, v are [batch, H, 1, head_dim] for slots 0..batch-1, written at cache_index[batch] (int64).
  // cache_index is where the token lands in the cache, which equals its position id only when the
  // row has no padding in front of it (Phase 2/4); static batching passes a different index.
  void write_decode(int64_t layer, int64_t batch, const torch::Tensor& cache_index, const torch::Tensor& k,
                    const torch::Tensor& v);

  // Views (no copy) of slots 0..batch-1 over positions [0, length): [batch, H, length, head_dim].
  torch::Tensor keys(int64_t layer, int64_t batch, int64_t length) const;
  torch::Tensor values(int64_t layer, int64_t batch, int64_t length) const;

  // Moves a slot's first `length` positions onto another slot. This is how the scheduler
  // compacts active requests to the front after one finishes, so decode keeps reading a view.
  void copy_slot(int64_t from, int64_t to, int64_t length);

  // Zeroes everything. Only needed by tests; serving reuses slots without clearing.
  void clear();

 private:
  void check(int64_t layer, int64_t batch_or_slot, int64_t length) const;

  int64_t n_layer_, n_slots_, n_head_, n_ctx_, head_dim_;
  torch::Tensor k_;  // [L, S, H, n_ctx, hd]
  torch::Tensor v_;
};

}  // namespace gpt2
