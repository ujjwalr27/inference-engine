#pragma once

#include <cstdint>
#include <tuple>

#include <torch/torch.h>

#include "cache/kv_cache.h"

namespace gpt2 {

// GPT-2's activation ("gelu_new"): the tanh approximation of GELU.
// Using the exact erf GELU instead drifts logits past a 1e-4 tolerance over 12 layers.
torch::Tensor gelu_new(const torch::Tensor& x);

// Most negative finite value for a floating dtype. Used to mask attention scores:
// never -inf (NaN on fully masked rows) and never a hard-coded -1e9 (overflows fp16).
double mask_fill_value(torch::Dtype dtype);

// Boolean mask of allowed attention, broadcastable to [B, n_head, T, T]:
// allowed[b, :, i, j] = (j <= i) && padding_mask[b, j].
// padding_mask: [B, T] bool, true = real token; undefined = no padding.
torch::Tensor build_attention_mask(int64_t seq_len, const torch::Tensor& padding_mask, torch::Device device);

// Decode-step mask [B, 1, 1, length]: row i may attend to keys 0..positions[i].
// Returns an undefined tensor when every row attends to the whole length (nothing to mask).
torch::Tensor build_decode_mask(const torch::Tensor& positions, int64_t length);

struct Linear {
  torch::Tensor weight;  // [out, in]
  torch::Tensor bias;    // [out]
  torch::Tensor operator()(const torch::Tensor& x) const { return torch::linear(x, weight, bias); }
};

struct LayerNorm {
  torch::Tensor weight;
  torch::Tensor bias;
  double eps = 1e-5;
  torch::Tensor operator()(const torch::Tensor& x) const;
};

struct Attention {
  Linear q_proj, k_proj, v_proj, c_proj;
  int64_t n_head = 12;

  // Uncached: x [B, T, D], allowed broadcastable to [B, n_head, T, T]. Returns [B, T, D].
  torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& allowed) const;

  // Prefill: as above for one sequence (B = 1), but keys/values are also written to
  // cache[layer][slot] at positions [start_pos, start_pos + T).
  torch::Tensor forward_prefill(const torch::Tensor& x, const torch::Tensor& allowed, int64_t layer, int64_t slot,
                                int64_t start_pos, KVCache& cache) const;

  // Padded prefill for slots 0..B-1 (static batching).
  torch::Tensor forward_prefill_batch(const torch::Tensor& x, const torch::Tensor& allowed, int64_t layer,
                                      int64_t start_pos, KVCache& cache) const;

  // Decode: x [B, 1, D] for slots 0..B-1. Writes this step's key/value at cache_index[i],
  // then attends over cached positions [0, length). allowed may be undefined (nothing masked).
  torch::Tensor forward_decode(const torch::Tensor& x, const torch::Tensor& cache_index, int64_t layer,
                               int64_t length, const torch::Tensor& allowed, KVCache& cache) const;

 private:
  // q, k, v shaped [B, T, n_head, head_dim] transposed to [B, n_head, T, head_dim].
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> project(const torch::Tensor& x) const;
  torch::Tensor attend(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                       const torch::Tensor& allowed) const;
};

struct MLP {
  Linear c_fc, c_proj;
  torch::Tensor forward(const torch::Tensor& x) const;
};

struct Block {
  LayerNorm ln_1;
  Attention attn;
  LayerNorm ln_2;
  MLP mlp;

  // Pre-norm residual block: x + attn(ln_1(x)), then + mlp(ln_2(x)).
  torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& allowed) const;
  torch::Tensor forward_prefill(const torch::Tensor& x, const torch::Tensor& allowed, int64_t layer, int64_t slot,
                                int64_t start_pos, KVCache& cache) const;
  torch::Tensor forward_prefill_batch(const torch::Tensor& x, const torch::Tensor& allowed, int64_t layer,
                                      int64_t start_pos, KVCache& cache) const;
  torch::Tensor forward_decode(const torch::Tensor& x, const torch::Tensor& cache_index, int64_t layer,
                               int64_t length, const torch::Tensor& allowed, KVCache& cache) const;
};

}  // namespace gpt2
