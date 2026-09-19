#pragma once

#include <string>
#include <vector>

#include <torch/torch.h>

#include "cache/kv_cache.h"
#include "io/safetensors.h"
#include "model/config.h"
#include "model/layers.h"

namespace gpt2 {

class GPT2Model {
 public:
  // Loads <dir>/config.json and <dir>/model.safetensors (written by scripts/export_weights.py).
  static GPT2Model load(const std::string& dir, torch::Device device = torch::kCPU,
                        torch::Dtype dtype = torch::kFloat32);

  GPT2Model(const GPT2Config& config, const SafeTensors& weights, torch::Device device, torch::Dtype dtype);

  const GPT2Config& config() const { return cfg_; }
  torch::Device device() const { return device_; }
  torch::Dtype dtype() const { return dtype_; }

  // Uncached full forward pass.
  //   ids:          [B, T] int64
  //   positions:    [B, T] int64, undefined = 0..T-1 for every row
  //   padding_mask: [B, T] bool, true = real token, undefined = no padding
  // Returns logits [B, T, vocab] in the model dtype.
  torch::Tensor forward(const torch::Tensor& ids, const torch::Tensor& positions = {},
                        const torch::Tensor& padding_mask = {}) const;

  // Same pass, returning hidden states laid out like Hugging Face's output_hidden_states:
  // [0] = embeddings, [i] = input to block i (i < n_layer), [n_layer] = ln_f(last block output).
  std::vector<torch::Tensor> hidden_states(const torch::Tensor& ids, const torch::Tensor& positions = {},
                                           const torch::Tensor& padding_mask = {}) const;

  // Cached prefill for one sequence: runs ids [1, T] at positions [start_pos, start_pos + T),
  // fills cache slot `slot`, and returns logits for the LAST position only, [1, vocab].
  torch::Tensor prefill(const torch::Tensor& ids, int64_t slot, KVCache& cache, int64_t start_pos = 0) const;

  // Padded prefill for static batching: ids/positions/padding_mask are [B, T] (left padded,
  // see scheduler/padding.h) and fill slots 0..B-1 keeping the padded layout.
  // Returns logits at the last column, [B, vocab] - valid for every row because padding is on the left.
  torch::Tensor prefill_batch(const torch::Tensor& ids, const torch::Tensor& positions,
                              const torch::Tensor& padding_mask, KVCache& cache) const;

  // Decode step where a token's cache slot differs from its position id (static batching):
  //   cache_index: [B] int64, where this token is stored
  //   key_mask:    [B, length] bool, which cached columns each row may attend to (pads excluded)
  torch::Tensor decode_padded(const torch::Tensor& ids, const torch::Tensor& positions,
                              const torch::Tensor& cache_index, int64_t length, const torch::Tensor& key_mask,
                              KVCache& cache) const;

  // Cached decode step for slots 0..B-1 (one new token each).
  //   ids:       [B] int64, the newest token per row
  //   positions: [B] int64, that token's position in its own sequence
  //   length:    max(positions) + 1, passed in by the caller so no GPU->CPU sync is needed here
  // Returns logits [B, vocab].
  torch::Tensor decode(const torch::Tensor& ids, const torch::Tensor& positions, int64_t length,
                       KVCache& cache) const;

 private:
  torch::Tensor run(const torch::Tensor& ids, const torch::Tensor& positions, const torch::Tensor& padding_mask,
                    std::vector<torch::Tensor>* hidden) const;

  GPT2Config cfg_;
  torch::Device device_;
  torch::Dtype dtype_;
  torch::Tensor wte_;  // [vocab, D], also the tied LM head
  torch::Tensor wpe_;  // [n_ctx, D]
  std::vector<Block> blocks_;
  LayerNorm ln_f_;
};

}  // namespace gpt2
