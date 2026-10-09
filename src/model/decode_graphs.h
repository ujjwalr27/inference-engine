#pragma once

#include <cstdint>
#include <memory>

#include <torch/torch.h>

#include "cache/kv_cache.h"
#include "model/gpt2.h"

namespace gpt2 {

// Replays the decode step from CUDA graphs instead of launching its ~200 kernels one by one.
//
// A decode step for GPT-2 small is launch-bound: one row or sixteen, it takes ~5 ms on a T4,
// almost all of it the CPU issuing kernels. A CUDA graph records the whole step once and replays
// it with a single launch. Graphs need fixed shapes and addresses, so each step is padded up to a
// bucket: the batch to the next power of two (capped at the slot count), the attention length to
// the next multiple of `length_step`. Padding is harmless by construction:
//  - extra keys sit past every row's position, so the decode mask hides them;
//  - padding rows (token 0, position 0) only write into slots no active request occupies, at
//    position 0, which a new request's prefill overwrites before anything reads it.
//
// All graphs are captured up front (capture_all) and share one memory pool, which is safe
// because they never run concurrently and each output is consumed before the next replay.
class DecodeGraphs {
 public:
  // Throws std::runtime_error when the model is not on CUDA or LibTorch was built without CUDA.
  DecodeGraphs(const GPT2Model& model, KVCache& cache, int64_t length_step = 64);
  ~DecodeGraphs();

  DecodeGraphs(const DecodeGraphs&) = delete;
  DecodeGraphs& operator=(const DecodeGraphs&) = delete;

  // Captures every (batch bucket, length bucket) graph. Writes to the KV cache at position 0 of
  // its slots, so call it before any request is admitted. Returns the number of graphs.
  int64_t capture_all();

  // One decode step for rows [0, batch): ids and positions are int64 [>= batch] host tensors
  // (pinned for an asynchronous copy). Returns logits [batch, vocab], a view of the graph's output
  // that stays valid until the next run().
  torch::Tensor run(const torch::Tensor& ids_host, const torch::Tensor& positions_host, int64_t batch,
                    int64_t length);

  int64_t batch_bucket(int64_t batch) const;
  int64_t length_bucket(int64_t length) const;

  static bool available();  // compiled against a CUDA build of LibTorch

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpt2
