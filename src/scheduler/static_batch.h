#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "cache/kv_cache.h"
#include "model/generate.h"
#include "model/gpt2.h"

namespace gpt2 {

struct StaticBatchResult {
  std::vector<std::vector<int64_t>> tokens;  // generated tokens per row
  std::vector<bool> hit_eos;
  GenerationStats stats;
  double padding_waste = 0.0;       // share of the padded prompt block that was padding
  int64_t wasted_decode_rows = 0;   // row-steps spent on rows that had already finished
};

// Called with (row, token) as each token is produced.
using BatchTokenCallback = std::function<void(size_t, int64_t)>;

// Static batching: pad the prompts, prefill them together, then decode the whole batch in
// lockstep until every row is done. Rows that finish early keep occupying the batch - that
// waste is what continuous batching (Phase 4) removes, and both numbers are reported here.
StaticBatchResult generate_batch_static(const GPT2Model& model, const std::vector<std::vector<int64_t>>& prompts,
                                        const GenerationOptions& options, KVCache* cache = nullptr,
                                        const BatchTokenCallback& on_token = {});

}  // namespace gpt2
