#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "cache/kv_cache.h"
#include "model/gpt2.h"

namespace gpt2 {

struct GenerationOptions {
  int64_t max_new_tokens = 32;
  bool use_cache = true;
  bool stop_on_eos = true;
  int64_t eos_token_id = -1;  // -1 = take it from the model config
};

struct GenerationStats {
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  int64_t steps = 0;  // number of generated tokens
  double ms_per_token() const { return steps > 0 ? decode_ms / static_cast<double>(steps) : 0.0; }
};

struct GenerationResult {
  std::vector<int64_t> tokens;  // generated tokens only, prompt not included
  GenerationStats stats;
  bool hit_eos = false;
};

// Called with each new token as soon as it is produced (used for streaming).
using TokenCallback = std::function<void(int64_t)>;

// Greedy generation with the KV cache: prefill the prompt, then one token per decode step.
// `cache` may be null, in which case a single-slot cache is allocated for this call.
GenerationResult generate_cached(const GPT2Model& model, const std::vector<int64_t>& prompt,
                                 const GenerationOptions& options, KVCache* cache = nullptr,
                                 const TokenCallback& on_token = {});

// Reference implementation: re-runs the whole sequence through the model every step.
// Slow by design; it exists to prove the cached path returns identical results.
GenerationResult generate_uncached(const GPT2Model& model, const std::vector<int64_t>& prompt,
                                   const GenerationOptions& options, const TokenCallback& on_token = {});

// Dispatches on options.use_cache.
GenerationResult generate_greedy(const GPT2Model& model, const std::vector<int64_t>& prompt,
                                 const GenerationOptions& options, KVCache* cache = nullptr,
                                 const TokenCallback& on_token = {});

}  // namespace gpt2
