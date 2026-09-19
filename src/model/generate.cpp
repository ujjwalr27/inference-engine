#include "model/generate.h"

#include <chrono>
#include <memory>
#include <stdexcept>

namespace gpt2 {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int64_t eos_of(const GPT2Model& model, const GenerationOptions& options) {
  return options.eos_token_id >= 0 ? options.eos_token_id : model.config().eos_token_id;
}

void check_room(const GPT2Model& model, size_t prompt_len, int64_t max_new_tokens) {
  if (prompt_len == 0) throw std::invalid_argument("prompt must not be empty");
  if (max_new_tokens < 0) throw std::invalid_argument("max_new_tokens must not be negative");
  const int64_t total = static_cast<int64_t>(prompt_len) + max_new_tokens;
  if (total > model.config().n_ctx) {
    throw std::invalid_argument("prompt (" + std::to_string(prompt_len) + ") + max_new_tokens (" +
                                std::to_string(max_new_tokens) + ") exceeds context " +
                                std::to_string(model.config().n_ctx));
  }
}

int64_t argmax_id(const torch::Tensor& logits_row) { return logits_row.argmax(-1).item<int64_t>(); }

}  // namespace

GenerationResult generate_cached(const GPT2Model& model, const std::vector<int64_t>& prompt,
                                 const GenerationOptions& options, KVCache* cache, const TokenCallback& on_token) {
  check_room(model, prompt.size(), options.max_new_tokens);
  torch::InferenceMode guard;

  std::unique_ptr<KVCache> owned;
  if (cache == nullptr) {
    owned = std::make_unique<KVCache>(model.config(), 1, model.device(), model.dtype());
    cache = owned.get();
  }

  GenerationResult result;
  const int64_t eos = eos_of(model, options);
  const auto ids = torch::tensor(prompt, torch::kInt64).unsqueeze(0);

  auto t0 = Clock::now();
  auto logits = model.prefill(ids, /*slot=*/0, *cache);
  int64_t next = argmax_id(logits[0]);
  result.stats.prefill_ms = ms_since(t0);

  int64_t position = static_cast<int64_t>(prompt.size());  // position of the token about to be fed back
  t0 = Clock::now();
  for (int64_t step = 0; step < options.max_new_tokens; ++step) {
    result.tokens.push_back(next);
    ++result.stats.steps;
    if (on_token) on_token(next);
    if (options.stop_on_eos && next == eos) {
      result.hit_eos = true;
      break;
    }
    if (step + 1 == options.max_new_tokens) break;

    const auto step_ids = torch::tensor({next}, torch::kInt64);
    const auto step_pos = torch::tensor({position}, torch::kInt64);
    logits = model.decode(step_ids, step_pos, position + 1, *cache);
    next = argmax_id(logits[0]);
    ++position;
  }
  result.stats.decode_ms = ms_since(t0);
  return result;
}

GenerationResult generate_uncached(const GPT2Model& model, const std::vector<int64_t>& prompt,
                                   const GenerationOptions& options, const TokenCallback& on_token) {
  check_room(model, prompt.size(), options.max_new_tokens);
  torch::InferenceMode guard;

  GenerationResult result;
  const int64_t eos = eos_of(model, options);
  std::vector<int64_t> sequence = prompt;

  auto t0 = Clock::now();
  auto logits = model.forward(torch::tensor(sequence, torch::kInt64).unsqueeze(0));
  int64_t next = argmax_id(logits[0][-1]);
  result.stats.prefill_ms = ms_since(t0);

  t0 = Clock::now();
  for (int64_t step = 0; step < options.max_new_tokens; ++step) {
    result.tokens.push_back(next);
    ++result.stats.steps;
    if (on_token) on_token(next);
    if (options.stop_on_eos && next == eos) {
      result.hit_eos = true;
      break;
    }
    if (step + 1 == options.max_new_tokens) break;

    sequence.push_back(next);
    logits = model.forward(torch::tensor(sequence, torch::kInt64).unsqueeze(0));
    next = argmax_id(logits[0][-1]);
  }
  result.stats.decode_ms = ms_since(t0);
  return result;
}

GenerationResult generate_greedy(const GPT2Model& model, const std::vector<int64_t>& prompt,
                                 const GenerationOptions& options, KVCache* cache, const TokenCallback& on_token) {
  return options.use_cache ? generate_cached(model, prompt, options, cache, on_token)
                           : generate_uncached(model, prompt, options, on_token);
}

}  // namespace gpt2
