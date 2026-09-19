#include "scheduler/static_batch.h"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

#include "scheduler/padding.h"

namespace gpt2 {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

StaticBatchResult generate_batch_static(const GPT2Model& model, const std::vector<std::vector<int64_t>>& prompts,
                                        const GenerationOptions& options, KVCache* cache,
                                        const BatchTokenCallback& on_token) {
  if (prompts.empty()) throw std::invalid_argument("no prompts");
  if (options.max_new_tokens < 1) throw std::invalid_argument("max_new_tokens must be at least 1");

  const auto batch = pad_left(prompts, model.config().eos_token_id);
  const int64_t B = batch.batch_size();
  const int64_t T = batch.max_length;
  if (T + options.max_new_tokens > model.config().n_ctx) {
    throw std::invalid_argument("longest prompt (" + std::to_string(T) + ") + max_new_tokens (" +
                                std::to_string(options.max_new_tokens) + ") exceeds context " +
                                std::to_string(model.config().n_ctx));
  }

  torch::InferenceMode guard;
  std::unique_ptr<KVCache> owned;
  if (cache == nullptr) {
    owned = std::make_unique<KVCache>(model.config(), B, model.device(), model.dtype());
    cache = owned.get();
  } else if (cache->n_slots() < B) {
    throw std::invalid_argument("cache has fewer slots than the batch size");
  }

  StaticBatchResult result;
  result.tokens.resize(static_cast<size_t>(B));
  result.hit_eos.assign(static_cast<size_t>(B), false);
  result.padding_waste = batch.padding_waste();

  const int64_t eos = options.eos_token_id >= 0 ? options.eos_token_id : model.config().eos_token_id;
  const auto lengths = torch::tensor(batch.lengths, torch::kInt64);
  std::vector<bool> finished(static_cast<size_t>(B), false);

  auto t0 = Clock::now();
  auto logits = model.prefill_batch(batch.ids, batch.positions, batch.padding_mask, *cache);
  auto next = logits.argmax(-1);  // [B], kept on the device
  result.stats.prefill_ms = ms_since(t0);

  t0 = Clock::now();
  for (int64_t step = 0; step < options.max_new_tokens; ++step) {
    const auto next_cpu = next.to(torch::kCPU, torch::kInt64).contiguous();  // one transfer per step
    const auto acc = next_cpu.accessor<int64_t, 1>();
    int64_t still_running = 0;
    for (int64_t b = 0; b < B; ++b) {
      const auto row = static_cast<size_t>(b);
      if (finished[row]) {
        ++result.wasted_decode_rows;
        continue;
      }
      const int64_t token = acc[b];
      result.tokens[row].push_back(token);
      if (on_token) on_token(row, token);
      if (options.stop_on_eos && token == eos) {
        finished[row] = true;
        result.hit_eos[row] = true;
      } else {
        ++still_running;
      }
    }
    ++result.stats.steps;
    if (still_running == 0 || step + 1 == options.max_new_tokens) break;

    // Cached columns: the padded prompt block, then one column per generated token so far.
    const auto generated_mask = torch::ones({B, step + 1}, torch::kBool);
    const auto key_mask = torch::cat({batch.padding_mask, generated_mask}, 1);
    const auto positions = lengths + step;                              // position id inside each row
    const auto cache_index = torch::full({B}, T + step, torch::kInt64);  // shared column in the padded cache

    logits = model.decode_padded(next, positions, cache_index, T + step + 1, key_mask, *cache);
    next = logits.argmax(-1);
  }
  result.stats.decode_ms = ms_since(t0);
  return result;
}

}  // namespace gpt2
