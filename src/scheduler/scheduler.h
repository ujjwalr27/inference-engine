#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "cache/kv_cache.h"
#include "model/decode_graphs.h"
#include "model/generate.h"
#include "model/gpt2.h"
#include "scheduler/queue.h"
#include "scheduler/request.h"

namespace gpt2 {

enum class BatchingPolicy {
  // Requests join and leave between decode steps; a free slot is refilled immediately.
  Continuous,
  // A batch is formed, run to completion, and only then is the next one admitted. This is the
  // baseline continuous batching is measured against: a request that arrives one step too late
  // waits for the whole batch, and a request that finishes early leaves its slot idle.
  Static,
};

const char* to_string(BatchingPolicy policy);
BatchingPolicy parse_policy(const std::string& name);  // throws std::invalid_argument

struct SchedulerOptions {
  BatchingPolicy policy = BatchingPolicy::Continuous;
  int64_t n_slots = 8;                // concurrent requests held in the KV cache
  size_t max_queue = 64;              // waiting requests before new ones are rejected
  int64_t prefill_budget_tokens = 512;  // prompt tokens admitted per step; caps the decode pause

  // Prefill the requests admitted in one step together, in packed passes of about
  // prefill_budget_tokens each (GPT2Model::prefill_packed), instead of one pass per request.
  bool batch_prefill = true;
  std::chrono::milliseconds idle_wait{20};

  // Static policy only: how many requests to gather, and how long to wait for them before
  // running a smaller batch. 0 means "as many as there are slots".
  int64_t static_batch_size = 0;
  std::chrono::milliseconds static_max_wait{50};

  // Replay decode steps from CUDA graphs (model/decode_graphs.h). CUDA only; the graphs are
  // captured when the scheduler is constructed, so construction takes a second or two longer.
  bool cuda_graphs = false;
  // Cache lengths are rounded up to a multiple of this to pick a graph. Smaller steps waste less
  // attention on padding but capture more graphs (6 batch buckets x 1024 / step at 32 slots).
  int64_t graph_length_step = 64;
};

struct SchedulerStats {
  uint64_t submitted = 0;
  uint64_t rejected = 0;    // queue was full
  uint64_t admitted = 0;    // prefilled and given a slot
  uint64_t finished = 0;
  uint64_t cancelled = 0;
  uint64_t decode_steps = 0;
  uint64_t generated_tokens = 0;
  int64_t max_batch = 0;    // largest number of slots busy at once
  int64_t active = 0;
  size_t queued = 0;
  uint64_t batches = 0;     // static policy: batches formed so far

  // Where the scheduler thread's time goes, cumulative, in milliseconds. Prefill and decode each
  // end in a device->host copy that waits for the GPU, so they include the GPU work itself, not
  // just the kernel launches. busy_ms - prefill_ms - decode_ms is bookkeeping: admitting,
  // publishing tokens, finishing requests and compacting their slots.
  double prefill_ms = 0;        // prompt forward passes, first token included
  uint64_t prefill_tokens = 0;
  uint64_t prefill_passes = 0;  // forward passes those prompts took; fewer than admitted when batched
  double decode_ms = 0;         // decode steps: inputs in, forward, argmax, tokens out
  double busy_ms = 0;           // every loop iteration except the time spent idle or gathering
};

// Continuous batching. One thread owns the model and the KV cache, so nothing else locks them.
// Each step: drop finished requests (compacting their slots away), admit waiting ones within the
// prefill budget, then run a single decode step for every active request together.
class Scheduler {
 public:
  Scheduler(const GPT2Model& model, const SchedulerOptions& options);
  ~Scheduler();

  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;

  void start();
  void stop();  // closes the queue, drains, joins; safe to call twice

  // Returns nullptr when the queue is full. Throws std::invalid_argument for a request that
  // can never run (empty prompt, or prompt + max_new_tokens past the context window).
  std::shared_ptr<Request> submit(std::vector<int64_t> prompt, const GenerationOptions& options);

  SchedulerStats stats() const;
  const SchedulerOptions& options() const { return options_; }

 private:
  void run();
  void admit();
  // How many requests may be admitted right now, and whether the prefill budget applies.
  // Returns 0 when the policy says to wait.
  size_t admission_limit(bool& apply_budget);
  // Prefills these requests in one pass, gives them the next slots, and publishes their first tokens.
  void prefill(const std::vector<std::shared_ptr<Request>>& requests);
  void step();
  void finish(size_t index, FinishReason reason);  // publishes, then frees the slot
  void release_slot(size_t index);                 // compacts the last active slot into this one

  const GPT2Model& model_;
  SchedulerOptions options_;
  KVCache cache_;
  RequestQueue queue_;
  std::unique_ptr<DecodeGraphs> graphs_;  // set when options_.cuda_graphs

  std::thread thread_;
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> next_id_{1};

  // Scheduler-thread only: active_[i]->slot == i for every i.
  std::vector<std::shared_ptr<Request>> active_;

  // Staging buffers for a decode step, allocated once. Pinned when the model is on CUDA, so the
  // per-step host->device copy can overlap instead of going through a pageable bounce buffer.
  // Safe to refill every step because the argmax copy back to the host synchronises first.
  torch::Tensor ids_host_;
  torch::Tensor positions_host_;

  // Static policy: when the current batch started gathering.
  std::optional<SchedClock::time_point> batch_wait_start_;

  mutable std::mutex stats_mutex_;
  SchedulerStats stats_;
};

}  // namespace gpt2
