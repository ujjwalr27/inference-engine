#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "model/generate.h"

namespace gpt2 {

using SchedClock = std::chrono::steady_clock;

// One request's token stream. The scheduler thread pushes; the requesting thread waits.
// This is the only object shared between them, so it carries its own lock.
class TokenChannel {
 public:
  void push(int64_t token);

  // Closes the stream. A non-empty message marks it as failed.
  void finish(std::string error = {});

  // Waits for the next token. Returns false once the stream is closed and drained.
  bool next(int64_t& token);

  // Waits until the stream closes and returns every token.
  std::vector<int64_t> collect();

  bool closed() const;
  std::string error() const;

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<int64_t> tokens_;
  bool closed_ = false;
  std::string error_;
};

enum class FinishReason { None, EosToken, MaxTokens, ContextFull, Cancelled, Error };

const char* to_string(FinishReason reason);

struct Request {
  // Set at submit time, then read-only.
  uint64_t id = 0;
  std::vector<int64_t> prompt;
  GenerationOptions options;
  std::shared_ptr<TokenChannel> out = std::make_shared<TokenChannel>();

  // Written by any thread, read by the scheduler.
  std::atomic<bool> cancelled{false};
  void cancel() { cancelled.store(true, std::memory_order_relaxed); }
  bool is_cancelled() const { return cancelled.load(std::memory_order_relaxed); }

  // Scheduler-thread state. Nothing else touches these.
  std::vector<int64_t> generated;
  int64_t slot = -1;         // index into the active list; always equals its cache slot
  int64_t position = 0;      // position the next fed token will occupy (= cached length)
  int64_t next_token = -1;   // token to feed at the next decode step
  FinishReason finish_reason = FinishReason::None;

  SchedClock::time_point t_enqueue;
  SchedClock::time_point t_prefill_start;
  SchedClock::time_point t_first_token;
  SchedClock::time_point t_done;
};

}  // namespace gpt2
