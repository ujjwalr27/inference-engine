#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>

#include "scheduler/request.h"

namespace gpt2 {

// Bounded FIFO of waiting requests. Producers are the HTTP threads, the single consumer is
// the scheduler thread. A full queue is how the server answers "busy" instead of piling up work.
class RequestQueue {
 public:
  explicit RequestQueue(size_t capacity) : capacity_(capacity) {}

  // Returns false when the queue is full or closed; the caller then rejects the request.
  bool try_push(std::shared_ptr<Request> request);

  // Returns nullptr when empty.
  std::shared_ptr<Request> try_pop();

  // Blocks until a request arrives, the queue closes, or the timeout expires.
  void wait_for_work(std::chrono::milliseconds timeout);

  // Wakes every waiter and refuses further pushes.
  void close();

  size_t size() const;
  bool closed() const;
  size_t capacity() const { return capacity_; }

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Request>> queue_;
  const size_t capacity_;
  bool closed_ = false;
};

}  // namespace gpt2
