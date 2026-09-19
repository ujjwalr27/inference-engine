#include "scheduler/queue.h"

namespace gpt2 {

bool RequestQueue::try_push(std::shared_ptr<Request> request) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || queue_.size() >= capacity_) return false;
    queue_.push_back(std::move(request));
  }
  cv_.notify_one();
  return true;
}

std::shared_ptr<Request> RequestQueue::try_pop() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (queue_.empty()) return nullptr;
  auto request = std::move(queue_.front());
  queue_.pop_front();
  return request;
}

void RequestQueue::wait_for_work(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait_for(lock, timeout, [this] { return !queue_.empty() || closed_; });
}

void RequestQueue::close() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
  }
  cv_.notify_all();
}

size_t RequestQueue::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size();
}

bool RequestQueue::closed() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return closed_;
}

}  // namespace gpt2
