#include "scheduler/request.h"

namespace gpt2 {

void TokenChannel::push(int64_t token) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return;
    tokens_.push_back(token);
  }
  cv_.notify_all();
}

void TokenChannel::finish(std::string error) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return;
    closed_ = true;
    error_ = std::move(error);
  }
  cv_.notify_all();
}

bool TokenChannel::next(int64_t& token) {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return !tokens_.empty() || closed_; });
  if (tokens_.empty()) return false;
  token = tokens_.front();
  tokens_.pop_front();
  return true;
}

std::vector<int64_t> TokenChannel::collect() {
  std::vector<int64_t> all;
  int64_t token = 0;
  while (next(token)) all.push_back(token);
  return all;
}

bool TokenChannel::closed() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return closed_;
}

std::string TokenChannel::error() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return error_;
}

const char* to_string(FinishReason reason) {
  switch (reason) {
    case FinishReason::None: return "none";
    case FinishReason::EosToken: return "eos";
    case FinishReason::MaxTokens: return "max_tokens";
    case FinishReason::ContextFull: return "context_full";
    case FinishReason::Cancelled: return "cancelled";
    case FinishReason::Error: return "error";
  }
  return "unknown";
}

}  // namespace gpt2
