#include "scheduler/scheduler.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <thread>

namespace gpt2 {

const char* to_string(BatchingPolicy policy) {
  return policy == BatchingPolicy::Static ? "static" : "continuous";
}

BatchingPolicy parse_policy(const std::string& name) {
  if (name == "continuous") return BatchingPolicy::Continuous;
  if (name == "static") return BatchingPolicy::Static;
  throw std::invalid_argument("policy must be 'continuous' or 'static', got '" + name + "'");
}

Scheduler::Scheduler(const GPT2Model& model, const SchedulerOptions& options)
    : model_(model),
      options_(options),
      cache_(model.config(), options.n_slots, model.device(), model.dtype()),
      queue_(options.max_queue) {
  if (options_.n_slots < 1) throw std::invalid_argument("n_slots must be at least 1");
  if (options_.prefill_budget_tokens < 1) throw std::invalid_argument("prefill budget must be at least 1");
  active_.reserve(static_cast<size_t>(options_.n_slots));

  auto host_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  if (model.device().is_cuda()) host_options = host_options.pinned_memory(true);
  ids_host_ = torch::empty({options_.n_slots}, host_options);
  positions_host_ = torch::empty({options_.n_slots}, host_options);
}

Scheduler::~Scheduler() { stop(); }

void Scheduler::start() {
  if (running_.exchange(true)) return;
  thread_ = std::thread([this] { run(); });
}

void Scheduler::stop() {
  const bool was_running = running_.exchange(false);
  queue_.close();
  if (was_running && thread_.joinable()) {
    thread_.join();  // run() drains the queue and the active slots on its way out
  } else {
    while (auto pending = queue_.try_pop()) pending->out->finish("scheduler stopped");
  }
}

std::shared_ptr<Request> Scheduler::submit(std::vector<int64_t> prompt, const GenerationOptions& options) {
  if (prompt.empty()) throw std::invalid_argument("prompt must not be empty");
  if (options.max_new_tokens < 1) throw std::invalid_argument("max_new_tokens must be at least 1");
  const int64_t total = static_cast<int64_t>(prompt.size()) + options.max_new_tokens;
  if (total > model_.config().n_ctx) {
    throw std::invalid_argument("prompt (" + std::to_string(prompt.size()) + ") + max_new_tokens (" +
                                std::to_string(options.max_new_tokens) + ") exceeds context " +
                                std::to_string(model_.config().n_ctx));
  }

  auto request = std::make_shared<Request>();
  request->id = next_id_.fetch_add(1);
  request->prompt = std::move(prompt);
  request->options = options;
  request->t_enqueue = SchedClock::now();

  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    ++stats_.submitted;
  }
  if (!queue_.try_push(request)) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    ++stats_.rejected;
    return nullptr;
  }
  return request;
}

void Scheduler::run() {
  while (running_.load()) {
    admit();
    if (active_.empty()) {
      if (queue_.size() == 0) {
        queue_.wait_for_work(options_.idle_wait);
      } else {
        // Static policy gathering a batch: requests are waiting but the policy says not yet.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      continue;
    }
    step();
  }

  // Shutting down: tell everyone still in flight, in the queue and in a slot.
  for (size_t i = active_.size(); i-- > 0;) finish(i, FinishReason::Error);
  while (auto pending = queue_.try_pop()) pending->out->finish("server shutting down");
}

size_t Scheduler::admission_limit(bool& apply_budget) {
  apply_budget = true;
  const size_t free_slots = static_cast<size_t>(options_.n_slots) - active_.size();

  if (options_.policy == BatchingPolicy::Continuous) {
    return free_slots;
  }

  // Static: nothing new joins until the current batch has finished completely.
  if (!active_.empty()) return 0;
  const size_t queued = queue_.size();
  if (queued == 0) {
    batch_wait_start_.reset();
    return 0;
  }

  const size_t target = options_.static_batch_size > 0
                            ? std::min<size_t>(static_cast<size_t>(options_.static_batch_size), free_slots)
                            : free_slots;
  if (queued < target) {
    // Wait for a full batch, but not forever, or a trickle of requests would never run.
    if (!batch_wait_start_) {
      batch_wait_start_ = SchedClock::now();
      return 0;
    }
    if (SchedClock::now() - *batch_wait_start_ < options_.static_max_wait) return 0;
  }
  batch_wait_start_.reset();
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    ++stats_.batches;
  }
  // The whole batch is prefilled now: applying the per-step budget here would split it in two.
  apply_budget = false;
  return std::min(target, queued);
}

void Scheduler::admit() {
  bool apply_budget = true;
  size_t remaining = admission_limit(apply_budget);
  int64_t prefill_tokens = 0;
  while (remaining > 0 && (!apply_budget || prefill_tokens < options_.prefill_budget_tokens)) {
    auto request = queue_.try_pop();
    if (!request) break;
    --remaining;

    if (request->is_cancelled()) {
      request->finish_reason = FinishReason::Cancelled;
      request->out->finish();
      std::lock_guard<std::mutex> lock(stats_mutex_);
      ++stats_.cancelled;
      continue;
    }

    const int64_t slot = static_cast<int64_t>(active_.size());
    request->slot = slot;
    request->t_prefill_start = SchedClock::now();

    const auto ids = torch::tensor(request->prompt, torch::kInt64).unsqueeze(0);
    const auto logits = model_.prefill(ids, slot, cache_);
    const int64_t first = logits[0].argmax(-1).item<int64_t>();

    request->position = static_cast<int64_t>(request->prompt.size());
    request->next_token = first;
    request->generated.push_back(first);
    request->t_first_token = SchedClock::now();
    request->out->push(first);
    prefill_tokens += static_cast<int64_t>(request->prompt.size());

    active_.push_back(request);
    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      ++stats_.admitted;
      ++stats_.generated_tokens;
      stats_.max_batch = std::max<int64_t>(stats_.max_batch, static_cast<int64_t>(active_.size()));
      stats_.active = static_cast<int64_t>(active_.size());
      stats_.queued = queue_.size();
    }

    // A one-token request, or one cancelled during prefill, is already done.
    const auto eos = request->options.eos_token_id >= 0 ? request->options.eos_token_id
                                                        : model_.config().eos_token_id;
    if (request->is_cancelled()) {
      finish(active_.size() - 1, FinishReason::Cancelled);
    } else if (request->options.stop_on_eos && first == eos) {
      finish(active_.size() - 1, FinishReason::EosToken);
    } else if (static_cast<int64_t>(request->generated.size()) >= request->options.max_new_tokens) {
      finish(active_.size() - 1, FinishReason::MaxTokens);
    }
  }
}

void Scheduler::step() {
  const size_t batch = active_.size();
  auto ids_acc = ids_host_.accessor<int64_t, 1>();
  auto positions_acc = positions_host_.accessor<int64_t, 1>();
  int64_t length = 0;
  for (size_t i = 0; i < batch; ++i) {
    const auto row = static_cast<int64_t>(i);
    ids_acc[row] = active_[i]->next_token;
    positions_acc[row] = active_[i]->position;
    length = std::max(length, active_[i]->position + 1);
  }

  const auto rows = static_cast<int64_t>(batch);
  const auto device = model_.device();
  const auto ids = ids_host_.narrow(0, 0, rows).to(device, /*non_blocking=*/true);
  const auto positions = positions_host_.narrow(0, 0, rows).to(device, /*non_blocking=*/true);
  const auto logits = model_.decode(ids, positions, length, cache_);
  // One device -> host transfer per step, never one per request.
  const auto next = logits.argmax(-1).to(torch::kCPU, torch::kInt64).contiguous();
  const auto acc = next.accessor<int64_t, 1>();

  for (size_t i = 0; i < batch; ++i) {
    auto& request = active_[i];
    ++request->position;
    const int64_t token = acc[static_cast<int64_t>(i)];
    request->next_token = token;
    request->generated.push_back(token);
    request->out->push(token);
  }
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    ++stats_.decode_steps;
    stats_.generated_tokens += batch;
  }

  // Walk backwards so compaction never skips a row.
  for (size_t i = batch; i-- > 0;) {
    const auto& request = active_[i];
    const auto eos =
        request->options.eos_token_id >= 0 ? request->options.eos_token_id : model_.config().eos_token_id;
    if (request->is_cancelled()) {
      finish(i, FinishReason::Cancelled);
    } else if (request->options.stop_on_eos && request->next_token == eos) {
      finish(i, FinishReason::EosToken);
    } else if (static_cast<int64_t>(request->generated.size()) >= request->options.max_new_tokens) {
      finish(i, FinishReason::MaxTokens);
    } else if (request->position + 1 > model_.config().n_ctx) {
      finish(i, FinishReason::ContextFull);
    }
  }
}

void Scheduler::finish(size_t index, FinishReason reason) {
  auto request = active_[index];
  request->finish_reason = reason;
  request->t_done = SchedClock::now();
  request->out->finish(reason == FinishReason::Error ? "server shutting down" : std::string{});
  release_slot(index);

  std::lock_guard<std::mutex> lock(stats_mutex_);
  if (reason == FinishReason::Cancelled) {
    ++stats_.cancelled;
  } else {
    ++stats_.finished;
  }
  stats_.active = static_cast<int64_t>(active_.size());
  stats_.queued = queue_.size();
}

void Scheduler::release_slot(size_t index) {
  const size_t last = active_.size() - 1;
  if (index != last) {
    // Move the last active request into the freed slot so slots stay packed at the front and
    // decode can keep reading cache.keys(layer, batch, length) as a view instead of a copy.
    auto moved = active_[last];
    cache_.copy_slot(moved->slot, static_cast<int64_t>(index), moved->position);
    moved->slot = static_cast<int64_t>(index);
    active_[index] = moved;
  }
  active_.pop_back();
}

SchedulerStats Scheduler::stats() const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  SchedulerStats copy = stats_;
  copy.queued = queue_.size();
  return copy;
}

}  // namespace gpt2
