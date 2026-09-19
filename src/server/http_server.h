#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "model/gpt2.h"
#include "scheduler/scheduler.h"

namespace gpt2 {

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8080;
  // Each streaming response holds a worker thread for its whole lifetime, so the pool must be
  // larger than slots + queue or concurrency is silently capped (cpp-httplib defaults to ~8).
  size_t threads = 0;  // 0 = n_slots + max_queue + 8
  int64_t default_max_tokens = 64;
};

// HTTP front end: tokenize, hand the token IDs to the scheduler, stream tokens back.
// Tokenization and detokenization happen on the HTTP worker threads; the scheduler thread
// only ever sees token IDs.
class InferenceServer {
 public:
  InferenceServer(const GPT2Model& model, Scheduler& scheduler, const std::string& tokenizer_path,
                  const ServerOptions& options);
  ~InferenceServer();

  // Blocks until stop(). Returns false if the port could not be bound.
  bool listen();

  // Binds an ephemeral port and starts serving on a background thread; returns the port.
  // Used by tests and by `--port 0`.
  int listen_on_any_port();

  void stop();
  bool is_running() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpt2
