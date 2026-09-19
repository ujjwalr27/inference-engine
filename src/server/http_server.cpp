#include "server/http_server.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "tokenizer/tokenizer.h"

namespace gpt2 {
namespace {

using json = nlohmann::json;

double ms_between(SchedClock::time_point a, SchedClock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

json error_body(const std::string& message) { return json{{"error", message}}; }

struct ParsedRequest {
  std::vector<int64_t> prompt;
  GenerationOptions options;
  bool stream = false;
};

}  // namespace

struct InferenceServer::Impl {
  Impl(const GPT2Model& model, Scheduler& scheduler, std::string tokenizer_path, const ServerOptions& options)
      : model(model), scheduler(scheduler), tokenizer_path(std::move(tokenizer_path)), options(options) {}

  const GPT2Model& model;
  Scheduler& scheduler;
  std::string tokenizer_path;
  ServerOptions options;
  httplib::Server server;
  std::thread thread;
  std::atomic<bool> running{false};

  // Parsing tokenizer.json takes a moment, so each worker thread keeps its own instance
  // instead of sharing one (the wrapper makes no thread-safety promise).
  const Tokenizer& tokenizer() {
    static thread_local std::unique_ptr<Tokenizer> local;
    if (!local) local = std::make_unique<Tokenizer>(Tokenizer::from_file(tokenizer_path));
    return *local;
  }

  // Throws std::invalid_argument with a client-facing message.
  ParsedRequest parse(const std::string& body) {
    json j;
    try {
      j = json::parse(body);
    } catch (const json::exception&) {
      throw std::invalid_argument("body must be JSON");
    }
    ParsedRequest parsed;
    if (j.contains("token_ids")) {
      parsed.prompt = j.at("token_ids").get<std::vector<int64_t>>();
    } else if (j.contains("prompt")) {
      parsed.prompt = tokenizer().encode(j.at("prompt").get<std::string>());
    } else {
      throw std::invalid_argument("provide 'prompt' or 'token_ids'");
    }
    if (parsed.prompt.empty()) throw std::invalid_argument("prompt is empty");
    for (int64_t id : parsed.prompt) {
      if (id < 0 || id >= model.config().vocab_size) throw std::invalid_argument("token id out of range");
    }

    parsed.options.max_new_tokens = j.value("max_tokens", options.default_max_tokens);
    parsed.options.stop_on_eos = j.value("stop_on_eos", true);
    parsed.stream = j.value("stream", false);
    return parsed;
  }

  json timings_of(const Request& request) const {
    return json{
        {"queue_ms", ms_between(request.t_enqueue, request.t_prefill_start)},
        {"ttft_ms", ms_between(request.t_enqueue, request.t_first_token)},
        {"total_ms", ms_between(request.t_enqueue, request.t_done)},
        {"output_tokens", request.generated.size()},
        {"prompt_tokens", request.prompt.size()},
    };
  }

  void handle_generate(const httplib::Request& req, httplib::Response& res);
  void install_routes();
};

void InferenceServer::Impl::handle_generate(const httplib::Request& req, httplib::Response& res) {
  ParsedRequest parsed;
  try {
    parsed = parse(req.body);
  } catch (const std::exception& e) {
    res.status = 400;
    res.set_content(error_body(e.what()).dump(), "application/json");
    return;
  }

  std::shared_ptr<Request> request;
  try {
    request = scheduler.submit(parsed.prompt, parsed.options);
  } catch (const std::invalid_argument& e) {  // can never run: too long, empty, bad max_tokens
    res.status = 400;
    res.set_content(error_body(e.what()).dump(), "application/json");
    return;
  }
  if (!request) {
    res.status = 429;
    res.set_header("Retry-After", "1");
    res.set_content(error_body("server busy: request queue is full").dump(), "application/json");
    return;
  }

  if (!parsed.stream) {
    const auto tokens = request->out->collect();
    if (!request->out->error().empty()) {
      res.status = 503;
      res.set_content(error_body(request->out->error()).dump(), "application/json");
      return;
    }
    // Safe to read scheduler-written fields now: the channel's mutex published them.
    res.set_content(json{{"text", tokenizer().decode(tokens)},
                         {"token_ids", tokens},
                         {"finish_reason", to_string(request->finish_reason)},
                         {"timings", timings_of(*request)}}
                        .dump(),
                    "application/json");
    return;
  }

  // Server-sent events. The provider runs on this worker thread until generation ends,
  // which is why the pool has to be bigger than the number of in-flight requests.
  auto decoder = std::make_shared<IncrementalDecoder>(tokenizer());
  res.set_chunked_content_provider("text/event-stream", [this, request, decoder](size_t, httplib::DataSink& sink) {
    // Notice a hung-up client even between writes (a held-back partial character writes nothing).
    if (!sink.is_writable()) {
      request->cancel();
      return false;
    }
    int64_t token = 0;
    if (request->out->next(token)) {
      const std::string piece = decoder->push(token);
      if (!piece.empty()) {
        const std::string event = "data: " + json{{"text", piece}, {"token_id", token}}.dump() + "\n\n";
        if (!sink.write(event.data(), event.size())) {
          request->cancel();  // client hung up: stop generating and free the slot
          return false;
        }
      }
      return true;
    }

    const std::string tail = decoder->flush();
    std::string event;
    if (!tail.empty()) event += "data: " + json{{"text", tail}}.dump() + "\n\n";
    event += "data: " + json{{"finish_reason", to_string(request->finish_reason)},
                             {"timings", timings_of(*request)}}
                            .dump() +
             "\n\n";
    event += "data: [DONE]\n\n";
    sink.write(event.data(), event.size());
    sink.done();
    return false;
  });
}

void InferenceServer::Impl::install_routes() {
  server.Post("/v1/generate", [this](const httplib::Request& req, httplib::Response& res) {
    handle_generate(req, res);
  });

  server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(json{{"status", "ok"}}.dump(), "application/json");
  });

  server.Get("/stats", [this](const httplib::Request&, httplib::Response& res) {
    const auto s = scheduler.stats();
    res.set_content(json{{"submitted", s.submitted},
                         {"rejected", s.rejected},
                         {"admitted", s.admitted},
                         {"finished", s.finished},
                         {"cancelled", s.cancelled},
                         {"decode_steps", s.decode_steps},
                         {"generated_tokens", s.generated_tokens},
                         {"max_batch", s.max_batch},
                         {"active", s.active},
                         {"queued", s.queued},
                         {"slots", scheduler.options().n_slots},
                         {"max_queue", scheduler.options().max_queue}}
                        .dump(),
                    "application/json");
  });

  server.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
    std::string message = "internal error";
    try {
      std::rethrow_exception(ep);
    } catch (const std::exception& e) {
      message = e.what();
    } catch (...) {
    }
    res.status = 500;
    res.set_content(error_body(message).dump(), "application/json");
  });
}

InferenceServer::InferenceServer(const GPT2Model& model, Scheduler& scheduler, const std::string& tokenizer_path,
                                 const ServerOptions& options)
    : impl_(std::make_unique<Impl>(model, scheduler, tokenizer_path, options)) {
  size_t threads = options.threads;
  if (threads == 0) {
    threads = static_cast<size_t>(scheduler.options().n_slots) + scheduler.options().max_queue + 8;
  }
  impl_->server.new_task_queue = [threads] { return new httplib::ThreadPool(threads); };
  impl_->install_routes();
}

InferenceServer::~InferenceServer() { stop(); }

bool InferenceServer::listen() {
  impl_->running = true;
  const bool ok = impl_->server.listen(impl_->options.host.c_str(), impl_->options.port);
  impl_->running = false;
  return ok;
}

int InferenceServer::listen_on_any_port() {
  const int port = impl_->server.bind_to_any_port(impl_->options.host.c_str());
  if (port <= 0) return port;
  impl_->options.port = port;
  impl_->running = true;
  impl_->thread = std::thread([this] {
    impl_->server.listen_after_bind();
    impl_->running = false;
  });
  while (!impl_->server.is_running()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  return port;
}

void InferenceServer::stop() {
  impl_->server.stop();
  if (impl_->thread.joinable()) impl_->thread.join();
  impl_->running = false;
}

bool InferenceServer::is_running() const { return impl_->running.load(); }

}  // namespace gpt2
