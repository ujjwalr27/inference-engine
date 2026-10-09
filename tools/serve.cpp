// Phase 5: HTTP server with token streaming.
// Usage: gpt2_serve [--weights weights] [--device cpu] [--dtype fp32]
//                   [--slots 8] [--max-queue 64] [--prefill-budget 512]
//                   [--policy continuous|static] [--static-batch N] [--static-wait MS]
//                   [--host 127.0.0.1] [--port 8080] [--threads 0]
//                   [--cuda-graphs | --no-cuda-graphs] [--graph-length-step 64] [--solo-prefill]
// CUDA graphs are on by default when the model runs on CUDA.
#include <chrono>
#include <csignal>
#include <iostream>
#include <optional>
#include <string>
#include <thread>

#include "common/device.h"
#include "model/decode_graphs.h"
#include "model/gpt2.h"
#include "scheduler/scheduler.h"
#include "server/http_server.h"

namespace {
gpt2::InferenceServer* g_server = nullptr;

void handle_signal(int) {
  if (g_server) g_server->stop();
}
}  // namespace

int main(int argc, char** argv) {
  std::string weights = "weights", device_name = "cpu", dtype_name = "fp32";
  gpt2::SchedulerOptions scheduler_options;
  gpt2::ServerOptions server_options;
  std::optional<bool> cuda_graphs;  // unset: on for CUDA, off otherwise

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
      return argv[++i];
    };
    try {
      if (arg == "--weights") weights = next();
      else if (arg == "--device") device_name = next();
      else if (arg == "--dtype") dtype_name = next();
      else if (arg == "--slots") scheduler_options.n_slots = std::stoll(next());
      else if (arg == "--max-queue") scheduler_options.max_queue = std::stoul(next());
      else if (arg == "--prefill-budget") scheduler_options.prefill_budget_tokens = std::stoll(next());
      else if (arg == "--policy") scheduler_options.policy = gpt2::parse_policy(next());
      else if (arg == "--static-batch") scheduler_options.static_batch_size = std::stoll(next());
      else if (arg == "--static-wait") scheduler_options.static_max_wait = std::chrono::milliseconds(std::stoll(next()));
      else if (arg == "--cuda-graphs") cuda_graphs = true;
      else if (arg == "--no-cuda-graphs") cuda_graphs = false;
      else if (arg == "--graph-length-step") scheduler_options.graph_length_step = std::stoll(next());
      else if (arg == "--solo-prefill") scheduler_options.batch_prefill = false;
      else if (arg == "--host") server_options.host = next();
      else if (arg == "--port") server_options.port = std::stoi(next());
      else if (arg == "--threads") server_options.threads = std::stoul(next());
      else if (arg == "--max-tokens") server_options.default_max_tokens = std::stoll(next());
      else throw std::invalid_argument("unknown argument " + arg);
    } catch (const std::exception& e) {
      std::cerr << "error: " << e.what() << "\n";
      return 2;
    }
  }

  try {
    const auto device = gpt2::parse_device(device_name);
    const auto dtype = gpt2::parse_dtype(dtype_name);
    std::cout << gpt2::runtime_summary() << "\n";

    scheduler_options.cuda_graphs = cuda_graphs.value_or(device.is_cuda() && gpt2::DecodeGraphs::available());
    const auto model = gpt2::GPT2Model::load(weights, device, dtype);
    gpt2::Scheduler scheduler(model, scheduler_options);
    scheduler.start();

    gpt2::InferenceServer server(model, scheduler, weights + "/tokenizer.json", server_options);
    g_server = &server;
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    const int port = server_options.port == 0 ? server.listen_on_any_port() : server_options.port;
    std::cout << "serving on http://" << server_options.host << ":" << port << " | policy "
              << gpt2::to_string(scheduler_options.policy) << " | slots " << scheduler_options.n_slots << " | queue "
              << scheduler_options.max_queue << " | prefill budget " << scheduler_options.prefill_budget_tokens
              << " tokens/step (" << (scheduler_options.batch_prefill ? "batched" : "one pass per request")
              << ") | cuda graphs "
              << (scheduler_options.cuda_graphs
                      ? "on, length step " + std::to_string(scheduler_options.graph_length_step)
                      : std::string("off"))
              << "\n"
              << "POST /v1/generate  {\"prompt\": \"...\", \"max_tokens\": 64, \"stream\": true}\n"
              << std::flush;  // stdout is block-buffered under docker or a log file, not line-buffered

    if (server_options.port != 0 && !server.listen()) {
      std::cerr << "error: could not bind " << server_options.host << ":" << port << "\n";
      return 1;
    }
    if (server_options.port == 0) {
      while (server.is_running()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    scheduler.stop();
    std::cout << "stopped\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
