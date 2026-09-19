// Micro-benchmarks for the two things the engine actually does: prefill a prompt and run one
// decode step for a batch. Everything is timed with the device synchronised, so GPU numbers are
// real rather than the cost of queueing kernels.
//
// Usage: gpt2_bench [--weights weights] [--device cpu|cuda] [--dtype fp32|fp16]
//                   [--slots 32] [--iters 20] [--warmup 5] [--out results/bench.csv]
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "cache/kv_cache.h"
#include "common/device.h"
#include "model/gpt2.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Row {
  std::string phase;
  int64_t batch;
  int64_t length;
  double ms;
  double tokens_per_s;
};

void sync(const torch::Device& device) {
  if (device.is_cuda()) torch::cuda::synchronize();
}

template <typename Fn>
double timed_ms(const torch::Device& device, int iters, int warmup, Fn&& fn) {
  for (int i = 0; i < warmup; ++i) fn();
  sync(device);
  const auto start = Clock::now();
  for (int i = 0; i < iters; ++i) fn();
  sync(device);
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count() / iters;
}

void print_row(const Row& row) {
  std::cout << std::left << std::setw(8) << row.phase << " batch " << std::setw(4) << row.batch << " len "
            << std::setw(6) << row.length << std::right << std::fixed << std::setprecision(3) << std::setw(10)
            << row.ms << " ms " << std::setprecision(1) << std::setw(9) << row.tokens_per_s << " tok/s\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights = "weights", device_name = "cpu", dtype_name = "fp32", out_path;
  int64_t slots = 32;
  int iters = 20, warmup = 5;

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
      else if (arg == "--slots") slots = std::stoll(next());
      else if (arg == "--iters") iters = std::stoi(next());
      else if (arg == "--warmup") warmup = std::stoi(next());
      else if (arg == "--out") out_path = next();
      else throw std::invalid_argument("unknown argument " + arg);
    } catch (const std::exception& e) {
      std::cerr << "error: " << e.what() << "\n";
      return 2;
    }
  }

  try {
    const auto device = gpt2::parse_device(device_name);
    const auto dtype = gpt2::parse_dtype(dtype_name);
    std::cout << gpt2::runtime_summary() << "\ndevice " << device << " dtype " << dtype_name << " slots " << slots
              << " iters " << iters << "\n\n";

    const auto model = gpt2::GPT2Model::load(weights, device, dtype);
    torch::InferenceMode guard;
    gpt2::KVCache cache(model.config(), slots, device, dtype);
    std::cout << "KV cache: " << cache.bytes() / (1024.0 * 1024.0) << " MiB for " << slots << " slots\n\n";

    std::vector<Row> rows;

    // Prefill: one sequence of a given length.
    for (int64_t length : {16, 64, 256, 512, 1024}) {
      const auto ids = torch::randint(0, model.config().vocab_size, {1, length},
                                      torch::TensorOptions().dtype(torch::kInt64));
      const double ms = timed_ms(device, iters, warmup, [&] { model.prefill(ids, 0, cache); });
      rows.push_back({"prefill", 1, length, ms, static_cast<double>(length) / ms * 1000.0});
      print_row(rows.back());
    }
    std::cout << "\n";

    // Decode: one step for `batch` rows whose caches already hold `length` tokens.
    for (int64_t length : {128, 512, 1023}) {
      for (int64_t batch : {1, 2, 4, 8, 16, 32}) {
        if (batch > slots) continue;
        const auto ids = torch::randint(0, model.config().vocab_size, {batch},
                                        torch::TensorOptions().dtype(torch::kInt64));
        const auto positions = torch::full({batch}, length, torch::TensorOptions().dtype(torch::kInt64));
        const double ms =
            timed_ms(device, iters, warmup, [&] { model.decode(ids, positions, length + 1, cache); });
        rows.push_back({"decode", batch, length, ms, static_cast<double>(batch) / ms * 1000.0});
        print_row(rows.back());
      }
      std::cout << "\n";
    }

    if (!out_path.empty()) {
      std::ofstream out(out_path);
      if (!out) {
        std::cerr << "warning: cannot write " << out_path << "\n";
      } else {
        out << "phase,device,dtype,batch,length,ms,tokens_per_s\n";
        for (const auto& row : rows) {
          out << row.phase << ',' << device_name << ',' << dtype_name << ',' << row.batch << ',' << row.length << ','
              << row.ms << ',' << row.tokens_per_s << '\n';
        }
        std::cout << "wrote " << out_path << "\n";
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
