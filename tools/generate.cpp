// Phase 2 CLI: generate text greedily, with or without the KV cache.
// Usage: gpt2_generate --prompt "The quick brown fox" [--max-tokens 50] [--cache on|off]
//                      [--weights weights] [--device cpu] [--dtype fp32] [--quiet]
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "common/device.h"
#include "model/generate.h"
#include "model/gpt2.h"
#include "tokenizer/tokenizer.h"

namespace {

std::vector<int64_t> parse_ids(const std::string& csv) {
  std::vector<int64_t> ids;
  std::stringstream ss(csv);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) ids.push_back(std::stoll(item));
  }
  return ids;
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights = "weights", device_name = "cpu", dtype_name = "fp32", prompt_text, ids_csv, cache_flag = "on";
  int64_t max_tokens = 50;
  bool quiet = false;

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
      else if (arg == "--prompt") prompt_text = next();
      else if (arg == "--ids") ids_csv = next();
      else if (arg == "--max-tokens") max_tokens = std::stoll(next());
      else if (arg == "--cache") cache_flag = next();
      else if (arg == "--quiet") quiet = true;
      else throw std::invalid_argument("unknown argument " + arg);
    } catch (const std::exception& e) {
      std::cerr << "error: " << e.what() << "\n";
      return 2;
    }
  }

  try {
    if (prompt_text.empty() && ids_csv.empty()) throw std::invalid_argument("pass --prompt or --ids");
    if (cache_flag != "on" && cache_flag != "off") throw std::invalid_argument("--cache must be on or off");

    const auto model = gpt2::GPT2Model::load(weights, gpt2::parse_device(device_name), gpt2::parse_dtype(dtype_name));
    gpt2::Tokenizer tokenizer = gpt2::Tokenizer::from_file(weights + "/tokenizer.json");
    const auto prompt = ids_csv.empty() ? tokenizer.encode(prompt_text) : parse_ids(ids_csv);

    gpt2::GenerationOptions options;
    options.max_new_tokens = max_tokens;
    options.use_cache = (cache_flag == "on");

    gpt2::IncrementalDecoder stream(tokenizer);
    if (!quiet) std::cout << tokenizer.decode(prompt) << std::flush;
    const auto result = gpt2::generate_greedy(model, prompt, options, nullptr, [&](int64_t token) {
      if (!quiet) std::cout << stream.push(token) << std::flush;
    });
    if (!quiet) std::cout << stream.flush() << "\n";

    const auto& s = result.stats;
    std::cout << "\n[prompt " << prompt.size() << " tokens | cache " << cache_flag << " | prefill " << s.prefill_ms
              << " ms | " << s.steps << " tokens in " << s.decode_ms << " ms = " << s.ms_per_token()
              << " ms/token | " << (s.ms_per_token() > 0 ? 1000.0 / s.ms_per_token() : 0.0) << " tok/s"
              << (result.hit_eos ? " | stopped at EOS" : "") << "]\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
