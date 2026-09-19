// Runs one uncached forward pass and prints the top next-token candidates.
// Usage: gpt2_forward --ids 464,2068,7586 [--weights weights] [--device cpu] [--dtype fp32] [--top 5]
// (Token IDs come from Python for now; the C++ tokenizer arrives in Phase 2.)
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "common/device.h"
#include "model/gpt2.h"

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

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights = "weights", device_name = "cpu", dtype_name = "fp32", ids_csv;
  int64_t top = 5;
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
      else if (arg == "--ids") ids_csv = next();
      else if (arg == "--top") top = std::stoll(next());
      else throw std::invalid_argument("unknown argument " + arg);
    } catch (const std::exception& e) {
      std::cerr << "error: " << e.what() << "\n";
      return 2;
    }
  }

  try {
    const auto ids = parse_ids(ids_csv);
    if (ids.empty()) throw std::invalid_argument("--ids is required, e.g. --ids 464,2068,7586");
    const auto device = gpt2::parse_device(device_name);
    const auto dtype = gpt2::parse_dtype(dtype_name);
    std::cout << gpt2::runtime_summary() << "\n";

    auto t0 = std::chrono::steady_clock::now();
    const auto model = gpt2::GPT2Model::load(weights, device, dtype);
    std::cout << "loaded " << weights << " in " << ms_since(t0) << " ms\n";

    torch::InferenceMode guard;
    const auto input = torch::tensor(ids, torch::kInt64).unsqueeze(0);
    t0 = std::chrono::steady_clock::now();
    auto logits = model.forward(input);
    if (device.is_cuda()) torch::cuda::synchronize();
    std::cout << "forward T=" << ids.size() << " in " << ms_since(t0) << " ms\n";

    const auto last = logits[0][-1].to(torch::kCPU, torch::kFloat32);
    const auto best = last.topk(top);
    const auto probs = torch::softmax(last, -1);
    for (int64_t k = 0; k < top; ++k) {
      const auto id = std::get<1>(best)[k].item<int64_t>();
      std::cout << "  #" << k + 1 << " id=" << id << " logit=" << std::get<0>(best)[k].item<float>()
                << " p=" << probs[id].item<float>() << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
