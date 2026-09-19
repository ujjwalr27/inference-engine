#include "common/device.h"

#include <sstream>
#include <stdexcept>

#include <torch/version.h>

namespace gpt2 {

torch::Device parse_device(const std::string& name) {
  if (name == "cpu") {
    return torch::Device(torch::kCPU);
  }
  if (name == "cuda" || name.rfind("cuda:", 0) == 0) {
    if (!torch::cuda::is_available()) {
      throw std::runtime_error("CUDA requested but not available");
    }
    int index = 0;
    if (name.size() > 5) {
      try {
        size_t consumed = 0;
        index = std::stoi(name.substr(5), &consumed);
        if (consumed != name.size() - 5) {
          throw std::invalid_argument("trailing characters");
        }
      } catch (const std::exception&) {
        throw std::invalid_argument("invalid device: " + name);
      }
    }
    if (index < 0 || index >= static_cast<int>(torch::cuda::device_count())) {
      throw std::invalid_argument("CUDA device index out of range: " + name);
    }
    return torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(index));
  }
  throw std::invalid_argument("invalid device: " + name);
}

torch::Dtype parse_dtype(const std::string& name) {
  if (name == "fp32") return torch::kFloat32;
  if (name == "fp16") return torch::kFloat16;
  throw std::invalid_argument("invalid dtype: " + name);
}

std::string runtime_summary() {
  std::ostringstream out;
  out << "LibTorch " << TORCH_VERSION << ", threads=" << torch::get_num_threads()
      << ", cuda=" << (torch::cuda::is_available() ? "yes" : "no");
  if (torch::cuda::is_available()) {
    out << " (devices=" << torch::cuda::device_count() << ")";
  }
  return out.str();
}

}  // namespace gpt2
