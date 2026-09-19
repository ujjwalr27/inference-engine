// Phase 0 smoke program: prints the runtime and does a matmul on the chosen device.
// Usage: hello_tensor [cpu|cuda|cuda:N]
#include <chrono>
#include <iostream>

#include <torch/torch.h>

#include "common/device.h"

int main(int argc, char** argv) {
  try {
    const std::string device_name = argc > 1 ? argv[1] : "cpu";
    const torch::Device device = gpt2::parse_device(device_name);
    torch::InferenceMode guard;

    std::cout << gpt2::runtime_summary() << "\n";

    torch::manual_seed(0);
    auto a = torch::rand({2, 3}, torch::TensorOptions().device(device));
    std::cout << "tensor on " << a.device() << ":\n" << a << "\n";

    auto x = torch::randn({1024, 1024}, torch::TensorOptions().device(device));
    auto start = std::chrono::steady_clock::now();
    auto y = torch::mm(x, x);
    if (device.is_cuda()) torch::cuda::synchronize();
    auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cout << "1024x1024 matmul: " << ms << " ms, checksum " << y.sum().item<double>() << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
