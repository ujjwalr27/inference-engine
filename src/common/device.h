#pragma once

#include <string>

#include <torch/torch.h>

namespace gpt2 {

// Parses "cpu", "cuda", or "cuda:N". Throws std::invalid_argument on bad input
// and std::runtime_error if CUDA is requested but unavailable.
torch::Device parse_device(const std::string& name);

// Parses "fp32" or "fp16". Throws std::invalid_argument on bad input.
torch::Dtype parse_dtype(const std::string& name);

// One-line summary of the LibTorch build and visible accelerators.
std::string runtime_summary();

}  // namespace gpt2
