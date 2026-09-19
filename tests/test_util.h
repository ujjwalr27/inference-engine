#pragma once

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace gpt2::test {

// Directory holding model.safetensors, config.json, reference.safetensors.
// Override with the GPT2_DATA_DIR environment variable.
inline std::string data_dir() {
  if (const char* env = std::getenv("GPT2_DATA_DIR")) return env;
  return GPT2_DEFAULT_DATA_DIR;
}

inline bool data_file_exists(const std::string& name) {
  return std::filesystem::exists(std::filesystem::path(data_dir()) / name);
}

// Like torch::allclose (|a - b| <= atol + rtol * |b|) but also reports the worst absolute error.
inline ::testing::AssertionResult AllClose(const torch::Tensor& actual, const torch::Tensor& expected, double rtol,
                                           double atol) {
  if (actual.sizes() != expected.sizes()) {
    return ::testing::AssertionFailure() << "shape mismatch: " << actual.sizes() << " vs " << expected.sizes();
  }
  const auto a = actual.detach().to(torch::kCPU, torch::kFloat64);
  const auto b = expected.detach().to(torch::kCPU, torch::kFloat64);
  const auto diff = (a - b).abs();
  const double max_abs = diff.max().item<double>();
  const bool ok = (diff <= atol + rtol * b.abs()).all().item<bool>();
  std::ostringstream msg;
  msg << "max |diff| = " << max_abs << " (rtol=" << rtol << ", atol=" << atol << ")";
  return ok ? ::testing::AssertionSuccess() << msg.str() : ::testing::AssertionFailure() << msg.str();
}

inline std::vector<int64_t> to_vector(const torch::Tensor& t) {
  const auto cpu = t.detach().to(torch::kCPU, torch::kInt64).contiguous();
  return std::vector<int64_t>(cpu.data_ptr<int64_t>(), cpu.data_ptr<int64_t>() + cpu.numel());
}

// Index of the first differing token, or -1 when the sequences are equal.
inline int64_t first_difference(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) {
    if (a[i] != b[i]) return static_cast<int64_t>(i);
  }
  return a.size() == b.size() ? -1 : static_cast<int64_t>(n);
}

// Gap between the two highest logits. A divergence in greedy tokens is only acceptable when
// this is tiny (a near-tie that different BLAS kernels can order differently).
inline double top2_gap(const torch::Tensor& logits_row) {
  const auto top2 = std::get<0>(logits_row.detach().to(torch::kCPU, torch::kFloat64).topk(2));
  return (top2[0] - top2[1]).item<double>();
}

}  // namespace gpt2::test
