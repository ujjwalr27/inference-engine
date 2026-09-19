#include <stdexcept>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "common/device.h"

TEST(Smoke, TensorMath) {
  torch::InferenceMode guard;
  auto t = torch::ones({3});
  EXPECT_EQ(t.sum().item<float>(), 3.0f);
}

TEST(Device, ParsesCpu) {
  EXPECT_TRUE(gpt2::parse_device("cpu").is_cpu());
}

TEST(Device, RejectsGarbage) {
  EXPECT_THROW(gpt2::parse_device("gpu"), std::invalid_argument);
  EXPECT_THROW(gpt2::parse_device(""), std::invalid_argument);
}

TEST(Device, CudaWhenAvailable) {
  if (!torch::cuda::is_available()) {
    EXPECT_THROW(gpt2::parse_device("cuda"), std::runtime_error);
    GTEST_SKIP() << "CUDA not available";
  }
  EXPECT_TRUE(gpt2::parse_device("cuda").is_cuda());
  EXPECT_THROW(gpt2::parse_device("cuda:abc"), std::invalid_argument);
  EXPECT_THROW(gpt2::parse_device("cuda:99"), std::invalid_argument);
}

TEST(Dtype, Parses) {
  EXPECT_EQ(gpt2::parse_dtype("fp32"), torch::kFloat32);
  EXPECT_EQ(gpt2::parse_dtype("fp16"), torch::kFloat16);
  EXPECT_THROW(gpt2::parse_dtype("bf16"), std::invalid_argument);
}
