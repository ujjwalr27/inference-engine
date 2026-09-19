// Layer-level tests with small random weights (no model files needed).
#include <stdexcept>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "model/config.h"
#include "model/layers.h"
#include "test_util.h"

namespace {

using gpt2::test::AllClose;

gpt2::Linear random_linear(int64_t in, int64_t out) {
  return {torch::randn({out, in}) * 0.2, torch::randn({out}) * 0.1};
}

gpt2::Attention random_attention(int64_t dim, int64_t heads) {
  gpt2::Attention a;
  a.q_proj = random_linear(dim, dim);
  a.k_proj = random_linear(dim, dim);
  a.v_proj = random_linear(dim, dim);
  a.c_proj = random_linear(dim, dim);
  a.n_head = heads;
  return a;
}

class Layers : public ::testing::Test {
 protected:
  void SetUp() override { torch::manual_seed(1234); }
  torch::InferenceMode guard_;
};

TEST_F(Layers, GeluNewIsTheTanhApproximation) {
  const auto x = torch::linspace(-6.0, 6.0, 10001, torch::kFloat64);
  const auto approx = gpt2::gelu_new(x);
  const auto exact = torch::gelu(x);
  const double max_diff = (approx - exact).abs().max().item<double>();
  EXPECT_LT(max_diff, 1e-3) << "gelu_new should be close to exact GELU";
  EXPECT_GT(max_diff, 1e-6) << "gelu_new must not be the exact erf GELU";
  // Spot value computed independently: 0.5 * (1 + tanh(sqrt(2/pi) * (1 + 0.044715)))
  EXPECT_NEAR(gpt2::gelu_new(torch::tensor({1.0}, torch::kFloat64)).item<double>(), 0.8411920, 1e-6);
}

TEST_F(Layers, MaskFillValueIsFiniteAndDtypeSafe) {
  EXPECT_EQ(gpt2::mask_fill_value(torch::kFloat16), -65504.0);
  EXPECT_LT(gpt2::mask_fill_value(torch::kFloat32), -3e38);
  EXPECT_THROW(gpt2::mask_fill_value(torch::kInt64), std::invalid_argument);

  // A fully masked fp16 row must not produce NaN.
  auto scores = torch::zeros({1, 4}, torch::kFloat16);
  scores.fill_(gpt2::mask_fill_value(torch::kFloat16));
  EXPECT_FALSE(torch::softmax(scores.to(torch::kFloat32), -1).isnan().any().item<bool>());
}

TEST_F(Layers, AttentionMaskShapeAndValues) {
  const auto causal = gpt2::build_attention_mask(3, {}, torch::kCPU);
  EXPECT_EQ(causal.sizes(), (std::vector<int64_t>{1, 1, 3, 3}));
  EXPECT_TRUE(torch::equal(causal[0][0], torch::tensor({{1, 0, 0}, {1, 1, 0}, {1, 1, 1}}).to(torch::kBool)));

  const auto pad = torch::tensor({{0, 1, 1}, {1, 1, 1}}).to(torch::kBool);
  const auto masked = gpt2::build_attention_mask(3, pad, torch::kCPU);
  EXPECT_EQ(masked.sizes(), (std::vector<int64_t>{2, 1, 3, 3}));
  EXPECT_TRUE(torch::equal(masked[0][0], torch::tensor({{0, 0, 0}, {0, 1, 0}, {0, 1, 1}}).to(torch::kBool)));
  EXPECT_TRUE(torch::equal(masked[1][0], causal[0][0]));
}

TEST_F(Layers, AttentionIsCausal) {
  const auto attn = random_attention(16, 4);
  auto x = torch::randn({1, 6, 16});
  const auto mask = gpt2::build_attention_mask(6, {}, torch::kCPU);
  const auto before = attn.forward(x, mask);

  auto changed = x.clone();
  changed.slice(1, 3).normal_();  // perturb future tokens 3..5
  const auto after = attn.forward(changed, mask);

  EXPECT_TRUE(AllClose(after.slice(1, 0, 3), before.slice(1, 0, 3), 0.0, 1e-6));
  EXPECT_FALSE(torch::allclose(after.slice(1, 3), before.slice(1, 3)));
}

TEST_F(Layers, PaddedKeysAreIgnored) {
  const auto attn = random_attention(16, 4);
  const auto real = torch::randn({1, 4, 16});
  const auto padded = torch::cat({torch::randn({1, 2, 16}) * 100.0, real}, 1);  // 2 garbage pad tokens on the left
  const auto pad_mask = torch::tensor({{0, 0, 1, 1, 1, 1}}).to(torch::kBool);

  const auto expected = attn.forward(real, gpt2::build_attention_mask(4, {}, torch::kCPU));
  const auto got = attn.forward(padded, gpt2::build_attention_mask(6, pad_mask, torch::kCPU));

  EXPECT_TRUE(AllClose(got.slice(1, 2), expected, 1e-5, 1e-5));
  EXPECT_FALSE(got.isnan().any().item<bool>()) << "fully masked pad rows must not produce NaN";
}

TEST_F(Layers, BlockPreservesShape) {
  gpt2::Block b;
  b.ln_1 = {torch::ones({16}), torch::zeros({16}), 1e-5};
  b.ln_2 = {torch::ones({16}), torch::zeros({16}), 1e-5};
  b.attn = random_attention(16, 4);
  b.mlp.c_fc = random_linear(16, 64);
  b.mlp.c_proj = random_linear(64, 16);
  const auto x = torch::randn({2, 5, 16});
  EXPECT_EQ(b.forward(x, gpt2::build_attention_mask(5, {}, torch::kCPU)).sizes(), x.sizes());
}

TEST(Config, ValidatesDimensions) {
  gpt2::GPT2Config c;
  EXPECT_NO_THROW(c.validate());
  EXPECT_EQ(c.head_dim(), 64);
  c.n_head = 7;
  EXPECT_THROW(c.validate(), std::invalid_argument);
}

TEST(Config, LoadsExportedConfigIfPresent) {
  if (!gpt2::test::data_file_exists("config.json")) GTEST_SKIP() << "run scripts/export_weights.py first";
  const auto c = gpt2::GPT2Config::from_json_file(gpt2::test::data_dir() + "/config.json");
  EXPECT_EQ(c.n_layer, 12);
  EXPECT_EQ(c.n_embd, 768);
  EXPECT_EQ(c.vocab_size, 50257);
  EXPECT_EQ(c.eos_token_id, 50256);
}

}  // namespace
