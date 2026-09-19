// Phase 2 correctness: the cached path must match the uncached path, and both must match
// Hugging Face's greedy output (scripts/reference.py).
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "cache/kv_cache.h"
#include "io/safetensors.h"
#include "model/generate.h"
#include "model/gpt2.h"
#include "test_util.h"

namespace {

using gpt2::test::AllClose;

constexpr double kRtol = 1e-4;
constexpr double kAtol = 1e-4;
constexpr double kNearTieGap = 1e-3;

using gpt2::test::first_difference;
using gpt2::test::to_vector;

class Generation : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!gpt2::test::data_file_exists("model.safetensors") || !gpt2::test::data_file_exists("reference.safetensors")) {
      return;
    }
    model_ = std::make_unique<gpt2::GPT2Model>(gpt2::GPT2Model::load(gpt2::test::data_dir()));
    ref_ = std::make_unique<gpt2::SafeTensors>(
        gpt2::SafeTensors::open(gpt2::test::data_dir() + "/reference.safetensors"));
  }
  static void TearDownTestSuite() {
    model_.reset();
    ref_.reset();
  }
  void SetUp() override {
    if (!model_) GTEST_SKIP() << "run scripts/export_weights.py and scripts/reference.py first";
  }

  static std::unique_ptr<gpt2::GPT2Model> model_;
  static std::unique_ptr<gpt2::SafeTensors> ref_;
  torch::InferenceMode guard_;
};

std::unique_ptr<gpt2::GPT2Model> Generation::model_;
std::unique_ptr<gpt2::SafeTensors> Generation::ref_;

// Teacher forcing: feed the same known tokens through both paths and compare logits at every
// step. This is the strict check; token equality below can legitimately diverge at a near-tie.
TEST_F(Generation, CachedLogitsMatchUncachedAtEveryStep) {
  const auto ids = to_vector(ref_->load("p0.input_ids"));
  ASSERT_GE(ids.size(), 4u);

  gpt2::KVCache cache(model_->config(), 1, model_->device(), model_->dtype());
  const std::vector<int64_t> first(ids.begin(), ids.begin() + 2);
  auto cached = model_->prefill(torch::tensor(first, torch::kInt64).unsqueeze(0), 0, cache);

  for (size_t t = 2; t <= ids.size(); ++t) {
    const std::vector<int64_t> prefix(ids.begin(), ids.begin() + static_cast<long>(t));
    const auto uncached = model_->forward(torch::tensor(prefix, torch::kInt64).unsqueeze(0))[0][-1];
    SCOPED_TRACE("prefix length " + std::to_string(t));
    EXPECT_TRUE(AllClose(cached[0], uncached, kRtol, kAtol));

    if (t < ids.size()) {  // feed ids[t] and advance the cache
      cached = model_->decode(torch::tensor({ids[t]}, torch::kInt64),
                              torch::tensor({static_cast<int64_t>(t)}, torch::kInt64),
                              static_cast<int64_t>(t) + 1, cache);
    }
  }
}

TEST_F(Generation, CachedUncachedAndHuggingFaceAgree) {
  int prompts_checked = 0;
  for (int i = 0; i < 8; ++i) {
    const std::string p = "p" + std::to_string(i);
    if (!ref_->contains(p + ".greedy")) continue;
    ++prompts_checked;

    const auto prompt = to_vector(ref_->load(p + ".input_ids"));
    const auto expected = to_vector(ref_->load(p + ".greedy"));
    SCOPED_TRACE(p + ": prompt " + std::to_string(prompt.size()) + " tokens, " + std::to_string(expected.size()) +
                 " generated");

    gpt2::GenerationOptions options;
    options.max_new_tokens = static_cast<int64_t>(expected.size());
    options.stop_on_eos = false;  // HF reference generated a fixed number of tokens

    options.use_cache = false;
    const auto uncached = gpt2::generate_uncached(*model_, prompt, options);
    options.use_cache = true;
    const auto cached = gpt2::generate_cached(*model_, prompt, options);

    EXPECT_EQ(first_difference(cached.tokens, uncached.tokens), -1) << "cached vs uncached";
    EXPECT_EQ(first_difference(cached.tokens, expected), -1) << "cached vs Hugging Face";

    // If tokens do diverge, it is only acceptable at a near-tie between the top two logits.
    const int64_t d = first_difference(cached.tokens, expected);
    if (d >= 0) {
      std::vector<int64_t> prefix = prompt;
      prefix.insert(prefix.end(), expected.begin(), expected.begin() + d);
      const auto logits = model_->forward(torch::tensor(prefix, torch::kInt64).unsqueeze(0))[0][-1];
      const double gap = gpt2::test::top2_gap(logits);
      std::cout << "[          ] " << p << " diverged at token " << d << ", top-2 gap " << gap << "\n";
      EXPECT_LT(gap, kNearTieGap) << "divergence at a clear (non-tie) argmax is a real bug";
    }
  }
  EXPECT_GE(prompts_checked, 3) << "expected greedy references for at least 3 prompts";
}

TEST_F(Generation, CacheIsMuchFasterForLongOutputs) {
  const auto prompt = to_vector(ref_->load("p4.input_ids"));
  gpt2::GenerationOptions options;
  options.max_new_tokens = 50;
  options.stop_on_eos = false;

  options.use_cache = false;
  const auto uncached = gpt2::generate_uncached(*model_, prompt, options);
  options.use_cache = true;
  const auto cached = gpt2::generate_cached(*model_, prompt, options);

  std::cout << "[          ] decode ms/token: uncached " << uncached.stats.ms_per_token() << ", cached "
            << cached.stats.ms_per_token() << " (speedup "
            << uncached.stats.ms_per_token() / cached.stats.ms_per_token() << "x)\n";
  EXPECT_LT(cached.stats.ms_per_token(), uncached.stats.ms_per_token());
}

TEST_F(Generation, StopsAtEosAndValidatesLength) {
  gpt2::GenerationOptions options;
  options.max_new_tokens = 3;
  const auto result = gpt2::generate_cached(*model_, {15496}, options);  // "Hello"
  EXPECT_EQ(result.tokens.size(), 3u);
  EXPECT_EQ(result.stats.steps, 3);

  options.max_new_tokens = model_->config().n_ctx;  // prompt + new tokens exceeds the context
  EXPECT_THROW(gpt2::generate_cached(*model_, {15496}, options), std::invalid_argument);
  EXPECT_THROW(gpt2::generate_cached(*model_, {}, options), std::invalid_argument);
}

TEST_F(Generation, EosTokenStopsGeneration) {
  gpt2::GenerationOptions options;
  options.max_new_tokens = 5;
  options.eos_token_id = -1;
  // Force an immediate stop by declaring the first generated token to be EOS.
  const auto probe = gpt2::generate_cached(*model_, {15496}, options);
  ASSERT_FALSE(probe.tokens.empty());
  options.eos_token_id = probe.tokens.front();
  const auto stopped = gpt2::generate_cached(*model_, {15496}, options);
  EXPECT_TRUE(stopped.hit_eos);
  EXPECT_EQ(stopped.tokens.size(), 1u);
}

}  // namespace
