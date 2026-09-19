// Phase 3: padding, per-row positions, and the rule that batching must not change answers.
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "io/safetensors.h"
#include "model/generate.h"
#include "model/gpt2.h"
#include "scheduler/padding.h"
#include "scheduler/static_batch.h"
#include "test_util.h"

namespace {

using gpt2::test::AllClose;
using gpt2::test::first_difference;
using gpt2::test::to_vector;

constexpr double kRtol = 1e-4;
constexpr double kAtol = 1e-4;
constexpr double kNearTieGap = 1e-3;

TEST(Padding, LeftPadsAndNumbersRealTokensFromZero) {
  const auto batch = gpt2::pad_left({{1, 2, 3}, {4}}, /*pad_token_id=*/0);
  EXPECT_EQ(batch.batch_size(), 2);
  EXPECT_EQ(batch.max_length, 3);
  EXPECT_EQ(batch.lengths, (std::vector<int64_t>{3, 1}));

  EXPECT_TRUE(torch::equal(batch.ids, torch::tensor({{1, 2, 3}, {0, 0, 4}}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(batch.padding_mask, torch::tensor({{1, 1, 1}, {0, 0, 1}}).to(torch::kBool)));
  // Row 1's single real token is position 0, not position 2.
  EXPECT_TRUE(torch::equal(batch.positions, torch::tensor({{0, 1, 2}, {0, 0, 0}}, torch::kInt64)));
  EXPECT_NEAR(batch.padding_waste(), 1.0 - 4.0 / 6.0, 1e-9);
}

TEST(Padding, RejectsEmptyInput) {
  EXPECT_THROW(gpt2::pad_left({}), std::invalid_argument);
  EXPECT_THROW(gpt2::pad_left({{1, 2}, {}}), std::invalid_argument);
}

class Batching : public ::testing::Test {
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

  // Prompts of deliberately different lengths: 1, 10, 26 and 31 tokens.
  static std::vector<std::vector<int64_t>> mixed_prompts() {
    std::vector<std::vector<int64_t>> prompts;
    for (const char* p : {"p1", "p0", "p4", "p2"}) {
      prompts.push_back(to_vector(ref_->load(std::string(p) + ".input_ids")));
    }
    return prompts;
  }

  static std::unique_ptr<gpt2::GPT2Model> model_;
  static std::unique_ptr<gpt2::SafeTensors> ref_;
  torch::InferenceMode guard_;
};

std::unique_ptr<gpt2::GPT2Model> Batching::model_;
std::unique_ptr<gpt2::SafeTensors> Batching::ref_;

// Strict version of the invariance rule: the padded batched prefill must produce the same
// next-token logits as running each prompt on its own.
TEST_F(Batching, PaddedPrefillLogitsMatchPerPrompt) {
  const auto prompts = mixed_prompts();
  const auto batch = gpt2::pad_left(prompts, model_->config().eos_token_id);
  gpt2::KVCache cache(model_->config(), batch.batch_size(), model_->device(), model_->dtype());
  const auto batched = model_->prefill_batch(batch.ids, batch.positions, batch.padding_mask, cache);

  for (size_t i = 0; i < prompts.size(); ++i) {
    gpt2::KVCache single_cache(model_->config(), 1, model_->device(), model_->dtype());
    const auto single =
        model_->prefill(torch::tensor(prompts[i], torch::kInt64).unsqueeze(0), 0, single_cache);
    SCOPED_TRACE("row " + std::to_string(i) + " (" + std::to_string(prompts[i].size()) + " tokens)");
    EXPECT_TRUE(AllClose(batched[static_cast<int64_t>(i)], single[0], kRtol, kAtol));
  }
}

// The Phase 3 headline test: same prompts, alone vs batched together, same text.
TEST_F(Batching, BatchingDoesNotChangeAnswers) {
  const auto prompts = mixed_prompts();
  gpt2::GenerationOptions options;
  options.max_new_tokens = 30;
  options.stop_on_eos = false;

  const auto batched = gpt2::generate_batch_static(*model_, prompts, options);
  ASSERT_EQ(batched.tokens.size(), prompts.size());
  std::cout << "[          ] padding waste " << batched.padding_waste * 100.0 << "% of the prompt block\n";

  for (size_t i = 0; i < prompts.size(); ++i) {
    const auto alone = gpt2::generate_cached(*model_, prompts[i], options).tokens;
    SCOPED_TRACE("row " + std::to_string(i) + " (" + std::to_string(prompts[i].size()) + " tokens)");
    const int64_t d = first_difference(batched.tokens[i], alone);
    EXPECT_EQ(d, -1) << "batched output differs from running this prompt alone";

    if (d >= 0) {  // tolerated only at a near-tie between the top two logits
      std::vector<int64_t> prefix = prompts[i];
      prefix.insert(prefix.end(), alone.begin(), alone.begin() + d);
      const double gap = gpt2::test::top2_gap(model_->forward(torch::tensor(prefix, torch::kInt64).unsqueeze(0))[0][-1]);
      std::cout << "[          ] row " << i << " diverged at token " << d << ", top-2 gap " << gap << "\n";
      EXPECT_LT(gap, kNearTieGap) << "divergence at a clear argmax is a real bug";
    }
  }
  EXPECT_EQ(batched.wasted_decode_rows, 0) << "no row hit EOS, so nothing should have been wasted";
  EXPECT_EQ(batched.stats.steps, options.max_new_tokens);
}

TEST_F(Batching, BatchOfOneMatchesUnbatched) {
  const auto prompt = to_vector(ref_->load("p0.input_ids"));
  gpt2::GenerationOptions options;
  options.max_new_tokens = 20;
  options.stop_on_eos = false;

  const auto batched = gpt2::generate_batch_static(*model_, {prompt}, options);
  const auto alone = gpt2::generate_cached(*model_, prompt, options);
  EXPECT_EQ(batched.tokens[0], alone.tokens);
  EXPECT_EQ(batched.padding_waste, 0.0);
}

TEST_F(Batching, ReportsWasteWhenARowFinishesEarly) {
  const auto prompts = mixed_prompts();
  gpt2::GenerationOptions options;
  options.max_new_tokens = 12;
  // Declare the first token of row 0 to be EOS: that row stops immediately while the others run on.
  const auto probe = gpt2::generate_batch_static(*model_, prompts, options);
  options.eos_token_id = probe.tokens[0].front();

  const auto result = gpt2::generate_batch_static(*model_, prompts, options);
  EXPECT_TRUE(result.hit_eos[0]);
  EXPECT_EQ(result.tokens[0].size(), 1u);
  EXPECT_GT(result.wasted_decode_rows, 0) << "finished rows still occupy the static batch";
  EXPECT_GT(result.tokens[1].size(), 1u) << "other rows keep going";
}

TEST_F(Batching, StreamsTokensPerRow) {
  const auto prompts = mixed_prompts();
  gpt2::GenerationOptions options;
  options.max_new_tokens = 5;
  options.stop_on_eos = false;

  std::vector<std::vector<int64_t>> streamed(prompts.size());
  const auto result = gpt2::generate_batch_static(*model_, prompts, options, nullptr,
                                                  [&](size_t row, int64_t token) { streamed[row].push_back(token); });
  EXPECT_EQ(streamed, result.tokens);
}

TEST_F(Batching, ValidatesInput) {
  gpt2::GenerationOptions options;
  options.max_new_tokens = 4;
  EXPECT_THROW(gpt2::generate_batch_static(*model_, {}, options), std::invalid_argument);

  gpt2::KVCache one_slot(model_->config(), 1, model_->device(), model_->dtype());
  EXPECT_THROW(gpt2::generate_batch_static(*model_, {{1, 2}, {3, 4}}, options, &one_slot), std::invalid_argument);

  options.max_new_tokens = model_->config().n_ctx;  // prompt + new tokens exceeds the context
  EXPECT_THROW(gpt2::generate_batch_static(*model_, {{1, 2}}, options), std::invalid_argument);
}

}  // namespace
