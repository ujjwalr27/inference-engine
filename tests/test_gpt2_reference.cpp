// End-to-end check against Hugging Face outputs produced by scripts/reference.py.
#include <iostream>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "io/safetensors.h"
#include "model/gpt2.h"
#include "test_util.h"

namespace {

using gpt2::test::AllClose;

constexpr double kRtol = 1e-4;
constexpr double kAtol = 1e-4;

class Gpt2Reference : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!gpt2::test::data_file_exists("model.safetensors") ||
        !gpt2::test::data_file_exists("reference.safetensors")) {
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
    if (!model_) GTEST_SKIP() << "missing data in " << gpt2::test::data_dir()
                              << "; run scripts/export_weights.py and scripts/reference.py";
  }

  static int num_prompts() {
    int n = 0;
    while (ref_->contains("p" + std::to_string(n) + ".input_ids")) ++n;
    return n;
  }

  static std::unique_ptr<gpt2::GPT2Model> model_;
  static std::unique_ptr<gpt2::SafeTensors> ref_;
  torch::InferenceMode guard_;
};

std::unique_ptr<gpt2::GPT2Model> Gpt2Reference::model_;
std::unique_ptr<gpt2::SafeTensors> Gpt2Reference::ref_;

TEST_F(Gpt2Reference, HiddenStatesMatchPerLayer) {
  const auto ids = ref_->load("p0.input_ids").unsqueeze(0);
  const auto hidden = model_->hidden_states(ids);
  const auto n_layer = model_->config().n_layer;
  ASSERT_EQ(static_cast<int64_t>(hidden.size()), n_layer + 1);

  for (int64_t l = 0; l <= n_layer; ++l) {
    const std::string name = "p0.hidden." + std::to_string(l);
    ASSERT_TRUE(ref_->contains(name)) << name;
    // Stops at the first bad layer, which is the one to debug.
    ASSERT_TRUE(AllClose(hidden[static_cast<size_t>(l)][0], ref_->load(name), kRtol, kAtol))
        << "hidden state " << l << (l == 0 ? " (embeddings)" : l == n_layer ? " (after ln_f)" : "");
  }
}

TEST_F(Gpt2Reference, LogitsMatchHuggingFace) {
  const int n = num_prompts();
  ASSERT_GE(n, 5) << "reference data should contain at least 5 prompts";

  for (int i = 0; i < n; ++i) {
    const std::string p = "p" + std::to_string(i);
    const auto ids = ref_->load(p + ".input_ids");
    const auto positions = ref_->load(p + ".positions");
    const auto expected = ref_->load(p + ".logits");
    SCOPED_TRACE(p + " (" + std::to_string(ids.size(0)) + " tokens)");

    const auto logits = model_->forward(ids.unsqueeze(0))[0].index_select(0, positions);
    const auto result = AllClose(logits, expected, kRtol, kAtol);
    std::cout << "[          ] " << p << " T=" << ids.size(0) << " " << result.message() << "\n";
    EXPECT_TRUE(result);
    EXPECT_TRUE(torch::equal(logits.argmax(-1), expected.argmax(-1))) << "argmax differs";
  }
}

TEST_F(Gpt2Reference, BatchOfIdenticalRowsMatchesSingle) {
  const auto ids = ref_->load("p4.input_ids").unsqueeze(0);
  const auto single = model_->forward(ids);
  const auto batched = model_->forward(ids.expand({3, ids.size(1)}).contiguous());
  for (int64_t b = 0; b < 3; ++b) {
    EXPECT_TRUE(AllClose(batched[b], single[0], kRtol, kAtol)) << "row " << b;
  }
}

TEST_F(Gpt2Reference, RejectsTooLongInput) {
  const auto ids = torch::zeros({1, model_->config().n_ctx + 1}, torch::kInt64);
  EXPECT_THROW(model_->forward(ids), c10::Error);
}

}  // namespace
