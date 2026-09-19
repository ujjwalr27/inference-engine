// Tokenizer parity with Python (weights/tokenizer_cases.json) and streaming-decode behaviour.
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "test_util.h"
#include "tokenizer/tokenizer.h"

namespace {

class TokenizerTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!gpt2::test::data_file_exists("tokenizer.json")) return;
    tok_ = std::make_unique<gpt2::Tokenizer>(gpt2::Tokenizer::from_file(gpt2::test::data_dir() + "/tokenizer.json"));
  }
  static void TearDownTestSuite() { tok_.reset(); }
  void SetUp() override {
    if (!tok_) GTEST_SKIP() << "run scripts/export_weights.py first";
  }
  static std::unique_ptr<gpt2::Tokenizer> tok_;
};

std::unique_ptr<gpt2::Tokenizer> TokenizerTest::tok_;

TEST_F(TokenizerTest, VocabSizeAndKnownIds) {
  EXPECT_EQ(tok_->vocab_size(), 50257u);
  EXPECT_EQ(tok_->encode("Hello"), (std::vector<int64_t>{15496}));
  EXPECT_EQ(tok_->decode({15496, 995}), "Hello world");
  EXPECT_TRUE(tok_->encode("").empty());
}

TEST_F(TokenizerTest, MatchesPythonOnAllCases) {
  if (!gpt2::test::data_file_exists("tokenizer_cases.json")) GTEST_SKIP() << "run scripts/reference.py first";
  std::ifstream in(gpt2::test::data_dir() + "/tokenizer_cases.json");
  ASSERT_TRUE(in.good());
  const auto cases = nlohmann::json::parse(in);
  ASSERT_GE(cases.size(), 1000u);

  size_t mismatches = 0;
  for (const auto& c : cases) {
    const auto text = c.at("text").get<std::string>();
    const auto expected = c.at("ids").get<std::vector<int64_t>>();
    const auto got = tok_->encode(text);
    if (got != expected) {
      if (++mismatches <= 5) ADD_FAILURE() << "encode mismatch for " << nlohmann::json(text).dump();
      continue;
    }
    EXPECT_EQ(tok_->decode(got), c.at("decoded").get<std::string>());
  }
  EXPECT_EQ(mismatches, 0u) << mismatches << " of " << cases.size() << " cases differ from Python";
}

TEST_F(TokenizerTest, IncrementalDecodeNeverEmitsPartialCharacters) {
  // Emoji and CJK text spans several tokens per character - the hard case for streaming.
  const std::string text = "Café 日本語 🚀👨‍👩‍👧‍👦 done";
  const auto ids = tok_->encode(text);
  ASSERT_GT(ids.size(), 5u);

  gpt2::IncrementalDecoder decoder(*tok_);
  std::string streamed;
  for (int64_t id : ids) {
    const std::string piece = decoder.push(id);
    EXPECT_EQ(piece.find("\xEF\xBF\xBD"), std::string::npos) << "emitted a replacement character";
    streamed += piece;
  }
  streamed += decoder.flush();
  EXPECT_EQ(streamed, tok_->decode(ids));
  EXPECT_EQ(streamed, text);
}

TEST_F(TokenizerTest, IncrementalDecodeHoldsBackThenReleases) {
  const auto ids = tok_->encode("🚀");
  ASSERT_GT(ids.size(), 1u) << "this test needs a character split across tokens";

  gpt2::IncrementalDecoder decoder(*tok_);
  EXPECT_TRUE(decoder.push(ids.front()).empty()) << "an incomplete character must be held back";
  std::string rest;
  for (size_t i = 1; i < ids.size(); ++i) rest += decoder.push(ids[i]);
  rest += decoder.flush();
  EXPECT_EQ(rest, "🚀");
  EXPECT_EQ(decoder.tokens().size(), ids.size());
}

TEST_F(TokenizerTest, RejectsMissingFile) {
  EXPECT_THROW(gpt2::Tokenizer::from_file("/nonexistent/tokenizer.json"), std::runtime_error);
}

}  // namespace
