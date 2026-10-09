// Tokenizer parity with Python (weights/tokenizer_cases.json) and streaming-decode behaviour.
#include <atomic>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <thread>
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

// The decoder only re-decodes the tokens since the last emitted character. Streams made mostly of
// tokens that are partial characters on their own (lone UTF-8 bytes) are the hardest case for
// that: the streamed text must still equal decoding everything at once.
TEST_F(TokenizerTest, IncrementalDecodeMatchesFullDecodeOnRandomStreams) {
  const auto vocab = static_cast<int64_t>(tok_->vocab_size());
  std::vector<int64_t> partial;  // tokens that decode to an incomplete character by themselves
  for (int64_t id = 0; id < vocab; ++id) {
    if (tok_->decode({id}).find("\xEF\xBF\xBD") != std::string::npos) partial.push_back(id);
  }
  ASSERT_GT(partial.size(), 100u);

  std::mt19937 rng(7);
  std::uniform_int_distribution<int64_t> any(0, vocab - 1);
  std::uniform_int_distribution<size_t> pick(0, partial.size() - 1);
  std::uniform_int_distribution<int> length(1, 60);
  for (int n = 0; n < 300; ++n) {
    std::vector<int64_t> ids(static_cast<size_t>(length(rng)));
    for (auto& id : ids) id = (rng() % 10 < 6) ? partial[pick(rng)] : any(rng);

    gpt2::IncrementalDecoder decoder(*tok_);
    std::string streamed;
    for (int64_t id : ids) streamed += decoder.push(id);
    streamed += decoder.flush();
    ASSERT_EQ(streamed, tok_->decode(ids)) << "stream #" << n;
  }
}

// The server shares one tokenizer between all worker threads. Each thread decodes different text,
// so a race inside the tokenizer would show up as a thread reading back another thread's result.
TEST_F(TokenizerTest, OneInstanceIsSafeToShareAcrossThreads) {
  const std::vector<std::string> texts = {"The quick brown fox", "日本語のテキスト", "🚀 launch sequence",
                                          "def fibonacci(n):", "Café naïve résumé", "1234567890"};
  std::vector<std::vector<int64_t>> ids;
  for (const auto& t : texts) ids.push_back(tok_->encode(t));

  std::atomic<int> mismatches{0};
  std::vector<std::thread> threads;
  for (size_t t = 0; t < 8; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < 300; ++i) {
        const size_t k = (t + static_cast<size_t>(i)) % texts.size();
        if (tok_->decode(ids[k]) != texts[k]) ++mismatches;
        if (tok_->encode(texts[k]) != ids[k]) ++mismatches;
      }
    });
  }
  for (auto& th : threads) th.join();
  EXPECT_EQ(mismatches.load(), 0) << "a shared tokenizer returned another thread's result";
}

TEST_F(TokenizerTest, RejectsMissingFile) {
  EXPECT_THROW(gpt2::Tokenizer::from_file("/nonexistent/tokenizer.json"), std::runtime_error);
}

}  // namespace
