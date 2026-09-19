// Phase 4: continuous batching must not change answers. Requests arriving at random times,
// sharing slots with each other, must produce exactly what they produce alone.
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "io/safetensors.h"
#include "model/generate.h"
#include "model/gpt2.h"
#include "scheduler/scheduler.h"
#include "test_util.h"

namespace {

using gpt2::test::first_difference;
using gpt2::test::to_vector;

constexpr double kNearTieGap = 1e-3;

// The stress test costs one solo run per request, so keep the default small and let CI or a
// manual run widen it: GPT2_STRESS=1 gives the 100 staggered requests from the plan.
int stress_requests() {
  const char* env = std::getenv("GPT2_STRESS");
  return (env && std::string(env) == "1") ? 100 : 16;
}

class SchedulerTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!gpt2::test::data_file_exists("model.safetensors") || !gpt2::test::data_file_exists("reference.safetensors")) {
      return;
    }
    model_ = std::make_unique<gpt2::GPT2Model>(gpt2::GPT2Model::load(gpt2::test::data_dir()));
    ref_ = std::make_unique<gpt2::SafeTensors>(
        gpt2::SafeTensors::open(gpt2::test::data_dir() + "/reference.safetensors"));
    corpus_ = to_vector(ref_->load("p5.input_ids"));  // 500 tokens of real text to slice prompts from
  }
  static void TearDownTestSuite() {
    model_.reset();
    ref_.reset();
  }
  void SetUp() override {
    if (!model_) GTEST_SKIP() << "run scripts/export_weights.py and scripts/reference.py first";
  }

  static std::vector<int64_t> slice_prompt(std::mt19937& rng) {
    std::uniform_int_distribution<size_t> length(3, 24);
    std::uniform_int_distribution<size_t> start(0, corpus_.size() - 32);
    const size_t begin = start(rng);
    const size_t n = length(rng);
    return std::vector<int64_t>(corpus_.begin() + static_cast<long>(begin),
                                corpus_.begin() + static_cast<long>(begin + n));
  }

  // What this prompt produces when it has the engine to itself.
  std::vector<int64_t> solo(const std::vector<int64_t>& prompt, int64_t max_new, gpt2::KVCache& cache) const {
    gpt2::GenerationOptions options;
    options.max_new_tokens = max_new;
    options.stop_on_eos = true;
    return gpt2::generate_cached(*model_, prompt, options, &cache).tokens;
  }

  static std::unique_ptr<gpt2::GPT2Model> model_;
  static std::unique_ptr<gpt2::SafeTensors> ref_;
  static std::vector<int64_t> corpus_;
};

std::unique_ptr<gpt2::GPT2Model> SchedulerTest::model_;
std::unique_ptr<gpt2::SafeTensors> SchedulerTest::ref_;
std::vector<int64_t> SchedulerTest::corpus_;

TEST_F(SchedulerTest, SingleRequestMatchesSoloGeneration) {
  const auto prompt = to_vector(ref_->load("p0.input_ids"));
  gpt2::KVCache solo_cache(model_->config(), 1, model_->device(), model_->dtype());
  const auto expected = solo(prompt, 20, solo_cache);

  gpt2::SchedulerOptions options;
  options.n_slots = 4;
  gpt2::Scheduler scheduler(*model_, options);
  scheduler.start();

  gpt2::GenerationOptions gen;
  gen.max_new_tokens = 20;
  auto request = scheduler.submit(prompt, gen);
  ASSERT_NE(request, nullptr);
  const auto tokens = request->out->collect();
  scheduler.stop();

  EXPECT_EQ(tokens, expected);
  EXPECT_EQ(request->finish_reason, gpt2::FinishReason::MaxTokens);
  EXPECT_EQ(scheduler.stats().finished, 1u);
}

// The Phase 4 headline test.
TEST_F(SchedulerTest, StaggeredRequestsMatchTheirSoloRuns) {
  const int n = stress_requests();
  std::mt19937 rng(20260918);
  std::uniform_int_distribution<int64_t> tokens_dist(4, 14);
  std::uniform_int_distribution<int> delay_dist(0, 8);

  struct Case {
    std::vector<int64_t> prompt;
    int64_t max_new;
    int delay_ms;
    std::vector<int64_t> expected;
  };
  std::vector<Case> cases;
  cases.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    cases.push_back({slice_prompt(rng), tokens_dist(rng), delay_dist(rng), {}});
  }

  // Expected outputs first, one request at a time, sharing a single-slot cache.
  gpt2::KVCache solo_cache(model_->config(), 1, model_->device(), model_->dtype());
  for (auto& c : cases) c.expected = solo(c.prompt, c.max_new, solo_cache);

  gpt2::SchedulerOptions options;
  options.n_slots = 6;  // fewer slots than requests, so slots are recycled and compacted
  options.max_queue = static_cast<size_t>(n);
  options.prefill_budget_tokens = 64;
  gpt2::Scheduler scheduler(*model_, options);
  scheduler.start();

  // Each sender waits for its own request, so the number of senders is the client concurrency.
  // Keep it above n_slots to make sure slots stay saturated and are recycled under pressure.
  constexpr int kSenders = 12;
  std::vector<std::vector<int64_t>> got(cases.size());
  std::atomic<int> submitted{0};
  std::vector<std::thread> senders;
  for (int t = 0; t < kSenders; ++t) {
    senders.emplace_back([&, t] {
      for (size_t i = static_cast<size_t>(t); i < cases.size(); i += kSenders) {
        std::this_thread::sleep_for(std::chrono::milliseconds(cases[i].delay_ms));
        gpt2::GenerationOptions gen;
        gen.max_new_tokens = cases[i].max_new;
        std::shared_ptr<gpt2::Request> request;
        while (!(request = scheduler.submit(cases[i].prompt, gen))) {
          std::this_thread::sleep_for(std::chrono::milliseconds(2));  // queue full: retry
        }
        submitted.fetch_add(1);
        got[i] = request->out->collect();
      }
    });
  }
  for (auto& s : senders) s.join();

  const auto stats = scheduler.stats();
  scheduler.stop();

  EXPECT_EQ(submitted.load(), n);
  EXPECT_EQ(stats.finished, static_cast<uint64_t>(n));
  EXPECT_EQ(stats.max_batch, options.n_slots) << "slots never filled up, so recycling was not exercised";
  std::cout << "[          ] " << n << " requests, max batch " << stats.max_batch << ", " << stats.decode_steps
            << " decode steps, " << stats.generated_tokens << " tokens\n";

  for (size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE("request " + std::to_string(i) + " (prompt " + std::to_string(cases[i].prompt.size()) +
                 " tokens, " + std::to_string(cases[i].max_new) + " new)");
    const int64_t d = first_difference(got[i], cases[i].expected);
    EXPECT_EQ(d, -1) << "continuous batching changed this request's output";
    if (d >= 0) {  // tolerated only at a near-tie between the top two logits
      std::vector<int64_t> prefix = cases[i].prompt;
      prefix.insert(prefix.end(), cases[i].expected.begin(), cases[i].expected.begin() + d);
      const double gap =
          gpt2::test::top2_gap(model_->forward(torch::tensor(prefix, torch::kInt64).unsqueeze(0))[0][-1]);
      std::cout << "[          ] request " << i << " diverged at token " << d << ", top-2 gap " << gap << "\n";
      EXPECT_LT(gap, kNearTieGap) << "divergence at a clear argmax is a real bug";
    }
  }
}

TEST_F(SchedulerTest, CancellationFreesTheSlotWithoutDisturbingOthers) {
  const auto prompt_a = to_vector(ref_->load("p0.input_ids"));
  const auto prompt_b = to_vector(ref_->load("p4.input_ids"));
  gpt2::KVCache solo_cache(model_->config(), 1, model_->device(), model_->dtype());
  const auto expected_b = solo(prompt_b, 25, solo_cache);

  gpt2::SchedulerOptions options;
  options.n_slots = 2;
  gpt2::Scheduler scheduler(*model_, options);
  scheduler.start();

  gpt2::GenerationOptions gen;
  gen.max_new_tokens = 25;
  auto a = scheduler.submit(prompt_a, gen);
  auto b = scheduler.submit(prompt_b, gen);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);

  int64_t token = 0;
  ASSERT_TRUE(a->out->next(token));  // wait until A is really running
  a->cancel();

  const auto tokens_a = a->out->collect();
  const auto tokens_b = b->out->collect();
  const auto stats = scheduler.stats();
  scheduler.stop();

  EXPECT_LT(tokens_a.size() + 1, static_cast<size_t>(gen.max_new_tokens)) << "cancel should cut A short";
  EXPECT_EQ(a->finish_reason, gpt2::FinishReason::Cancelled);
  EXPECT_EQ(tokens_b, expected_b) << "B's output must be unaffected by A being cancelled mid-flight";
  EXPECT_EQ(stats.cancelled, 1u);
}

TEST_F(SchedulerTest, FullQueueIsRejected) {
  gpt2::SchedulerOptions options;
  options.n_slots = 1;
  options.max_queue = 2;
  gpt2::Scheduler scheduler(*model_, options);  // deliberately not started: nothing drains the queue

  gpt2::GenerationOptions gen;
  gen.max_new_tokens = 2;
  EXPECT_NE(scheduler.submit({15496}, gen), nullptr);
  EXPECT_NE(scheduler.submit({15496}, gen), nullptr);
  EXPECT_EQ(scheduler.submit({15496}, gen), nullptr) << "third request must be rejected, not queued";
  EXPECT_EQ(scheduler.stats().rejected, 1u);
  EXPECT_EQ(scheduler.stats().submitted, 3u);
}

TEST_F(SchedulerTest, RejectsImpossibleRequests) {
  gpt2::Scheduler scheduler(*model_, gpt2::SchedulerOptions{});
  gpt2::GenerationOptions gen;
  gen.max_new_tokens = 5;
  EXPECT_THROW(scheduler.submit({}, gen), std::invalid_argument);
  gen.max_new_tokens = model_->config().n_ctx;
  EXPECT_THROW(scheduler.submit({15496}, gen), std::invalid_argument);
}

TEST_F(SchedulerTest, StopClosesPendingRequests) {
  gpt2::SchedulerOptions options;
  options.n_slots = 1;
  options.max_queue = 8;
  gpt2::Scheduler scheduler(*model_, options);

  gpt2::GenerationOptions gen;
  gen.max_new_tokens = 5;
  auto pending = scheduler.submit(to_vector(ref_->load("p0.input_ids")), gen);
  ASSERT_NE(pending, nullptr);
  scheduler.stop();  // never started: the request is still waiting in the queue

  EXPECT_TRUE(pending->out->closed());
  EXPECT_FALSE(pending->out->error().empty());
}

}  // namespace
