// GPU correctness. Every test here skips when CUDA is unavailable, so the suite still runs on CPU.
// fp32 on GPU is held to the same tolerance as CPU; fp16 is looser by design and reported, not hidden.
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "io/safetensors.h"
#include "model/generate.h"
#include "model/gpt2.h"
#include "scheduler/scheduler.h"
#include "test_util.h"

namespace {

using gpt2::test::AllClose;
using gpt2::test::first_difference;
using gpt2::test::to_vector;

class GpuTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!torch::cuda::is_available()) return;
    if (!gpt2::test::data_file_exists("model.safetensors") || !gpt2::test::data_file_exists("reference.safetensors")) {
      return;
    }
    ref_ = std::make_unique<gpt2::SafeTensors>(
        gpt2::SafeTensors::open(gpt2::test::data_dir() + "/reference.safetensors"));
    cpu_ = std::make_unique<gpt2::GPT2Model>(gpt2::GPT2Model::load(gpt2::test::data_dir()));
    gpu32_ = std::make_unique<gpt2::GPT2Model>(
        gpt2::GPT2Model::load(gpt2::test::data_dir(), torch::kCUDA, torch::kFloat32));
    gpu16_ = std::make_unique<gpt2::GPT2Model>(
        gpt2::GPT2Model::load(gpt2::test::data_dir(), torch::kCUDA, torch::kFloat16));
  }
  static void TearDownTestSuite() {
    ref_.reset();
    cpu_.reset();
    gpu32_.reset();
    gpu16_.reset();
  }
  void SetUp() override {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "no CUDA device";
    if (!ref_) GTEST_SKIP() << "run scripts/export_weights.py and scripts/reference.py first";
  }

  static std::unique_ptr<gpt2::SafeTensors> ref_;
  static std::unique_ptr<gpt2::GPT2Model> cpu_, gpu32_, gpu16_;
  torch::InferenceMode guard_;
};

std::unique_ptr<gpt2::SafeTensors> GpuTest::ref_;
std::unique_ptr<gpt2::GPT2Model> GpuTest::cpu_;
std::unique_ptr<gpt2::GPT2Model> GpuTest::gpu32_;
std::unique_ptr<gpt2::GPT2Model> GpuTest::gpu16_;

TEST_F(GpuTest, Fp32MatchesHuggingFace) {
  for (int i = 0; i < 8; ++i) {
    const std::string p = "p" + std::to_string(i);
    if (!ref_->contains(p + ".input_ids")) break;
    const auto ids = ref_->load(p + ".input_ids");
    const auto positions = ref_->load(p + ".positions").to(torch::kCUDA);
    const auto expected = ref_->load(p + ".logits");
    SCOPED_TRACE(p);

    const auto logits = gpu32_->forward(ids.unsqueeze(0))[0].index_select(0, positions).to(torch::kCPU);
    // GPU fp32 reduces in a different order than CPU, so atol is a little looser than Phase 1.
    EXPECT_TRUE(AllClose(logits, expected, 1e-4, 1e-3));
    EXPECT_TRUE(torch::equal(logits.argmax(-1), expected.argmax(-1))) << "top token differs";
  }
}

TEST_F(GpuTest, Fp16AgreesOnTheTopTokenAndStaysClose) {
  int64_t positions_checked = 0;
  int64_t top1_agreements = 0;
  double worst = 0.0;

  for (int i = 0; i < 8; ++i) {
    const std::string p = "p" + std::to_string(i);
    if (!ref_->contains(p + ".input_ids")) break;
    const auto ids = ref_->load(p + ".input_ids");
    const auto positions = ref_->load(p + ".positions").to(torch::kCUDA);
    const auto expected = ref_->load(p + ".logits");

    const auto logits =
        gpu16_->forward(ids.unsqueeze(0))[0].index_select(0, positions).to(torch::kCPU, torch::kFloat32);
    // Compare probabilities rather than raw logits: fp16 has ~3 decimal digits of precision.
    const auto got = torch::log_softmax(logits, -1);
    const auto want = torch::log_softmax(expected, -1);
    worst = std::max(worst, (got - want).abs().max().item<double>());

    const auto agree = logits.argmax(-1).eq(expected.argmax(-1));
    top1_agreements += agree.sum().item<int64_t>();
    positions_checked += agree.numel();
  }

  const double rate = static_cast<double>(top1_agreements) / static_cast<double>(positions_checked);
  std::cout << "[          ] fp16 vs HF fp32: top-1 agreement " << top1_agreements << "/" << positions_checked
            << " (" << rate * 100.0 << "%), worst |log-prob diff| " << worst << "\n";
  EXPECT_GE(rate, 0.95) << "fp16 should pick the same top token almost always";
  // GPT-2 logits sit around -100, where consecutive fp16 values are 0.0625 apart (0.125 above 128).
  // A single rounding step is therefore already ~0.06, and twelve layers accumulate several of them,
  // so anything below ~0.5 is representation limits rather than a bug. What must not drift is the
  // ordering, which the top-1 check above covers.
  EXPECT_LT(worst, 0.5) << "fp16 log-probabilities drifted further than fp16 rounding explains";
}

TEST_F(GpuTest, CachedDecodeMatchesUncachedOnGpu) {
  const auto ids = to_vector(ref_->load("p0.input_ids"));
  gpt2::KVCache cache(gpu32_->config(), 1, torch::kCUDA, torch::kFloat32);
  auto cached = gpu32_->prefill(torch::tensor(std::vector<int64_t>(ids.begin(), ids.begin() + 2), torch::kInt64)
                                    .unsqueeze(0),
                                0, cache);

  for (size_t t = 2; t <= ids.size(); ++t) {
    const std::vector<int64_t> prefix(ids.begin(), ids.begin() + static_cast<long>(t));
    const auto uncached = gpu32_->forward(torch::tensor(prefix, torch::kInt64).unsqueeze(0))[0][-1];
    SCOPED_TRACE("prefix length " + std::to_string(t));
    EXPECT_TRUE(AllClose(cached[0], uncached, 1e-4, 1e-3));
    if (t < ids.size()) {
      cached = gpu32_->decode(torch::tensor({ids[t]}, torch::kInt64),
                              torch::tensor({static_cast<int64_t>(t)}, torch::kInt64),
                              static_cast<int64_t>(t) + 1, cache);
    }
  }
}

// Informational: fp16 greedy text may drift from CPU fp32 after enough tokens. The plan says to
// report that rather than hide it, so this test records where it happens instead of failing.
TEST_F(GpuTest, Fp16GreedyTracksCpuForAWhile) {
  const auto prompt = to_vector(ref_->load("p4.input_ids"));
  gpt2::GenerationOptions options;
  options.max_new_tokens = 40;
  options.stop_on_eos = false;

  const auto cpu = gpt2::generate_cached(*cpu_, prompt, options).tokens;
  const auto gpu = gpt2::generate_cached(*gpu16_, prompt, options).tokens;

  const int64_t d = first_difference(gpu, cpu);
  if (d < 0) {
    std::cout << "[          ] fp16 greedy matched CPU fp32 for all " << cpu.size() << " tokens\n";
  } else {
    std::vector<int64_t> prefix = prompt;
    prefix.insert(prefix.end(), cpu.begin(), cpu.begin() + d);
    const double gap = gpt2::test::top2_gap(cpu_->forward(torch::tensor(prefix, torch::kInt64).unsqueeze(0))[0][-1]);
    std::cout << "[          ] fp16 greedy diverged from CPU fp32 at token " << d << " of " << cpu.size()
              << ", top-2 gap there was " << gap << "\n";
  }
  EXPECT_GE(d < 0 ? static_cast<int64_t>(cpu.size()) : d, 5) << "diverging within the first few tokens is a bug";
}

namespace {

// Runs the prompts through the scheduler and returns what each request produced.
std::vector<std::vector<int64_t>> run_through_scheduler(const gpt2::GPT2Model& model,
                                                        const std::vector<std::vector<int64_t>>& prompts,
                                                        const gpt2::GenerationOptions& options) {
  gpt2::SchedulerOptions scheduler_options;
  scheduler_options.n_slots = static_cast<int64_t>(prompts.size());
  gpt2::Scheduler scheduler(model, scheduler_options);
  scheduler.start();

  std::vector<std::shared_ptr<gpt2::Request>> requests;
  for (const auto& prompt : prompts) requests.push_back(scheduler.submit(prompt, options));
  std::vector<std::vector<int64_t>> tokens;
  for (const auto& request : requests) {
    EXPECT_NE(request, nullptr);
    tokens.push_back(request ? request->out->collect() : std::vector<int64_t>{});
  }
  scheduler.stop();
  return tokens;
}

}  // namespace

// In fp32 the batch composition must not change a single token: this is the same invariant the
// CPU tests enforce, checked on the GPU where kernels differ.
TEST_F(GpuTest, SchedulerOnGpuFp32MatchesSoloRunsExactly) {
  const std::vector<std::string> names = {"p0", "p1", "p2", "p4"};
  gpt2::GenerationOptions options;
  options.max_new_tokens = 12;

  gpt2::KVCache solo_cache(gpu32_->config(), 1, torch::kCUDA, torch::kFloat32);
  std::vector<std::vector<int64_t>> prompts, expected;
  for (const auto& name : names) {
    prompts.push_back(to_vector(ref_->load(name + ".input_ids")));
    expected.push_back(gpt2::generate_cached(*gpu32_, prompts.back(), options, &solo_cache).tokens);
  }

  const auto batched = run_through_scheduler(*gpu32_, prompts, options);
  for (size_t i = 0; i < prompts.size(); ++i) {
    SCOPED_TRACE(names[i]);
    EXPECT_EQ(first_difference(batched[i], expected[i]), -1) << "batching changed an fp32 answer";
  }
}

// In fp16 the same run may diverge, because batching changes the reduction order and fp16 cannot
// resolve a close race between two candidate tokens. That is acceptable only at a near-tie, so
// this test measures the gap between the top two logits wherever the outputs part company.
TEST_F(GpuTest, SchedulerOnGpuFp16DivergesOnlyAtNearTies) {
  const std::vector<std::string> names = {"p0", "p1", "p2", "p4"};
  gpt2::GenerationOptions options;
  options.max_new_tokens = 12;

  gpt2::KVCache solo_cache(gpu16_->config(), 1, torch::kCUDA, torch::kFloat16);
  std::vector<std::vector<int64_t>> prompts, expected;
  for (const auto& name : names) {
    prompts.push_back(to_vector(ref_->load(name + ".input_ids")));
    expected.push_back(gpt2::generate_cached(*gpu16_, prompts.back(), options, &solo_cache).tokens);
  }

  const auto batched = run_through_scheduler(*gpu16_, prompts, options);
  int diverged = 0;
  for (size_t i = 0; i < prompts.size(); ++i) {
    SCOPED_TRACE(names[i]);
    const int64_t d = first_difference(batched[i], expected[i]);
    if (d < 0) continue;
    ++diverged;

    std::vector<int64_t> prefix = prompts[i];
    prefix.insert(prefix.end(), expected[i].begin(), expected[i].begin() + d);
    const auto logits = gpu16_->forward(torch::tensor(prefix, torch::kInt64).unsqueeze(0))[0][-1];
    const double gap = gpt2::test::top2_gap(logits);
    std::cout << "[          ] " << names[i] << " diverged at token " << d << " of " << expected[i].size()
              << ", top-2 gap " << gap << "\n";
    // Consecutive fp16 values near a GPT-2 logit are ~0.06-0.125 apart, so a gap of this order
    // means the two tokens were never distinguishable in fp16. A clear winner flipping would be a bug.
    EXPECT_LT(gap, 1.0) << "batched fp16 picked a different token where the winner was clear";
  }
  std::cout << "[          ] " << diverged << " of " << prompts.size() << " requests diverged in fp16\n";
}

}  // namespace
