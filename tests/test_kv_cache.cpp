#include <gtest/gtest.h>
#include <torch/torch.h>

#include "cache/kv_cache.h"
#include "model/config.h"
#include "test_util.h"

namespace {

gpt2::GPT2Config small_config() {
  gpt2::GPT2Config c;
  c.n_layer = 2;
  c.n_head = 2;
  c.n_embd = 8;  // head_dim = 4
  c.n_ctx = 6;
  return c;
}

class Cache : public ::testing::Test {
 protected:
  void SetUp() override { torch::manual_seed(99); }
  torch::InferenceMode guard_;
  gpt2::GPT2Config cfg_ = small_config();
};

TEST_F(Cache, ShapesAndSize) {
  gpt2::KVCache cache(cfg_, /*n_slots=*/3, torch::kCPU, torch::kFloat32);
  EXPECT_EQ(cache.n_slots(), 3);
  EXPECT_EQ(cache.n_ctx(), 6);
  EXPECT_EQ(cache.keys(0, 3, 6).sizes(), (std::vector<int64_t>{3, 2, 6, 4}));
  // 2 layers * 3 slots * 2 heads * 6 positions * 4 dims * 4 bytes * 2 tensors
  EXPECT_EQ(cache.bytes(), 2 * 3 * 2 * 6 * 4 * 4 * 2);
}

TEST_F(Cache, PrefillThenReadBack) {
  gpt2::KVCache cache(cfg_, 2, torch::kCPU, torch::kFloat32);
  const auto k = torch::randn({1, 2, 3, 4});
  const auto v = torch::randn({1, 2, 3, 4});
  cache.write_prefill(/*layer=*/1, /*slot=*/0, /*start_pos=*/0, k, v);

  EXPECT_TRUE(torch::equal(cache.keys(1, 1, 3)[0], k[0]));
  EXPECT_TRUE(torch::equal(cache.values(1, 1, 3)[0], v[0]));
  EXPECT_TRUE(cache.keys(0, 1, 3).eq(0).all().item<bool>()) << "other layers must stay untouched";
  EXPECT_TRUE(cache.keys(1, 2, 3)[1].eq(0).all().item<bool>()) << "other slots must stay untouched";
}

TEST_F(Cache, DecodeWritesPerRowPositions) {
  gpt2::KVCache cache(cfg_, 3, torch::kCPU, torch::kFloat32);
  const auto k = torch::randn({2, 2, 1, 4});
  const auto v = torch::randn({2, 2, 1, 4});
  const auto positions = torch::tensor({4, 1}, torch::kInt64);  // row 0 -> position 4, row 1 -> position 1
  cache.write_decode(/*layer=*/0, /*batch=*/2, positions, k, v);

  const auto keys = cache.keys(0, 2, 6);
  EXPECT_TRUE(torch::equal(keys.index({0, torch::indexing::Slice(), 4}), k.squeeze(2)[0]));
  EXPECT_TRUE(torch::equal(keys.index({1, torch::indexing::Slice(), 1}), k.squeeze(2)[1]));
  EXPECT_TRUE(torch::equal(cache.values(0, 2, 6).index({1, torch::indexing::Slice(), 1}), v.squeeze(2)[1]));
  EXPECT_TRUE(keys.index({0, torch::indexing::Slice(), 3}).eq(0).all().item<bool>());
}

TEST_F(Cache, ViewsAreNotCopies) {
  gpt2::KVCache cache(cfg_, 1, torch::kCPU, torch::kFloat32);
  auto view = cache.keys(0, 1, 6);
  cache.write_prefill(0, 0, 0, torch::ones({1, 2, 6, 4}), torch::ones({1, 2, 6, 4}));
  EXPECT_TRUE(view.eq(1).all().item<bool>()) << "keys() must return a view of the preallocated buffer";
}

TEST_F(Cache, RejectsOutOfRange) {
  gpt2::KVCache cache(cfg_, 2, torch::kCPU, torch::kFloat32);
  const auto k = torch::randn({1, 2, 3, 4});
  EXPECT_THROW(cache.write_prefill(5, 0, 0, k, k), c10::Error);           // no such layer
  EXPECT_THROW(cache.write_prefill(0, 7, 0, k, k), c10::Error);           // no such slot
  EXPECT_THROW(cache.write_prefill(0, 0, 5, k, k), c10::Error);           // runs past n_ctx
  EXPECT_THROW(cache.keys(0, 2, 99), c10::Error);                         // length past n_ctx
  EXPECT_THROW(cache.write_decode(0, 2, torch::tensor({0}, torch::kInt64), k, k), c10::Error);  // batch mismatch
}

TEST_F(Cache, CopySlotMovesOnlyTheCachedPrefix) {
  gpt2::KVCache cache(cfg_, 3, torch::kCPU, torch::kFloat32);
  const auto k = torch::randn({1, 2, 6, 4});
  cache.write_prefill(0, /*slot=*/2, 0, k, k);          // a request living in slot 2
  cache.write_prefill(0, /*slot=*/0, 0, torch::ones({1, 2, 6, 4}), torch::ones({1, 2, 6, 4}));

  cache.copy_slot(/*from=*/2, /*to=*/0, /*length=*/4);  // compaction: 4 cached positions

  const auto slot0 = cache.keys(0, 1, 6)[0];
  EXPECT_TRUE(torch::equal(slot0.slice(1, 0, 4), k[0].slice(1, 0, 4)));
  EXPECT_TRUE(slot0.slice(1, 4, 6).eq(1).all().item<bool>()) << "positions past the cached length stay untouched";
  EXPECT_THROW(cache.copy_slot(0, 9, 1), c10::Error);
}

TEST_F(Cache, ClearZeroes) {
  gpt2::KVCache cache(cfg_, 1, torch::kCPU, torch::kFloat32);
  cache.write_prefill(0, 0, 0, torch::ones({1, 2, 6, 4}), torch::ones({1, 2, 6, 4}));
  cache.clear();
  EXPECT_TRUE(cache.keys(0, 1, 6).eq(0).all().item<bool>());
}

TEST(CacheSizing, MatchesTheDocumentedFormula) {
  gpt2::GPT2Config gpt2_small;  // 12 layers, 12 heads, 1024 ctx, head_dim 64
  gpt2::KVCache cache(gpt2_small, 1, torch::kCPU, torch::kFloat16);
  EXPECT_EQ(cache.bytes(), 12L * 1 * 12 * 1024 * 64 * 2 * 2);  // ~36 MiB per slot in fp16
  EXPECT_NEAR(static_cast<double>(cache.bytes()) / (1024 * 1024), 36.0, 0.1);
}

}  // namespace
