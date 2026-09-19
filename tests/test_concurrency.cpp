// Queue and channel behaviour under threads. No model here on purpose: this file is what the
// ThreadSanitizer job runs (GPT2_DATA_DIR pointed at nothing, so model tests skip).
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "scheduler/queue.h"
#include "scheduler/request.h"

namespace {

std::shared_ptr<gpt2::Request> make_request(uint64_t id) {
  auto r = std::make_shared<gpt2::Request>();
  r->id = id;
  r->prompt = {1, 2, 3};
  return r;
}

TEST(TokenChannel, DeliversInOrderAcrossThreads) {
  gpt2::TokenChannel channel;
  std::thread producer([&] {
    for (int64_t i = 0; i < 500; ++i) channel.push(i);
    channel.finish();
  });

  const auto tokens = channel.collect();
  producer.join();

  ASSERT_EQ(tokens.size(), 500u);
  for (int64_t i = 0; i < 500; ++i) EXPECT_EQ(tokens[static_cast<size_t>(i)], i);
  EXPECT_TRUE(channel.closed());
  EXPECT_TRUE(channel.error().empty());
}

TEST(TokenChannel, CloseWakesAWaitingConsumer) {
  gpt2::TokenChannel channel;
  std::atomic<bool> returned{false};
  std::thread consumer([&] {
    int64_t token = 0;
    EXPECT_FALSE(channel.next(token));  // must not block forever
    returned.store(true);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_FALSE(returned.load());
  channel.finish();
  consumer.join();
  EXPECT_TRUE(returned.load());
}

TEST(TokenChannel, ReportsErrorAndIgnoresLateTokens) {
  gpt2::TokenChannel channel;
  channel.push(7);
  channel.finish("boom");
  channel.push(8);  // ignored: the stream is closed
  EXPECT_EQ(channel.collect(), (std::vector<int64_t>{7}));
  EXPECT_EQ(channel.error(), "boom");
}

TEST(RequestQueue, EnforcesCapacity) {
  gpt2::RequestQueue queue(3);
  EXPECT_TRUE(queue.try_push(make_request(1)));
  EXPECT_TRUE(queue.try_push(make_request(2)));
  EXPECT_TRUE(queue.try_push(make_request(3)));
  EXPECT_FALSE(queue.try_push(make_request(4))) << "a full queue must reject, not grow";
  EXPECT_EQ(queue.size(), 3u);

  ASSERT_NE(queue.try_pop(), nullptr);
  EXPECT_TRUE(queue.try_push(make_request(5)));
  EXPECT_EQ(queue.size(), 3u);
}

TEST(RequestQueue, ManyProducersOneConsumer) {
  constexpr int kProducers = 4;
  constexpr int kPerProducer = 250;
  gpt2::RequestQueue queue(64);
  std::atomic<int> accepted{0};
  std::atomic<int> consumed{0};

  std::vector<std::thread> producers;
  for (int p = 0; p < kProducers; ++p) {
    producers.emplace_back([&, p] {
      for (int i = 0; i < kPerProducer; ++i) {
        while (!queue.try_push(make_request(static_cast<uint64_t>(p * kPerProducer + i)))) {
          std::this_thread::yield();  // queue full: back off like the server does
        }
        accepted.fetch_add(1);
      }
    });
  }

  std::thread consumer([&] {
    while (consumed.load() < kProducers * kPerProducer) {
      if (queue.try_pop()) {
        consumed.fetch_add(1);
      } else {
        queue.wait_for_work(std::chrono::milliseconds(5));
      }
    }
  });

  for (auto& t : producers) t.join();
  consumer.join();
  EXPECT_EQ(accepted.load(), kProducers * kPerProducer);
  EXPECT_EQ(consumed.load(), kProducers * kPerProducer);
  EXPECT_EQ(queue.size(), 0u);
}

TEST(RequestQueue, CloseRejectsAndWakes) {
  gpt2::RequestQueue queue(2);
  std::thread waiter([&] { queue.wait_for_work(std::chrono::seconds(5)); });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  queue.close();
  waiter.join();

  EXPECT_TRUE(queue.closed());
  EXPECT_FALSE(queue.try_push(make_request(1)));
}

TEST(Request, CancellationIsVisibleToAnotherThread) {
  auto request = make_request(1);
  std::thread canceller([&] { request->cancel(); });
  canceller.join();
  EXPECT_TRUE(request->is_cancelled());
  EXPECT_STREQ(gpt2::to_string(gpt2::FinishReason::Cancelled), "cancelled");
}

}  // namespace
