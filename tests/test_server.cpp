// Phase 5: HTTP behaviour - streaming, overload, cancellation on disconnect.
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include "io/safetensors.h"
#include "model/generate.h"
#include "model/gpt2.h"
#include "scheduler/scheduler.h"
#include "server/http_server.h"
#include "test_util.h"

namespace {

using json = nlohmann::json;
using gpt2::test::to_vector;

// Concatenates the "text" fields of an SSE body, i.e. what a streaming client would display.
std::string sse_text(const std::string& body) {
  std::string text;
  size_t pos = 0;
  while ((pos = body.find("data: ", pos)) != std::string::npos) {
    const size_t start = pos + 6;
    const size_t end = body.find("\n\n", start);
    if (end == std::string::npos) break;
    const std::string payload = body.substr(start, end - start);
    pos = end;
    if (payload == "[DONE]") continue;
    const auto event = json::parse(payload, nullptr, false);
    if (!event.is_discarded() && event.contains("text")) text += event.at("text").get<std::string>();
  }
  return text;
}

class ServerTest : public ::testing::Test {
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
    start_server(default_scheduler_options());
  }
  void TearDown() override {
    if (server_) server_->stop();
    if (scheduler_) scheduler_->stop();
    server_.reset();
    scheduler_.reset();
  }

  static gpt2::SchedulerOptions default_scheduler_options() {
    gpt2::SchedulerOptions options;
    options.n_slots = 4;
    options.max_queue = 16;
    return options;
  }

  void start_server(const gpt2::SchedulerOptions& scheduler_options) {
    scheduler_ = std::make_unique<gpt2::Scheduler>(*model_, scheduler_options);
    scheduler_->start();
    server_ = std::make_unique<gpt2::InferenceServer>(*model_, *scheduler_, gpt2::test::data_dir() + "/tokenizer.json",
                                                      gpt2::ServerOptions{});
    port_ = server_->listen_on_any_port();
    ASSERT_GT(port_, 0) << "could not bind a port";
  }

  httplib::Client client() const {
    httplib::Client c("127.0.0.1", port_);
    c.set_read_timeout(120, 0);
    return c;
  }

  static std::unique_ptr<gpt2::GPT2Model> model_;
  static std::unique_ptr<gpt2::SafeTensors> ref_;
  std::unique_ptr<gpt2::Scheduler> scheduler_;
  std::unique_ptr<gpt2::InferenceServer> server_;
  int port_ = 0;
};

std::unique_ptr<gpt2::GPT2Model> ServerTest::model_;
std::unique_ptr<gpt2::SafeTensors> ServerTest::ref_;

TEST_F(ServerTest, HealthAndStats) {
  auto c = client();
  auto health = c.Get("/health");
  ASSERT_TRUE(health);
  EXPECT_EQ(health->status, 200);
  EXPECT_EQ(json::parse(health->body).at("status"), "ok");

  auto stats = c.Get("/stats");
  ASSERT_TRUE(stats);
  const auto body = json::parse(stats->body);
  EXPECT_EQ(body.at("slots"), 4);
  EXPECT_EQ(body.at("submitted"), 0);
}

TEST_F(ServerTest, GenerateMatchesSoloGeneration) {
  const auto prompt = to_vector(ref_->load("p0.input_ids"));
  gpt2::GenerationOptions options;
  options.max_new_tokens = 16;
  gpt2::KVCache cache(model_->config(), 1, model_->device(), model_->dtype());
  const auto expected = gpt2::generate_cached(*model_, prompt, options, &cache).tokens;

  auto response = client().Post("/v1/generate",
                                json{{"token_ids", prompt}, {"max_tokens", 16}}.dump(), "application/json");
  ASSERT_TRUE(response);
  ASSERT_EQ(response->status, 200);
  const auto body = json::parse(response->body);
  EXPECT_EQ(body.at("token_ids").get<std::vector<int64_t>>(), expected);
  EXPECT_EQ(body.at("finish_reason"), "max_tokens");
  EXPECT_GE(body.at("timings").at("ttft_ms").get<double>(), 0.0);
  EXPECT_EQ(body.at("timings").at("output_tokens"), 16);
}

TEST_F(ServerTest, StreamingProducesTheSameTextAsNonStreaming) {
  const json request{{"prompt", "The future of inference engines is"}, {"max_tokens", 12}};
  auto plain = client().Post("/v1/generate", request.dump(), "application/json");
  ASSERT_TRUE(plain);
  ASSERT_EQ(plain->status, 200);
  const auto expected_text = json::parse(plain->body).at("text").get<std::string>();

  json streaming = request;
  streaming["stream"] = true;
  auto streamed = client().Post("/v1/generate", streaming.dump(), "application/json");
  ASSERT_TRUE(streamed);
  ASSERT_EQ(streamed->status, 200);
  EXPECT_NE(streamed->body.find("data: [DONE]"), std::string::npos);
  EXPECT_EQ(sse_text(streamed->body), expected_text);
  EXPECT_NE(streamed->body.find("\"finish_reason\""), std::string::npos);
}

TEST_F(ServerTest, RejectsBadRequests) {
  auto c = client();
  EXPECT_EQ(c.Post("/v1/generate", "not json", "application/json")->status, 400);
  EXPECT_EQ(c.Post("/v1/generate", json{{"max_tokens", 4}}.dump(), "application/json")->status, 400);
  EXPECT_EQ(c.Post("/v1/generate", json{{"token_ids", std::vector<int64_t>{}}}.dump(), "application/json")->status,
            400);
  EXPECT_EQ(c.Post("/v1/generate", json{{"token_ids", std::vector<int64_t>{999999}}}.dump(), "application/json")
                ->status,
            400);
  // Prompt plus requested tokens cannot fit the 1024-token context.
  EXPECT_EQ(c.Post("/v1/generate", json{{"token_ids", std::vector<int64_t>{15496}}, {"max_tokens", 1024}}.dump(),
                   "application/json")
                ->status,
            400);
}

TEST_F(ServerTest, RepliesBusyWhenTheQueueIsFull) {
  TearDown();  // restart with a deliberately tiny queue
  gpt2::SchedulerOptions tight;
  tight.n_slots = 1;
  tight.max_queue = 1;
  start_server(tight);

  constexpr int kClients = 8;
  std::atomic<int> busy{0};
  std::atomic<int> ok{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < kClients; ++i) {
    threads.emplace_back([&] {
      auto response = client().Post(
          "/v1/generate", json{{"token_ids", std::vector<int64_t>{15496, 995}}, {"max_tokens", 24}}.dump(),
          "application/json");
      if (!response) return;
      if (response->status == 429) ++busy;
      if (response->status == 200) ++ok;
    });
  }
  for (auto& t : threads) t.join();

  EXPECT_GT(busy.load(), 0) << "an overloaded server must reject, not queue without limit";
  EXPECT_GT(ok.load(), 0) << "some requests should still succeed";
  EXPECT_EQ(busy.load() + ok.load(), kClients);
  EXPECT_GT(scheduler_->stats().rejected, 0u);
}

TEST_F(ServerTest, ClientDisconnectCancelsGeneration) {
  std::atomic<int> chunks{0};
  // The client must be destroyed for the socket to close - keep-alive holds it open otherwise,
  // and then the server has no way to know the client is gone.
  std::thread hangup([&] {
    auto c = client();
    c.Post("/v1/generate", httplib::Headers{},
           json{{"token_ids", to_vector(ref_->load("p0.input_ids"))}, {"max_tokens", 200}, {"stream", true}}.dump(),
           "application/json",
           [&](const char*, size_t) {
             ++chunks;
             return false;  // stop reading
           });
  });
  hangup.join();

  EXPECT_GT(chunks.load(), 0);
  for (int i = 0; i < 100 && scheduler_->stats().cancelled == 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  const auto stats = scheduler_->stats();
  EXPECT_EQ(stats.cancelled, 1u) << "a disconnected client must free its slot";
  EXPECT_EQ(stats.active, 0);
}

}  // namespace
