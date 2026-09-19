// Open-loop load generator: fires requests on a Poisson schedule regardless of whether earlier
// ones have finished (a closed-loop client would hide queueing delay - "coordinated omission").
//
// Usage: gpt2_loadgen [--url http://127.0.0.1:8080] [--rate 4] [--duration 30]
//                     [--prompt-min 16] [--prompt-max 128] [--max-tokens 32]
//                     [--stream on|off] [--out results/run.csv] [--seed 1]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

namespace {

using Clock = std::chrono::steady_clock;
using json = nlohmann::json;

struct Record {
  uint64_t id = 0;
  int64_t prompt_tokens = 0;
  int64_t output_tokens = 0;
  double send_ms = 0;    // relative to run start
  double ttft_ms = -1;   // -1 when no token ever arrived
  double done_ms = -1;
  int status = 0;        // HTTP status, or 0 for a transport error
};

double ms_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

double percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double rank = p / 100.0 * static_cast<double>(values.size() - 1);
  const size_t low = static_cast<size_t>(std::floor(rank));
  const size_t high = static_cast<size_t>(std::ceil(rank));
  return values[low] + (values[high] - values[low]) * (rank - static_cast<double>(low));
}

struct Options {
  std::string url = "http://127.0.0.1:8080";
  double rate = 4.0;
  double duration = 30.0;
  int64_t prompt_min = 16, prompt_max = 128, max_tokens = 32;
  bool stream = true;
  std::string out;
  uint32_t seed = 1;
};

// Random token IDs: the engine's cost depends on token counts, not on the text making sense.
std::vector<int64_t> random_prompt(std::mt19937& rng, int64_t min_len, int64_t max_len) {
  std::uniform_int_distribution<int64_t> length(min_len, max_len);
  std::uniform_int_distribution<int64_t> token(100, 40000);
  std::vector<int64_t> prompt(static_cast<size_t>(length(rng)));
  for (auto& id : prompt) id = token(rng);
  return prompt;
}

void send_one(const Options& options, Record& record, std::vector<int64_t> prompt, Clock::time_point run_start) {
  httplib::Client client(options.url);
  client.set_read_timeout(300, 0);
  client.set_write_timeout(30, 0);

  const json body{{"token_ids", prompt}, {"max_tokens", options.max_tokens}, {"stream", options.stream}};
  record.prompt_tokens = static_cast<int64_t>(prompt.size());
  const auto sent = Clock::now();
  record.send_ms = ms_between(run_start, sent);

  if (options.stream) {
    bool first_seen = false;
    int64_t tokens = 0;
    auto result = client.Post(
        "/v1/generate", httplib::Headers{}, body.dump(), "application/json",
        [&](const char* data, size_t len) {
          const std::string chunk(data, len);
          if (!first_seen && chunk.find("\"text\"") != std::string::npos) {
            record.ttft_ms = ms_between(sent, Clock::now());
            first_seen = true;
          }
          // Each token arrives as its own "data: {...}" event; [DONE] closes the stream.
          size_t pos = 0;
          while ((pos = chunk.find("\"token_id\"", pos)) != std::string::npos) {
            ++tokens;
            ++pos;
          }
          return true;
        });
    record.done_ms = ms_between(sent, Clock::now());
    record.output_tokens = tokens;
    record.status = result ? result->status : 0;
  } else {
    auto result = client.Post("/v1/generate", body.dump(), "application/json");
    record.done_ms = ms_between(sent, Clock::now());
    record.ttft_ms = record.done_ms;  // non-streaming: first byte is the whole answer
    record.status = result ? result->status : 0;
    if (result && result->status == 200) {
      try {
        record.output_tokens = static_cast<int64_t>(json::parse(result->body).at("token_ids").size());
      } catch (const std::exception&) {
      }
    }
  }
}

void write_csv(const std::string& path, const std::vector<Record>& records) {
  std::ofstream out(path);
  if (!out) {
    std::cerr << "warning: cannot write " << path << "\n";
    return;
  }
  out << "id,prompt_tokens,output_tokens,send_ms,ttft_ms,done_ms,status\n";
  for (const auto& r : records) {
    out << r.id << ',' << r.prompt_tokens << ',' << r.output_tokens << ',' << r.send_ms << ',' << r.ttft_ms << ','
        << r.done_ms << ',' << r.status << '\n';
  }
  std::cout << "wrote " << path << "\n";
}

void summarize(const std::vector<Record>& records, double wall_seconds) {
  std::vector<double> ttft, tpot, total;
  int64_t tokens = 0;
  size_t ok = 0, busy = 0, failed = 0;
  for (const auto& r : records) {
    if (r.status == 200) {
      ++ok;
      tokens += r.output_tokens;
      if (r.ttft_ms >= 0) ttft.push_back(r.ttft_ms);
      total.push_back(r.done_ms);
      if (r.output_tokens > 1 && r.ttft_ms >= 0) {
        tpot.push_back((r.done_ms - r.ttft_ms) / static_cast<double>(r.output_tokens - 1));
      }
    } else if (r.status == 429) {
      ++busy;
    } else {
      ++failed;
    }
  }

  std::cout << "\nrequests: " << records.size() << " sent | " << ok << " ok | " << busy << " busy(429) | " << failed
            << " failed\n"
            << "wall: " << wall_seconds << " s | completed " << static_cast<double>(ok) / wall_seconds << " req/s | "
            << static_cast<double>(tokens) / wall_seconds << " output tok/s\n"
            << "TTFT ms   p50 " << percentile(ttft, 50) << " | p90 " << percentile(ttft, 90) << " | p99 "
            << percentile(ttft, 99) << "\n"
            << "TPOT ms   p50 " << percentile(tpot, 50) << " | p99 " << percentile(tpot, 99) << "\n"
            << "total ms  p50 " << percentile(total, 50) << " | p99 " << percentile(total, 99) << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
      return argv[++i];
    };
    try {
      if (arg == "--url") options.url = next();
      else if (arg == "--rate") options.rate = std::stod(next());
      else if (arg == "--duration") options.duration = std::stod(next());
      else if (arg == "--prompt-min") options.prompt_min = std::stoll(next());
      else if (arg == "--prompt-max") options.prompt_max = std::stoll(next());
      else if (arg == "--max-tokens") options.max_tokens = std::stoll(next());
      else if (arg == "--stream") options.stream = (next() == "on");
      else if (arg == "--out") options.out = next();
      else if (arg == "--seed") options.seed = static_cast<uint32_t>(std::stoul(next()));
      else throw std::invalid_argument("unknown argument " + arg);
    } catch (const std::exception& e) {
      std::cerr << "error: " << e.what() << "\n";
      return 2;
    }
  }
  if (options.rate <= 0 || options.duration <= 0) {
    std::cerr << "error: --rate and --duration must be positive\n";
    return 2;
  }

  std::mt19937 rng(options.seed);
  std::exponential_distribution<double> gap(options.rate);  // Poisson arrivals

  std::vector<Record> records;
  std::vector<std::thread> in_flight;
  std::mutex records_mutex;
  const auto run_start = Clock::now();
  const auto deadline = run_start + std::chrono::duration_cast<Clock::duration>(
                                        std::chrono::duration<double>(options.duration));

  std::cout << "open-loop load: " << options.rate << " req/s for " << options.duration << " s -> " << options.url
            << " (stream " << (options.stream ? "on" : "off") << ")\n";

  uint64_t id = 0;
  auto next_arrival = run_start;
  std::vector<std::unique_ptr<Record>> owned;
  while (next_arrival < deadline) {
    std::this_thread::sleep_until(next_arrival);
    auto record = std::make_unique<Record>();
    record->id = id++;
    Record* raw = record.get();
    owned.push_back(std::move(record));
    auto prompt = random_prompt(rng, options.prompt_min, options.prompt_max);
    // Fire on schedule whether or not earlier requests have come back.
    in_flight.emplace_back([&options, raw, prompt = std::move(prompt), run_start]() mutable {
      send_one(options, *raw, std::move(prompt), run_start);
    });
    next_arrival += std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(gap(rng)));
  }

  for (auto& t : in_flight) t.join();
  const double wall = ms_between(run_start, Clock::now()) / 1000.0;

  records.reserve(owned.size());
  for (const auto& r : owned) records.push_back(*r);
  summarize(records, wall);
  if (!options.out.empty()) write_csv(options.out, records);
  return 0;
}
