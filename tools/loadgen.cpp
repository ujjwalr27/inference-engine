// Open-loop load generator: fires requests on a Poisson schedule regardless of whether earlier
// ones have finished (a closed-loop client would hide queueing delay - "coordinated omission").
//
// Usage: gpt2_loadgen [--url http://127.0.0.1:8080] [--rate 4] [--duration 30]
//                     [--prompt-min 16] [--prompt-max 128] [--max-tokens 32] [--max-tokens-min N]
//                     [--stream on|off] [--out results/run.csv] [--seed 1]
//                     [--api engine|openai] [--model gpt2]
//
// --api openai drives an OpenAI-compatible /v1/completions server (vLLM) with the same token-id
// prompts, so the same load can be compared across servers.
//
// --max-tokens-min N gives each request its own output length, drawn uniformly from
// [N, --max-tokens]. With every request the same length, a static batch finishes all at once and
// never idles a slot; mixed lengths show what static batching really costs.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
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
  int64_t max_tokens_min = 0;  // > 0: output lengths drawn from [max_tokens_min, max_tokens]
  bool stream = true;
  std::string out;
  uint32_t seed = 1;
  bool openai = false;         // --api openai: POST /v1/completions instead of /v1/generate
  std::string model = "gpt2";  // the "model" field an OpenAI-compatible server requires
};

// Random token IDs: the engine's cost depends on token counts, not on the text making sense.
std::vector<int64_t> random_prompt(std::mt19937& rng, int64_t min_len, int64_t max_len) {
  std::uniform_int_distribution<int64_t> length(min_len, max_len);
  std::uniform_int_distribution<int64_t> token(100, 40000);
  std::vector<int64_t> prompt(static_cast<size_t>(length(rng)));
  for (auto& id : prompt) id = token(rng);
  return prompt;
}

// Every request generates exactly max_tokens on every server (EOS is ignored), so every server
// does the same work for the same seed.
json request_body(const Options& options, const std::vector<int64_t>& prompt, int64_t max_tokens) {
  if (!options.openai) {
    return json{{"token_ids", prompt}, {"max_tokens", max_tokens}, {"stream", options.stream},
                {"stop_on_eos", false}};
  }
  json body{{"model", options.model}, {"prompt", prompt}, {"max_tokens", max_tokens},
            {"temperature", 0}, {"ignore_eos", true}, {"stream", options.stream}};
  if (options.stream) body["stream_options"] = {{"include_usage", true}};  // exact token count at the end
  return body;
}

// OpenAI-style stream: "data: {json}\n\n" events, a final usage event, then "data: [DONE]".
// Events can be split across network reads, so complete ones are cut out of a buffer.
void send_openai_stream(httplib::Client& client, const json& body, Record& record, Clock::time_point sent) {
  std::string buffer;
  int64_t chunks = 0, usage_tokens = -1;
  auto result = client.Post(
      "/v1/completions", httplib::Headers{}, body.dump(), "application/json", [&](const char* data, size_t len) {
        buffer.append(data, len);
        size_t end;
        while ((end = buffer.find("\n\n")) != std::string::npos) {
          const std::string event = buffer.substr(0, end);
          buffer.erase(0, end + 2);
          if (event.rfind("data: ", 0) != 0 || event == "data: [DONE]") continue;
          try {
            const auto j = json::parse(event.substr(6));
            if (j.contains("choices") && !j["choices"].empty()) {
              if (record.ttft_ms < 0) record.ttft_ms = ms_between(sent, Clock::now());
              ++chunks;
            }
            if (j.contains("usage") && j["usage"].is_object()) {
              usage_tokens = j["usage"].value("completion_tokens", int64_t{-1});
            }
          } catch (const std::exception&) {
          }
        }
        return true;
      });
  record.done_ms = ms_between(sent, Clock::now());
  record.output_tokens = usage_tokens >= 0 ? usage_tokens : chunks;
  record.status = result ? result->status : 0;
}

// Tokens in an engine stream. Counting "data:" events undercounts: the server holds back a token
// that ends partway through a UTF-8 character and sends it with the next one, so one event can
// carry two tokens (~1% of requests on random prompts). The final event's timings has the exact
// count; events are only counted when a stream ends without it.
int64_t streamed_token_count(const std::string& stream) {
  int64_t events = 0;
  size_t start = 0, end;
  while ((end = stream.find("\n\n", start)) != std::string::npos) {
    const std::string event = stream.substr(start, end - start);
    start = end + 2;
    if (event.rfind("data: {", 0) != 0) continue;
    try {
      const auto j = json::parse(event.substr(6));
      if (j.contains("timings")) return j.at("timings").at("output_tokens").get<int64_t>();
      if (j.contains("token_id")) ++events;
    } catch (const std::exception&) {
    }
  }
  return events;
}

void send_one(const Options& options, Record& record, std::vector<int64_t> prompt, int64_t max_tokens,
              Clock::time_point run_start) {
  httplib::Client client(options.url);
  client.set_read_timeout(300, 0);
  client.set_write_timeout(30, 0);

  const json body = request_body(options, prompt, max_tokens);
  record.prompt_tokens = static_cast<int64_t>(prompt.size());
  const auto sent = Clock::now();
  record.send_ms = ms_between(run_start, sent);

  if (options.openai) {
    if (options.stream) {
      send_openai_stream(client, body, record, sent);
      return;
    }
    auto result = client.Post("/v1/completions", body.dump(), "application/json");
    record.done_ms = ms_between(sent, Clock::now());
    record.ttft_ms = record.done_ms;
    record.status = result ? result->status : 0;
    if (result && result->status == 200) {
      try {
        record.output_tokens = json::parse(result->body).at("usage").at("completion_tokens").get<int64_t>();
      } catch (const std::exception&) {
      }
    }
    return;
  }

  if (options.stream) {
    bool first_seen = false;
    std::string stream;  // a few KB per request; parsed once the stream ends
    auto result = client.Post(
        "/v1/generate", httplib::Headers{}, body.dump(), "application/json",
        [&](const char* data, size_t len) {
          if (!first_seen && std::string_view(data, len).find("\"text\"") != std::string_view::npos) {
            record.ttft_ms = ms_between(sent, Clock::now());
            first_seen = true;
          }
          stream.append(data, len);
          return true;
        });
    record.done_ms = ms_between(sent, Clock::now());
    record.output_tokens = streamed_token_count(stream);
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
      else if (arg == "--max-tokens-min") options.max_tokens_min = std::stoll(next());
      else if (arg == "--stream") options.stream = (next() == "on");
      else if (arg == "--out") options.out = next();
      else if (arg == "--seed") options.seed = static_cast<uint32_t>(std::stoul(next()));
      else if (arg == "--model") options.model = next();
      else if (arg == "--api") {
        const std::string api = next();
        if (api != "engine" && api != "openai") throw std::invalid_argument("--api must be engine or openai");
        options.openai = api == "openai";
      }
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
  if (options.max_tokens_min < 0 || options.max_tokens_min > options.max_tokens) {
    std::cerr << "error: --max-tokens-min must be between 0 and --max-tokens\n";
    return 2;
  }
  std::uniform_int_distribution<int64_t> output_length(std::max<int64_t>(options.max_tokens_min, 1),
                                                       options.max_tokens);

  std::mt19937 rng(options.seed);
  std::exponential_distribution<double> gap(options.rate);  // Poisson arrivals

  std::vector<Record> records;
  std::vector<std::thread> in_flight;
  std::mutex records_mutex;
  const auto run_start = Clock::now();
  const auto deadline = run_start + std::chrono::duration_cast<Clock::duration>(
                                        std::chrono::duration<double>(options.duration));

  std::cout << "open-loop load: " << options.rate << " req/s for " << options.duration << " s -> " << options.url
            << " (stream " << (options.stream ? "on" : "off") << (options.openai ? ", OpenAI API" : "") << ", "
            << (options.max_tokens_min > 0 ? std::to_string(options.max_tokens_min) + "-" : std::string())
            << options.max_tokens << " output tokens)\n";

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
    // Drawn only when asked for, so a run without --max-tokens-min replays the same prompts as before.
    const int64_t max_tokens = options.max_tokens_min > 0 ? output_length(rng) : options.max_tokens;
    // Fire on schedule whether or not earlier requests have come back.
    in_flight.emplace_back([&options, raw, prompt = std::move(prompt), max_tokens, run_start]() mutable {
      send_one(options, *raw, std::move(prompt), max_tokens, run_start);
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
