#include "model/config.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace gpt2 {

GPT2Config GPT2Config::from_json_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open config: " + path);
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(in);
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("bad config JSON in " + path + ": " + e.what());
  }

  GPT2Config c;
  try {
    c.n_layer = j.at("n_layer").get<int64_t>();
    c.n_head = j.at("n_head").get<int64_t>();
    c.n_embd = j.at("n_embd").get<int64_t>();
    c.n_ctx = j.at("n_ctx").get<int64_t>();
    c.vocab_size = j.at("vocab_size").get<int64_t>();
    c.layer_norm_epsilon = j.at("layer_norm_epsilon").get<double>();
    c.eos_token_id = j.at("eos_token_id").get<int64_t>();
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("missing/invalid field in " + path + ": " + e.what());
  }
  c.validate();
  return c;
}

void GPT2Config::validate() const {
  if (n_layer <= 0 || n_head <= 0 || n_embd <= 0 || n_ctx <= 0 || vocab_size <= 0) {
    throw std::invalid_argument("config dimensions must be positive");
  }
  if (n_embd % n_head != 0) {
    throw std::invalid_argument("n_embd must be divisible by n_head");
  }
  if (layer_norm_epsilon <= 0.0) {
    throw std::invalid_argument("layer_norm_epsilon must be positive");
  }
}

}  // namespace gpt2
