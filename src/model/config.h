#pragma once

#include <cstdint>
#include <string>

namespace gpt2 {

struct GPT2Config {
  int64_t n_layer = 12;
  int64_t n_head = 12;
  int64_t n_embd = 768;
  int64_t n_ctx = 1024;
  int64_t vocab_size = 50257;
  double layer_norm_epsilon = 1e-5;
  int64_t eos_token_id = 50256;

  int64_t head_dim() const { return n_embd / n_head; }

  // Reads config.json written by scripts/export_weights.py and validates it.
  static GPT2Config from_json_file(const std::string& path);
  void validate() const;
};

}  // namespace gpt2
