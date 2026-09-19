#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gpt2 {

// GPT-2 byte-level BPE, backed by the Hugging Face tokenizers library (mlc-ai/tokenizers-cpp).
// Loads the same tokenizer.json that Python uses, so token IDs match exactly.
//
// Thread safety is not assumed: give each thread its own instance (they share nothing).
class Tokenizer {
 public:
  static Tokenizer from_file(const std::string& tokenizer_json_path);
  static Tokenizer from_blob(const std::string& tokenizer_json);

  std::vector<int64_t> encode(const std::string& text) const;
  std::string decode(const std::vector<int64_t>& ids) const;
  size_t vocab_size() const;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

// Streaming helper. GPT-2 works on bytes, so one character can span several tokens:
// decoding tokens one at a time would emit U+FFFD replacement characters. This keeps the
// tokens seen so far, decodes them together, and returns only the newly completed text.
class IncrementalDecoder {
 public:
  explicit IncrementalDecoder(const Tokenizer& tokenizer) : tokenizer_(&tokenizer) {}

  // Returns the text completed by this token; empty while a character is still incomplete.
  std::string push(int64_t token);

  // Returns whatever is still held back (only non-empty for a truncated character).
  std::string flush();

  const std::vector<int64_t>& tokens() const { return ids_; }

 private:
  const Tokenizer* tokenizer_;
  std::vector<int64_t> ids_;
  size_t emitted_bytes_ = 0;
};

}  // namespace gpt2
