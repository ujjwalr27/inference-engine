#include "tokenizer/tokenizer.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <tokenizers_cpp.h>

namespace gpt2 {
namespace {

// UTF-8 encoding of U+FFFD REPLACEMENT CHARACTER, produced when a character is cut mid-sequence.
constexpr char kReplacement[] = "\xEF\xBF\xBD";
constexpr size_t kReplacementLen = 3;

bool ends_with_replacement(const std::string& s) {
  return s.size() >= kReplacementLen && s.compare(s.size() - kReplacementLen, kReplacementLen, kReplacement) == 0;
}

}  // namespace

struct Tokenizer::Impl {
  std::unique_ptr<tokenizers::Tokenizer> tok;
};

Tokenizer Tokenizer::from_blob(const std::string& tokenizer_json) {
  Tokenizer t;
  t.impl_ = std::make_shared<Impl>();
  t.impl_->tok = tokenizers::Tokenizer::FromBlobJSON(tokenizer_json);
  if (!t.impl_->tok) throw std::runtime_error("failed to parse tokenizer.json");
  return t;
}

Tokenizer Tokenizer::from_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open tokenizer file: " + path);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return from_blob(buffer.str());
}

std::vector<int64_t> Tokenizer::encode(const std::string& text) const {
  const auto ids = impl_->tok->Encode(text);
  return std::vector<int64_t>(ids.begin(), ids.end());
}

std::string Tokenizer::decode(const std::vector<int64_t>& ids) const {
  std::vector<int32_t> narrow;
  narrow.reserve(ids.size());
  for (int64_t id : ids) narrow.push_back(static_cast<int32_t>(id));
  return impl_->tok->Decode(narrow);
}

size_t Tokenizer::vocab_size() const { return impl_->tok->GetVocabSize(); }

std::string IncrementalDecoder::push(int64_t token) {
  ids_.push_back(token);
  // Byte-level BPE decoding is a concatenation of per-token bytes, so the text produced so far
  // is a stable prefix: decoding everything again and taking the tail is safe (and cheap at n_ctx = 1024).
  const std::string text = tokenizer_->decode(ids_);
  if (text.size() < emitted_bytes_) return {};  // defensive: never go backwards
  if (ends_with_replacement(text)) return {};   // character still incomplete, wait for more tokens
  std::string fresh = text.substr(emitted_bytes_);
  emitted_bytes_ = text.size();
  return fresh;
}

std::string IncrementalDecoder::flush() {
  const std::string text = tokenizer_->decode(ids_);
  if (text.size() <= emitted_bytes_) return {};
  std::string rest = text.substr(emitted_bytes_);
  emitted_bytes_ = text.size();
  return rest;
}

}  // namespace gpt2
