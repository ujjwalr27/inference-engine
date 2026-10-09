#include "tokenizer/tokenizer.h"

#include <fstream>
#include <mutex>
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
  // tokenizers-cpp's Decode writes into a buffer held by the handle and then reads it back, so two
  // threads decoding at once could read each other's text. Held only for one call (microseconds).
  std::mutex mutex;
  // Read once: the bundled tokenizers 0.21 answers get_vocab_size(true) by building the whole
  // 50k-entry vocabulary map, ~8 ms a call. A test calling it per token took 417 s on the T4.
  size_t vocab_size = 0;
};

Tokenizer Tokenizer::from_blob(const std::string& tokenizer_json) {
  Tokenizer t;
  t.impl_ = std::make_shared<Impl>();
  t.impl_->tok = tokenizers::Tokenizer::FromBlobJSON(tokenizer_json);
  if (!t.impl_->tok) throw std::runtime_error("failed to parse tokenizer.json");
  t.impl_->vocab_size = t.impl_->tok->GetVocabSize();
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
  std::vector<int32_t> ids;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    ids = impl_->tok->Encode(text);
  }
  return std::vector<int64_t>(ids.begin(), ids.end());
}

std::string Tokenizer::decode(const std::vector<int64_t>& ids) const {
  std::vector<int32_t> narrow;
  narrow.reserve(ids.size());
  for (int64_t id : ids) narrow.push_back(static_cast<int32_t>(id));
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->tok->Decode(narrow);
}

size_t Tokenizer::vocab_size() const { return impl_->vocab_size; }

// Byte-level BPE decodes each token to its own bytes and the text is those bytes concatenated,
// so decoding can restart at any character boundary. Everything up to pending_start_ has been
// emitted and ended on a whole character, which makes the suffix after it decode to exactly the
// text still owed. Decoding only that suffix keeps each token O(1) instead of re-decoding the
// whole output - which grew quadratically with output length, under the tokenizer's lock that
// every stream on the server shares.
std::string IncrementalDecoder::pending_text() const {
  const std::vector<int64_t> pending(ids_.begin() + static_cast<std::ptrdiff_t>(pending_start_), ids_.end());
  return pending.empty() ? std::string{} : tokenizer_->decode(pending);
}

std::string IncrementalDecoder::push(int64_t token) {
  ids_.push_back(token);
  std::string text = pending_text();
  if (ends_with_replacement(text)) return {};  // character still incomplete, wait for more tokens
  pending_start_ = ids_.size();
  return text;
}

std::string IncrementalDecoder::flush() {
  std::string text = pending_text();
  pending_start_ = ids_.size();
  return text;
}

}  // namespace gpt2
