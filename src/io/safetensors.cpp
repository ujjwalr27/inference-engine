#include "io/safetensors.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace gpt2 {
namespace {

constexpr uint64_t kMaxHeaderBytes = 100ULL * 1024 * 1024;

[[noreturn]] void fail(const std::string& path, const std::string& what) {
  throw std::runtime_error("safetensors: " + path + ": " + what);
}

bool host_is_little_endian() {
  const uint16_t probe = 1;
  unsigned char first = 0;
  std::memcpy(&first, &probe, 1);
  return first == 1;
}

torch::Dtype parse_dtype(const std::string& s, const std::string& path) {
  if (s == "F64") return torch::kFloat64;
  if (s == "F32") return torch::kFloat32;
  if (s == "F16") return torch::kFloat16;
  if (s == "BF16") return torch::kBFloat16;
  if (s == "I64") return torch::kInt64;
  if (s == "I32") return torch::kInt32;
  if (s == "I16") return torch::kInt16;
  if (s == "I8") return torch::kInt8;
  if (s == "U8") return torch::kUInt8;
  if (s == "BOOL") return torch::kBool;
  fail(path, "unsupported dtype " + s);
}

}  // namespace

SafeTensors SafeTensors::open(const std::string& path) {
  if (!host_is_little_endian()) {
    fail(path, "big-endian hosts are not supported");
  }
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) fail(path, "cannot open file");
  const auto file_size = static_cast<uint64_t>(in.tellg());
  in.seekg(0);

  if (file_size < 8) fail(path, "file too small");
  std::array<unsigned char, 8> len_bytes{};
  in.read(reinterpret_cast<char*>(len_bytes.data()), 8);
  uint64_t header_len = 0;
  for (int i = 7; i >= 0; --i) header_len = (header_len << 8) | len_bytes[static_cast<size_t>(i)];
  if (header_len == 0 || header_len > kMaxHeaderBytes || 8 + header_len > file_size) {
    fail(path, "invalid header length " + std::to_string(header_len));
  }

  std::string header(header_len, '\0');
  in.read(header.data(), static_cast<std::streamsize>(header_len));
  if (!in) fail(path, "truncated header");

  nlohmann::json j;
  try {
    j = nlohmann::json::parse(header);
  } catch (const nlohmann::json::exception& e) {
    fail(path, std::string("bad header JSON: ") + e.what());
  }
  if (!j.is_object()) fail(path, "header is not a JSON object");

  SafeTensors st;
  st.path_ = path;
  st.data_start_ = 8 + header_len;
  const uint64_t data_size = file_size - st.data_start_;

  for (auto it = j.begin(); it != j.end(); ++it) {
    if (it.key() == "__metadata__") {
      for (auto m = it.value().begin(); m != it.value().end(); ++m) {
        if (m.value().is_string()) st.metadata_[m.key()] = m.value().get<std::string>();
      }
      continue;
    }
    const auto& v = it.value();
    try {
      Entry e{parse_dtype(v.at("dtype").get<std::string>(), path), v.at("shape").get<std::vector<int64_t>>(),
              v.at("data_offsets").at(0).get<uint64_t>(), v.at("data_offsets").at(1).get<uint64_t>()};
      uint64_t numel = 1;
      for (int64_t d : e.shape) {
        if (d < 0) fail(path, it.key() + ": negative dimension");
        numel *= static_cast<uint64_t>(d);
      }
      const uint64_t expected = numel * c10::elementSize(e.dtype);
      if (e.end < e.begin || e.end > data_size) fail(path, it.key() + ": data offsets out of range");
      if (e.end - e.begin != expected) fail(path, it.key() + ": byte size does not match shape and dtype");
      st.entries_.emplace(it.key(), std::move(e));
    } catch (const nlohmann::json::exception& ex) {
      fail(path, it.key() + ": malformed entry: " + ex.what());
    }
  }
  return st;
}

bool SafeTensors::contains(const std::string& name) const { return entries_.count(name) > 0; }

std::vector<std::string> SafeTensors::names() const {
  std::vector<std::string> out;
  out.reserve(entries_.size());
  for (const auto& kv : entries_) out.push_back(kv.first);
  std::sort(out.begin(), out.end());
  return out;
}

const SafeTensors::Entry& SafeTensors::entry(const std::string& name) const {
  auto it = entries_.find(name);
  if (it == entries_.end()) fail(path_, "no tensor named '" + name + "'");
  return it->second;
}

const std::vector<int64_t>& SafeTensors::shape(const std::string& name) const { return entry(name).shape; }

torch::Dtype SafeTensors::dtype(const std::string& name) const { return entry(name).dtype; }

torch::Tensor SafeTensors::load(const std::string& name) const {
  const Entry& e = entry(name);
  auto tensor = torch::empty(e.shape, torch::TensorOptions().dtype(e.dtype));
  const uint64_t nbytes = e.end - e.begin;
  if (nbytes == 0) return tensor;

  std::ifstream in(path_, std::ios::binary);
  if (!in) fail(path_, "cannot reopen file");
  in.seekg(static_cast<std::streamoff>(data_start_ + e.begin));
  in.read(static_cast<char*>(tensor.data_ptr()), static_cast<std::streamsize>(nbytes));
  if (!in) fail(path_, name + ": truncated tensor data");
  return tensor;
}

}  // namespace gpt2
