#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

namespace gpt2 {

// Minimal reader for the safetensors format:
//   [u64 little-endian header length][JSON header][raw tensor bytes]
// The header is parsed and validated on open; tensor bytes are read on demand
// straight into a freshly allocated CPU tensor (no intermediate buffer).
class SafeTensors {
 public:
  static SafeTensors open(const std::string& path);

  bool contains(const std::string& name) const;
  std::vector<std::string> names() const;  // sorted
  const std::vector<int64_t>& shape(const std::string& name) const;
  torch::Dtype dtype(const std::string& name) const;
  const std::map<std::string, std::string>& metadata() const { return metadata_; }

  // Reads one tensor into a new contiguous CPU tensor. Throws if the name is unknown.
  torch::Tensor load(const std::string& name) const;

 private:
  struct Entry {
    torch::Dtype dtype;
    std::vector<int64_t> shape;
    uint64_t begin;  // offsets relative to the start of the data section
    uint64_t end;
  };

  const Entry& entry(const std::string& name) const;

  std::string path_;
  uint64_t data_start_ = 0;
  std::unordered_map<std::string, Entry> entries_;
  std::map<std::string, std::string> metadata_;
};

}  // namespace gpt2
