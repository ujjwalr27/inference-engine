#pragma once

#include <cstdint>
#include <vector>

#include <torch/torch.h>

namespace gpt2 {

// A batch of prompts padded to a common length. Padding goes on the LEFT so that the last
// column is a real token for every row, which keeps "logits of the last position" valid.
//
//   ids          [B, T] int64   pad_token_id in the padded cells
//   positions    [B, T] int64   cumsum(mask) - 1, so each row's real tokens start at 0
//   padding_mask [B, T] bool    true = real token
struct PaddedBatch {
  torch::Tensor ids;
  torch::Tensor positions;
  torch::Tensor padding_mask;
  std::vector<int64_t> lengths;  // real (unpadded) length per row
  int64_t max_length = 0;

  int64_t batch_size() const { return static_cast<int64_t>(lengths.size()); }
  // Fraction of the [B, T] block that is padding: the cost static batching pays.
  double padding_waste() const;
};

PaddedBatch pad_left(const std::vector<std::vector<int64_t>>& prompts, int64_t pad_token_id = 0);

}  // namespace gpt2
