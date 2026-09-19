#include "scheduler/padding.h"

#include <algorithm>
#include <stdexcept>

namespace gpt2 {

double PaddedBatch::padding_waste() const {
  const int64_t total = batch_size() * max_length;
  if (total == 0) return 0.0;
  int64_t real = 0;
  for (int64_t len : lengths) real += len;
  return 1.0 - static_cast<double>(real) / static_cast<double>(total);
}

PaddedBatch pad_left(const std::vector<std::vector<int64_t>>& prompts, int64_t pad_token_id) {
  if (prompts.empty()) throw std::invalid_argument("pad_left: no prompts");

  PaddedBatch batch;
  batch.lengths.reserve(prompts.size());
  for (const auto& p : prompts) {
    if (p.empty()) throw std::invalid_argument("pad_left: empty prompt");
    batch.lengths.push_back(static_cast<int64_t>(p.size()));
  }
  batch.max_length = *std::max_element(batch.lengths.begin(), batch.lengths.end());

  const int64_t B = batch.batch_size();
  const int64_t T = batch.max_length;
  batch.ids = torch::full({B, T}, pad_token_id, torch::kInt64);
  batch.padding_mask = torch::zeros({B, T}, torch::kBool);

  auto ids_acc = batch.ids.accessor<int64_t, 2>();
  auto mask_acc = batch.padding_mask.accessor<bool, 2>();
  for (int64_t b = 0; b < B; ++b) {
    const auto& prompt = prompts[static_cast<size_t>(b)];
    const int64_t offset = T - batch.lengths[static_cast<size_t>(b)];  // left padding
    for (size_t i = 0; i < prompt.size(); ++i) {
      ids_acc[b][offset + static_cast<int64_t>(i)] = prompt[i];
      mask_acc[b][offset + static_cast<int64_t>(i)] = true;
    }
  }

  // Real tokens are numbered 0, 1, 2, ...; padded cells reuse position 0 and are masked out anyway.
  batch.positions = (batch.padding_mask.to(torch::kInt64).cumsum(1) - 1).clamp_min(0);
  return batch;
}

}  // namespace gpt2
