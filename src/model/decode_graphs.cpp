#include "model/decode_graphs.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef GPT2_WITH_CUDA
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace gpt2 {

struct DecodeGraphs::Impl {
  Impl(const GPT2Model& model, KVCache& cache, int64_t length_step)
      : model(model), cache(cache), length_step(length_step), n_ctx(model.config().n_ctx) {}

  const GPT2Model& model;
  KVCache& cache;
  int64_t length_step;
  int64_t n_ctx;
  std::vector<int64_t> batch_buckets;  // ascending; the last one is the slot count
  torch::Tensor ids, positions;        // the graphs' fixed inputs, on the device [max batch]

#ifdef GPT2_WITH_CUDA
  struct Graph {
    std::unique_ptr<at::cuda::CUDAGraph> graph;
    torch::Tensor logits;  // the graph's fixed output [batch bucket, vocab]
  };
  std::map<std::pair<int64_t, int64_t>, Graph> graphs;
  decltype(at::cuda::graph_pool_handle()) pool;  // shared by every graph; set once CUDA is known to work

  Graph& capture(int64_t batch, int64_t length) {
    torch::InferenceMode guard;
    const auto ids_b = ids.narrow(0, 0, batch);
    const auto positions_b = positions.narrow(0, 0, batch);

    // Capture on a side stream, after a warm-up there: the first run on a stream initialises
    // cuBLAS state, which must not happen inside a capture.
    torch::cuda::synchronize();
    auto stream = c10::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    model.decode(ids_b, positions_b, length, cache, /*padded_length=*/true);
    stream.synchronize();

    Graph entry;
    entry.graph = std::make_unique<at::cuda::CUDAGraph>();
    entry.graph->capture_begin(pool);
    entry.logits = model.decode(ids_b, positions_b, length, cache, /*padded_length=*/true);
    entry.graph->capture_end();
    stream.synchronize();
    return graphs.emplace(std::make_pair(batch, length), std::move(entry)).first->second;
  }
#endif
};

bool DecodeGraphs::available() {
#ifdef GPT2_WITH_CUDA
  return torch::cuda::is_available();
#else
  return false;
#endif
}

DecodeGraphs::DecodeGraphs(const GPT2Model& model, KVCache& cache, int64_t length_step)
    : impl_(std::make_unique<Impl>(model, cache, length_step)) {
  if (!available()) throw std::runtime_error("CUDA graphs need a CUDA build of LibTorch and a CUDA device");
  if (!model.device().is_cuda()) throw std::runtime_error("CUDA graphs need the model on a CUDA device");
  if (length_step < 1) throw std::invalid_argument("length_step must be positive");

  const int64_t max_batch = cache.n_slots();
  for (int64_t b = 1; b < max_batch; b *= 2) impl_->batch_buckets.push_back(b);
  impl_->batch_buckets.push_back(max_batch);

#ifdef GPT2_WITH_CUDA
  impl_->pool = at::cuda::graph_pool_handle();
#endif
  const auto options = torch::TensorOptions().dtype(torch::kInt64).device(model.device());
  impl_->ids = torch::zeros({max_batch}, options);
  impl_->positions = torch::zeros({max_batch}, options);
}

DecodeGraphs::~DecodeGraphs() = default;

int64_t DecodeGraphs::batch_bucket(int64_t batch) const {
  for (int64_t b : impl_->batch_buckets) {
    if (b >= batch) return b;
  }
  throw std::invalid_argument("batch " + std::to_string(batch) + " exceeds the cache's slots");
}

int64_t DecodeGraphs::length_bucket(int64_t length) const {
  const int64_t step = impl_->length_step;
  return std::min(impl_->n_ctx, (length + step - 1) / step * step);
}

int64_t DecodeGraphs::capture_all() {
#ifdef GPT2_WITH_CUDA
  for (int64_t batch : impl_->batch_buckets) {
    for (int64_t length = impl_->length_step;; length += impl_->length_step) {
      const int64_t bucket = std::min(length, impl_->n_ctx);
      if (!impl_->graphs.count({batch, bucket})) impl_->capture(batch, bucket);
      if (bucket == impl_->n_ctx) break;
    }
  }
  return static_cast<int64_t>(impl_->graphs.size());
#else
  return 0;
#endif
}

torch::Tensor DecodeGraphs::run(const torch::Tensor& ids_host, const torch::Tensor& positions_host, int64_t batch,
                                int64_t length) {
#ifdef GPT2_WITH_CUDA
  // The fixed inputs may have been created under InferenceMode (as inference tensors), and the
  // scheduler calls this from its own thread where the mode is off: copy into them inside it.
  torch::InferenceMode guard;
  TORCH_CHECK(batch >= 1, "batch must be positive");
  TORCH_CHECK(length >= 1 && length <= impl_->n_ctx, "length ", length, " outside [1, ", impl_->n_ctx, "]");
  const int64_t b = batch_bucket(batch);
  const int64_t l = length_bucket(length);
  TORCH_CHECK(ids_host.device().is_cpu() && positions_host.device().is_cpu(), "inputs must be host tensors");
  TORCH_CHECK(ids_host.scalar_type() == torch::kInt64 && positions_host.scalar_type() == torch::kInt64,
              "ids/positions must be int64");
  TORCH_CHECK(ids_host.size(0) >= b && positions_host.size(0) >= b, "host buffers must hold the batch bucket (", b,
              " rows)");

  // Padding rows: token 0 at position 0, in slots no active request occupies.
  auto ids_acc = ids_host.accessor<int64_t, 1>();
  auto positions_acc = positions_host.accessor<int64_t, 1>();
  for (int64_t row = batch; row < b; ++row) {
    ids_acc[row] = 0;
    positions_acc[row] = 0;
  }

  auto it = impl_->graphs.find({b, l});
  Impl::Graph& entry = it != impl_->graphs.end() ? it->second : impl_->capture(b, l);
  impl_->ids.narrow(0, 0, b).copy_(ids_host.narrow(0, 0, b), /*non_blocking=*/true);
  impl_->positions.narrow(0, 0, b).copy_(positions_host.narrow(0, 0, b), /*non_blocking=*/true);
  entry.graph->replay();
  return entry.logits.narrow(0, 0, batch);
#else
  (void)ids_host;
  (void)positions_host;
  (void)batch;
  (void)length;
  throw std::runtime_error("CUDA graphs are not available in this build");
#endif
}

}  // namespace gpt2
