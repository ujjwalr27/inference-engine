# GPT-2 Inference Engine (LibTorch) — Implementation Plan

A from-scratch GPT-2 serving engine in C++17 on LibTorch: HTTP server with token streaming,
a single-threaded continuous-batching scheduler, a preallocated slot-based KV cache, and a
load generator + benchmark suite run on a Kaggle T4.

Checklist lives in [TODO.md](TODO.md). This file holds the design and the decisions behind it.

---

## 1. Architecture

```
                 ┌──────────────────────────── server process ─────────────────────────────┐
  loadgen /      │                                                                         │
  curl  ──HTTP──▶│  HTTP worker threads (cpp-httplib pool)                                 │
                 │   • validate request, tokenize prompt (tokenizers-cpp)                  │
                 │   • push Request into bounded queue  ──(full → 429 Busy)                │
                 │   • wait on per-request TokenChannel, incremental-decode, stream SSE    │
                 │                    │                                ▲                   │
                 │                    ▼                                │ token ids         │
                 │            ┌───────────────┐                        │                   │
                 │            │ RequestQueue  │  (mutex + condvar)     │                   │
                 │            └───────┬───────┘                        │                   │
                 │                    ▼                                │                   │
                 │   Scheduler thread (ONLY thread touching model + cache)                 │
                 │    loop: evict finished/cancelled → admit + prefill (token budget)      │
                 │          → one batched decode step → publish tokens                     │
                 │                    │                                                    │
                 │          ┌─────────┴──────────┐                                         │
                 │          ▼                    ▼                                         │
                 │   GPT2Model (LibTorch)   KVCache [L, S, H, Tmax, Dh] ×2 (K, V)          │
                 └─────────────────────────────────────────────────────────────────────────┘
```

Design rules:
- **Single writer.** Only the scheduler thread touches the model and KV cache. No locks around tensors.
- **No allocation while serving.** KV cache and per-step staging buffers are allocated at startup.
- **Tokenization stays off the scheduler thread.** The scheduler only sees token IDs.
- **Same code on CPU and GPU.** `--device cpu|cuda`, `--dtype fp32|fp16`.

---

## 2. Environment & Toolchain

| Where | What |
|---|---|
| Local dev | **WSL2 Ubuntu 26.04** (not native Windows — see §10), gcc/clang, CMake ≥ 3.24, Ninja, LibTorch CPU (Linux, cxx11 ABI), Rust (`rustup`, for tokenizers-cpp), Python 3.12 venv. One-shot: `scripts/setup_wsl.sh` |
| Python venv | `torch` (CPU), `transformers`, `safetensors`, `pandas`, `matplotlib` |
| CI | GitHub Actions `ubuntu-latest`, CPU LibTorch (cached), build + ctest, TSan job on scheduler tests |
| GPU | Kaggle notebook, **T4** accelerator, LibTorch from Kaggle's pip `torch` (`torch.utils.cmake_prefix_path`) |
| Container | Dockerfile (CPU only), multi-stage: builder with Rust + LibTorch, slim runtime |

**Version pinning (confirmed 2026-09-17):**
| | Local (WSL2) | Kaggle |
|---|---|---|
| OS | Ubuntu 26.04.1, 16 threads (LibTorch uses 8), ~6 GB RAM visible to WSL | Ubuntu 22.04 image, **4 vCPU**, 31 GB RAM, 20 GB `/kaggle/working` |
| Compiler / CMake | GCC 15.2 | **GCC 11.4**, CMake 3.31, Ninja 1.13 — stay strictly C++17 so both compile |
| PyTorch / LibTorch | LibTorch **2.10.0+cpu** (`libtorch-shared-with-deps-2.10.0%2Bcpu.zip`, cxx11 ABI only) | pip `torch 2.10.0+cu128` (2.11.0+cu128 since 2026-10; both pass), `_GLIBCXX_USE_CXX11_ABI = True` |
| CUDA | — | **12.8**, nvcc 12.8.93 present, driver 580, **2× Tesla T4** (sm 7.5, 15 GB each), cuDNN 9.10 |
| Python | 3.12 via `uv` (system 3.14 left alone) | 3.12.13 |
| Rust | via `rustup` | **not installed** → `rustup` in the job, or prebuilt tokenizers libs as a dataset |

**Phase 0 results (2026-09-17):**
- Local CPU: build + 5 tests pass (CUDA test skipped), 1024×1024 fp32 matmul ≈ 31 ms.
- Kaggle T4: C++ program linked against pip torch runs fp16 matmul + `scaled_dot_product_attention` on `cuda:0`.
- Kaggle CMake warnings that are **harmless**: nvrtc shorthash, `kineto_LIBRARY-NOTFOUND`, cuDNN/cuSPARSELt/cuDSS/cuFile off (none needed).
- Use `TORCH_CUDA_ARCH_LIST=7.5`; Torch ignores `CMAKE_CUDA_ARCHITECTURES`.
- Kaggle gives **two** T4s: pin the engine to one (`CUDA_VISIBLE_DEVICES=0`) so benchmarks are single-GPU.
- Only 4 vCPUs on Kaggle: loadgen + server contention is real (see §10).

**Reproducible third-party builds:** cpp-httplib enables OpenSSL, zlib, brotli and zstd whenever it *detects* them, so the build silently depends on what each machine has installed. On GitHub's runners zstd was detected but the imported target it then links (`zstd::libzstd`) was never defined, so configure failed there while succeeding locally. All four are forced off in `CMakeLists.txt`: the engine serves plain local HTTP for benchmarking, and WSL, CI and Kaggle now build identically.

**Repo location:** source stays on the Windows side (`C:\Users\Ujjwal\Desktop\Projects\inference_engine`, i.e. `/mnt/c/...` in WSL); the **build directory lives on the Linux filesystem** (`~/build/gpt2-engine-*`) so compile I/O stays fast. `.gitattributes` forces LF line endings.

**WSL memory:** WSL sees ~6 GB by default, and a torch + GoogleTest translation unit needs ~2 GB to compile, so `scripts/build.sh` defaults to `JOBS=2`. Building with 6 jobs made WSL thrash badly enough to stop responding. Raise `JOBS` only after giving WSL more memory in `%UserProfile%\.wslconfig` (`[wsl2]` → `memory=10GB`).

**Third-party (C++):**
| Library | Use | How |
|---|---|---|
| LibTorch | tensors, kernels | downloaded zip locally / pip torch on Kaggle |
| GoogleTest | tests | `FetchContent` |
| nlohmann/json | safetensors header, HTTP JSON | `FetchContent` |
| cpp-httplib | HTTP server + loadgen client | `FetchContent` (header-only) |
| mlc-ai/tokenizers-cpp | HF tokenizer (`tokenizer.json`) | git submodule, needs `cargo`; `MLC_ENABLE_SENTENCEPIECE_TOKENIZER=OFF` |

---

## 3. Repository Layout

```
gpt2-engine/
  CMakeLists.txt
  cmake/                    # FindLibTorch helpers, sanitizer options
  third_party/tokenizers-cpp/  (submodule)
  scripts/
    export_weights.py       # HF → weights/model.safetensors (+ Conv1D transposed, c_attn split) + tokenizer.json
    reference.py            # HF logits / greedy outputs → tests/data/*.safetensors|json
    kaggle_run.py           # build + test + bench entry point on Kaggle
    plot_results.py         # CSV → charts
  src/
    common/                 # config, device/dtype options, timing, logging
    io/safetensors.{h,cpp}  # minimal safetensors reader
    model/                  # Embeddings, LayerNorm, Attention, MLP, Block, GPT2Model
    cache/                  # KVCache (slots, free list, compaction)
    scheduler/              # Request, RequestQueue, Scheduler, policies (static/continuous)
    tokenizer/              # Tokenizer wrapper + IncrementalDecoder
    server/                 # HTTP handlers, SSE streaming, main.cpp
  tools/
    generate.cpp            # CLI: prompt → text (Phase 2 demo)
    loadgen.cpp             # open-loop load generator → CSV
    bench.cpp               # micro-benchmarks (prefill/decode step timing)
  tests/
    test_*.cpp              # model tests read weights/ (override with GPT2_DATA_DIR) and skip if it's missing
  results/                  # CSVs + PNG charts, one subfolder per run (tagged with GPU/CPU)
  weights/                  # gitignored: model.safetensors, config.json, tokenizer.json, reference.safetensors/json
```

---

## 4. Model

### 4.1 Weight export (`scripts/export_weights.py`)
- Load `GPT2LMHeadModel.from_pretrained("gpt2")`, `float32`.
- Transpose every Conv1D weight (`attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`) so C++ can use `torch::linear` (`[out, in]`).
- Split `c_attn` (2304-wide) into `q_proj`, `k_proj`, `v_proj` (weights and biases).
- `lm_head` is tied to `wte`; export only `wte`, C++ reuses it.
- Write `weights/model.safetensors` plus `weights/config.json` (n_layer, n_head, n_embd, n_ctx, vocab, eps) so gpt2-medium works without code changes.
- `AutoTokenizer.from_pretrained("gpt2").save_pretrained("weights/")` → `tokenizer.json`.

### 4.2 Safetensors loader (`src/io/safetensors`)
Format: `u64 header_len` (little endian) → JSON header `{name: {dtype, shape, data_offsets}}` → raw bytes.
Read the whole file, `torch::from_blob` each tensor, `.clone()` and `.to(device, dtype)`. About 60 lines.

### 4.3 Layers
| Piece | Notes |
|---|---|
| Embedding | `wte[token_ids] + wpe[position_ids]` — position IDs are **per row** |
| LayerNorm | `torch::layer_norm`, **eps = 1e-5** |
| Attention | q/k/v linear → reshape `[B, H, T, 64]` → cache write → attention → `c_proj`. Scale `1/sqrt(64)` |
| MLP | `c_fc` (768→3072) → **gelu_new** → `c_proj`. `gelu_new(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` — written explicitly, *not* `torch::gelu(x)` |
| Block | `x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))` |
| Head | `ln_f` → `linear(x, wte)` |

Mask fill value: `torch::finfo(dtype).min`-equivalent (`-3.4e38` for fp32, `-65504` for fp16), **never `-inf` or `-1e9`** (NaN on fully-masked rows / fp16 overflow).

### 4.4 Model API
```cpp
// Uncached full forward (reference path, Phase 1–3)
Tensor forward(const Tensor& ids /*[B,T]*/, const Tensor& pos /*[B,T]*/, const Tensor& attn_mask /*[B,T]*/);

// Cached paths (Phase 2+)
Tensor prefill(const Tensor& ids /*[1,T]*/, int slot, KVCache&);              // returns last-token logits [1,V]
Tensor decode(const Tensor& ids /*[B]*/, const Tensor& pos /*[B]*/, int batch, KVCache&); // logits [B,V]; rows 0..B-1 = slots 0..B-1
```
Everything runs under `torch::InferenceMode guard;`.

---

## 5. KV Cache (`src/cache`)

- Two tensors `K, V` of shape `[n_layer, n_slots, n_head, n_ctx, head_dim]`, allocated once.
- Memory per slot = `12 × 12 × 1024 × 64 × 2 × bytes` → **~72 MiB fp32 / ~36 MiB fp16** (GPT-2 small).
- **Compaction:** active requests always occupy slots `0..B-1`. When the request in slot `i` finishes, copy the last active slot `B-1` into `i` (only `[0, len)`), update that request's slot index, `B -= 1`. Decode then uses `K[l].narrow(0, 0, B)` — a view, no copy.
- **Write:** `K[l].index_put_({arange(B), :, pos}, k_new)` via advanced indexing (one scatter per layer).
- **Read length:** slice time dim to `T_eff = max(pos) + 1`; mask `key_j` for row `i` iff `j > pos[i]`.
- **CUDA graphs** (`model/decode_graphs`, `--cuda-graphs`): a graph per bucket, batch ∈ {1, 2, 4, …, n_slots} and `T_eff` rounded up to a multiple of 128 (8 lengths, so 48 graphs for 32 slots, captured in 0.7 s on the T4 when the scheduler starts). Padding is safe by construction: extra keys lie past every row's position, so the decode mask (now built even for one row) hides them; padding rows feed token 0 at position 0 into slots `B..bucket-1`, which no active request occupies and which a prefill overwrites before anything reads them. All graphs share one memory pool; they never run concurrently and each output is read before the next replay.

---

## 6. Scheduler (`src/scheduler`)

### 6.1 Request
See `scheduler/request.h`. Fields fall into three groups, which is what keeps the locking simple:
- **submit-time, then read-only**: `id`, `prompt`, `options`, `out` (the `TokenChannel`)
- **written by any thread**: `cancelled` (atomic)
- **scheduler-thread only**: `generated`, `slot`, `position`, `next_token`, `finish_reason`, timestamps

`slot` always equals the request's index in the scheduler's active list, which is what makes compaction a two-line operation.

`position` is where the *next* fed token will land, so it doubles as the number of cached tokens — exactly the length `copy_slot` needs when compacting.

### 6.2 Step loop (continuous policy)
Implemented in `scheduler/scheduler.cpp`:
1. **Admit:** while free slots remain and the prefill budget (`prefill_budget_tokens`, default 512) is not spent → `prefill` into the next free slot, publish the first token. A request whose prompt alone exceeds the budget is still admitted, so nothing can starve.
2. If nothing is active → wait on the queue condition variable (no busy spin), then loop.
3. **Decode:** build `ids[B]`, `positions[B]`, `length = max(position) + 1` → one `decode` → argmax on the device → **one** `.to(cpu)` for the whole step.
4. **Publish:** push each token to its request's channel.
5. **Evict:** walk the active list backwards; requests that hit EOS, `max_new_tokens`, the context limit, or `cancelled` close their channel and free their slot by compaction (last active slot copied over the freed one, its `slot` index updated). Backwards means compaction can never skip a row.

On `stop()`: close the queue, fail whatever is still in flight, drain the waiting requests.

### 6.3 Static policy (baseline for benchmarks)
`--policy static` changes only *when* requests run, never what they produce. The scheduler gathers up to `static_batch_size` requests (waiting at most `static_max_wait` so a trickle still runs), admits them together, and **admits nothing more until every one of them has finished**. A request arriving one step too late waits for the whole batch; a request that finishes early leaves its slot idle. That is the cost continuous batching removes, and both policies share the same model, cache and decode path, so a comparison measures scheduling alone.

`scheduler/static_batch.cpp` remains the offline padded-batch path, and reports the other two costs: `padding_waste` and `wasted_decode_rows`.

**Padded cache layout.** Static batching keeps rows in their padded form inside the cache, so a token's cache column is no longer its position id. Two things follow, and they are why the model has a separate `decode_padded`:
- `positions` (for the position embedding) comes from `cumsum(mask) - 1`; `cache_index` is the shared column `T + step`.
- the decode mask is an explicit `key_mask [B, length]` (padding columns excluded), not `arange <= position`.

Continuous batching (§6.2) avoids all of this: each request is prefilled on its own, left-aligned in its slot, so cache column == position id and the mask comes straight from the positions.

### 6.4 Admission control
- Bounded queue `--max-queue`. Full → HTTP 429.
- Reject `len(prompt) + max_tokens > n_ctx` with HTTP 400.

---

## 7. Tokenizer (`src/tokenizer`)

- `tokenizers::Tokenizer::FromBlobJSON(read("weights/tokenizer.json"))`, `Encode`, `Decode`.
- **One instance, loaded at server start and shared by every HTTP worker**, with calls serialised by a mutex inside the wrapper: tokenizers-cpp's `Decode` writes its result into a buffer held by the handle and reads it back, so concurrent calls on one handle could swap results. The lock is held for microseconds.
- It was originally one instance per worker thread (`thread_local`). Loading `tokenizer.json` takes ~200 ms, and with ~300 workers most low-load requests landed on a thread paying that load — the Hugging Face baseline exposed it (see the Phase 7 results).
- **IncrementalDecoder** for streaming: decode only the tokens since the last emitted character and emit that text; hold back while it ends in `U+FFFD` (incomplete UTF-8 sequence). Byte-level BPE decodes each token to its own bytes, so decoding can restart at any character boundary and the pieces always concatenate to the full decode (checked on 3,000 random streams, half built from tokens that are partial characters on their own). It used to re-decode the whole output every token, under the lock every stream shares: 500,500 token decodes (75 ms) for a 1,000-token answer, now 1,499 (1 ms).
- `vocab_size()` is read once at load: the bundled tokenizers 0.21 answers it by building the whole 50k-entry vocabulary map, ~8 ms a call (a test that called it per token took 417 s).

---

## 8. Server & Load Generator

### 8.1 HTTP API
| Method | Path | Body / Response |
|---|---|---|
| POST | `/v1/generate` | `{"prompt": str \| "token_ids": [int], "max_tokens": int, "stream": bool, "temperature"?: float, "top_k"?: int}` |
| | | stream → `text/event-stream`: `data: {"text": "...", "token_id": n}` … `data: [DONE]` |
| | | non-stream → `{"text": ..., "token_ids": [...], "timings": {...}}` |
| GET | `/health` | `200 ok` |
| GET | `/stats` | queue depth, active slots, tokens/s (optional) |

Errors: `400` invalid/too long, `429` queue full, `503` shutting down.

**cpp-httplib thread pool:** each streaming response holds a worker thread for its whole lifetime. The pool size must be `> n_slots + max_queue` (set via `svr.new_task_queue`), otherwise concurrency is silently capped (default ≈ 8).

**Cancellation:** stream provider checks `sink.is_writable()`; on disconnect set `request->cancelled = true`.

### 8.2 Load generator (`tools/loadgen.cpp`)
- **Open loop:** Poisson arrivals at `--rate` req/s for `--duration`; each request fires on schedule regardless of earlier responses (one thread per in-flight request, or a thread pool larger than the peak concurrency).
- Prompt length / output length from a configurable distribution (uniform, or sampled from a file of prompts).
- Records per request: `id, prompt_tokens, output_tokens, t_send, t_first_token, t_done, status`.
- Also accepts `token_ids` mode so tokenizer cost can be excluded.

### 8.3 Metrics
- TTFT p50/p90/p99, TPOT (= (t_done − t_first) / (out_tokens − 1)) p50/p99, end-to-end latency, output tokens/s, completed req/s, 429 rate.
- Server-side timestamps split TTFT into **queue wait** vs **prefill compute**.

---

## 9. Correctness Contract

Text equality alone is flaky: batch shape changes BLAS kernel choice → ~1e-6 differences → a near-tie argmax can flip. So:

| Test | Primary check | Secondary check |
|---|---|---|
| Phase 1 layers/model vs HF | logits `allclose(rtol=1e-4, atol=1e-4)` + same argmax | — |
| Cached vs uncached | teacher-forced logits allclose at every step | greedy tokens identical, or first divergence at top-2 gap < 1e-3 |
| Batched vs alone | same as above, per row | same |
| Continuous (100 staggered) | same as above, per request | same |
| GPU fp16 vs CPU fp32 | log-softmax `atol ≈ 1e-2` (calibrate), top-1 agreement rate reported | divergence position logged, not hidden |

Reference runs use HF with `attn_implementation="eager"`, `float32`, and the same thread count.

**Phase 7 results (Kaggle T4, fp16, 2026-10-05, after the tokenizer fix).** One session, one build (Kaggle now ships torch 2.11.0+cu128; 79/79 tests pass on it). Every run: prompts of 32–256 tokens, 64 output tokens, open-loop Poisson arrivals for 20 s, each configuration on a fresh server. Data and charts in `results/2026-10-05_t4/`; regenerate the charts with `python scripts/plot_results.py readme results/2026-10-05_t4 --rate 32`. Zero transport failures in all 25 runs.

**Baseline: the engine vs plain Hugging Face.** `scripts/hf_server.py` serves `transformers` `generate()` (fp16, on the same GPU) behind the engine's own API, one request at a time; `scripts/kaggle_baseline.py` drives both with the same load generator and seed.

| Load | Engine TTFT p50 / p99 | HF TTFT p50 / p99 | Engine TPOT p50 | HF TPOT p50 | Engine tok/s | HF tok/s |
|---|---|---|---|---|---|---|
| 1 req/s | **7 / 23 ms** | 16 / 864 ms | **5.0 ms** | 8.6 ms | 62 | 60 |
| 2 req/s | **8 / 20 ms** | 958 ms / 3.9 s | **5.4 ms** | 8.2 ms | 127 | 118 |
| 4 req/s | **9 / 21 ms** | 11.0 / 20.8 s | **5.6 ms** | 8.3 ms | 246 | 120 |
| 8 req/s | **9 / 20 ms** | 37.1 / 74.2 s | **5.9 ms** | 8.5 ms | **551** | 117 |

- **4.7× the throughput at 8 req/s**, where the engine is nowhere near its limit (capacity ~50 req/s, below). Serial `generate()` tops out at ~1.9 req/s (~118 tok/s): 64 tokens × 8.3 ms ≈ 0.53 s per request.
- **Each token is 1.4–1.7× cheaper**, even with little batching: `generate()` pays Python-level overhead every step (logits processors, stopping criteria, the streamer); the engine's decode loop is C++.
- **First token in 7–9 ms at every load**, against 16 ms for Hugging Face when idle and tens of seconds once its queue builds. A whole 64-token request takes 380 ms at 8 req/s, against 37.6 s.
- This is the naive baseline. Hugging Face TGI and vLLM batch continuously too; vLLM is measured in the Phase 8 results below.
- Kaggle sessions vary: the 2026-10-01 session measured Hugging Face at 11.3 ms/token and 87 tok/s (6.3× at 8 req/s) and the engine at 6.7–7.3 ms/token. Compare within a session only.

**Continuous vs static batching** (32 slots; static gathers up to 32 requests for at most 50 ms, then runs that batch to completion).

| Rate | Policy | Served | TTFT p50 | TTFT p99 | TPOT p50 | Total p50 | Output tok/s |
|---|---|---|---|---|---|---|---|
| 2 | continuous | 40/40 | **7 ms** | 242 ms | 5.4 ms | **348 ms** | 127 |
| 2 | static | 40/40 | 139 ms | 420 ms | 4.8 ms | 500 ms | 126 |
| 8 | continuous | 174/174 | **9 ms** | 91 ms | 5.9 ms | **379 ms** | 551 |
| 8 | static | 174/174 | 255 ms | 472 ms | 5.9 ms | 635 ms | 549 |
| 16 | continuous | 355/355 | **10 ms** | 144 ms | 6.6 ms | **427 ms** | 1111 |
| 16 | static | 355/355 | 281 ms | 535 ms | 6.3 ms | 682 ms | 1097 |
| 32 | continuous | 705/705 | **11 ms** | 272 ms | 7.9 ms | **509 ms** | 2205 |
| 32 | static | 705/705 | 350 ms | 663 ms | 7.3 ms | 815 ms | 2182 |
| 64 | continuous | 1259/1345 (86 × 429) | 3.7 s | 4.9 s | 9.6 ms | 4.3 s | 3194 |
| 64 | static | 1260/1345 (85 × 429) | 3.8 s | 4.8 s | 8.2 ms | 4.3 s | 3214 |

- **Below capacity, continuous batching cuts first-token latency 95–97% and whole-request latency 30–40%** at the same throughput. Under static batching a newcomer waits for the running batch to finish all 64 steps (~0.4 s); under continuous it joins at the next step.
- Static has slightly lower TPOT (7.3 vs 7.9 ms at 32 req/s): continuous batching interleaves newcomers' prefills between running requests' decode steps. That is the trade it makes.
- **Capacity is ~50 req/s (~3200 output tok/s)** for this workload, for both policies. Throughput is equal because every request produces exactly 64 tokens, so static batching never strands a finished request's slot. That is continuous batching's other advantage, and this workload cannot show it until the load generator varies output lengths.
- Overload is handled cleanly: at 64 req/s the queue fills and the excess gets 429s, with no dropped connections.

**Slots vs throughput** (saturating load, 64 req/s offered): 200 / 649 / 1188 / 2048 / 3239 tok/s for 1 / 4 / 8 / 16 / 32 slots, **16× from batching and still rising at 32** (the last doubling added 58%). TPOT grows from 4.9 to 9.5 ms over the same range; that is the per-request price of the throughput.

**fp32 vs fp16 serving at 16 req/s** (below capacity): same throughput (1109 vs 1111 tok/s; the load sets it), but fp16 is faster per request: TTFT p50 10 vs 16 ms, TPOT 6.6 vs 8.6 ms, whole request 427 vs 562 ms. fp16 also halves the KV cache (1152 vs 2304 MiB for 32 slots).

**Served vs isolated decode speed.** At saturation the server ran 2522 decode steps in ~25 s, about 10 ms per step including the prefills between them, against 5.3–6.1 ms for an isolated batch-32 fp16 decode step. Served throughput is ~54% of the isolated decode rate. Before the tokenizer fix, the server managed only ~15% of it (~40 ms per step, ~730 tok/s at an apparent capacity of ~11 req/s). That gap is gone: it was the tokenizer bug, not the 4-vCPU host. What remains is prefill time and host overhead, which scheduler-side timing in `/stats` would split.

**The bug the baseline found.** The first baseline run (2026-09-30) had Hugging Face *faster* to the first token at 1 req/s, 18 against 143 ms. Every HTTP worker thread loaded its own tokenizer on its first request; loading `tokenizer.json` measures 180–260 ms, and with ~300 workers nearly every low-load request landed on a thread that had not loaded one yet. Under heavy load it was worse: ~300 workers each parsing a 3.5 MB file is up to a minute of CPU time inside a 20 s run on 4 vCPUs. Fixed by sharing one tokenizer (§7).

**It also distorted time per token, in the flattering direction.** That run reported an engine TPOT of 3.6–4.1 ms, faster than a single decode step measured in the same session (5.0–5.3 ms), which is impossible. The scheduler kept generating while the HTTP thread was loading its tokenizer; tokens piled up in the request's channel and then left in a burst, so the first token looked late and the rest looked impossibly fast. The early claim of "2.3–2.6× cheaper per token" came from that artifact.

**The first sweep (2026-09-30), superseded.** It ran with the bug and concluded: capacity ~11 req/s (~730 tok/s), continuous batching ~40% lower latency than static at 8 req/s, throughput nearly flat past 16 slots, served throughput ~8× below the GPU's decode rate, and overload showing up as dropped connections instead of 429s. The re-run above overturns every one of those except the direction of the policy comparison; all five were symptoms of CPU time lost to tokenizer loads. Its data is in git history (`results/2026-09-30_t4/`, removed in the commit that added the re-run).

Two tooling bugs found along the way, both fixed:
- The "port may be shared" warning fired on 8 runs, but each gap was exactly the number of connection failures, i.e. requests the server never saw. The check now counts only requests that got an HTTP answer.
- `plot_results.py` computed wall time as `max(send) + max(done)`, pairing the latest send with the slowest request, so its tokens/s ran ~30% low. It now uses `max(send + done)` and reproduces the load generator's figure exactly.

**Phase 8 results: CUDA graphs, and the engine against vLLM (Kaggle T4, fp16, 2026-10-08).** One session and one build (`perf/scheduler-timing` at `18338e4`); 82/82 tests pass, including the two CUDA graph tests. Same workload as Phase 7: prompts of 32–256 tokens, 64 output tokens, open-loop Poisson arrivals for 20 s, a fresh server per configuration. Data and charts in `results/2026-10-08_t4_graphs/`; regenerate the charts with `python scripts/plot_results.py readme results/2026-10-08_t4_graphs`.

**A decode step, isolated** (`gpt2_bench`, fp16, ms per step):

| Cache length | Batch | Eager | CUDA graphs | Speed-up |
|---|---|---|---|---|
| 128 | 1 | 4.83 | **2.24** | 2.2× |
| 128 | 8 | 5.53 | **2.76** | 2.0× |
| 128 | 32 | 5.84 | **4.62** | 1.3× |
| 512 | 8 | 5.39 | **3.45** | 1.6× |
| 512 | 32 | **6.26** | 7.33 | 0.85× |
| 1023 | 32 | 10.13 | 10.12 | 1.0× |

- **The step was launch-bound, as predicted.** Replaying it as one graph halves it at small batches (4.8 → 2.2 ms for one request). The gain shrinks as batch × length grows, because then the GPU work itself dominates.
- **One corner is slower: batch 32 with a 512-token cache.** The length bucket rounds 513 up to 640, which adds 25% of attention work, and at that size the attention outweighs the launches saved. The served workload never gets there (its contexts stay under 320 tokens). A step of 64 would halve the worst-case padding; untested.

**Served, three servers in one session** (all fp16, at most 32 sequences in flight each):

| Rate | TPOT p50: engine / + graphs / vLLM | TTFT p50: engine / + graphs / vLLM | Whole request p50: engine / + graphs / vLLM |
|---|---|---|---|
| 4 | 6.0 / **2.5** / 2.8 ms | 9 / **7** / 21 ms | 384 / **164** / 195 ms |
| 8 | 6.4 / **2.7** / 3.0 ms | 10 / **8** / 21 ms | 410 / **177** / 208 ms |
| 16 | 7.0 / **3.4** / 3.7 ms | 10 / **8** / 24 ms | 452 / **219** / 258 ms |
| 32 | 8.2 / **5.5** / 5.6 ms | 11 / **9** / 32 ms | 528 / **356** / 381 ms |
| 64 | 10.1 / **9.0** / 9.2 ms | 3.8 / **2.6** / 3.2 s | 4.5 / **3.1** / 3.8 s |

At 64 req/s (past capacity): engine + graphs **3,486** output tok/s with all 1,345 requests served; vLLM 3,325; the eager engine 3,121 with 102 turned away with 429.

- **CUDA graphs cut time per token 2.4× at low load** (6.0 → 2.5 ms) and whole requests 2.3× (384 → 164 ms). Capacity rises ~12% (48.9 → 54.7 completed req/s).
- **Against vLLM, the engine with graphs is on par or slightly ahead on this workload:** time per token 2–10% lower at 4–32 req/s, whole requests 7–16% faster, first token 2.7–3.5× sooner (7–9 against 21–32 ms), and 5% more throughput past capacity. vLLM's extra first-token time is its Python API layer and request handling.
- **Read that narrowly.** GPT-2 small on a T4, where vLLM cannot use FlashAttention (it needs compute capability 8.0; vLLM ran Triton attention with `torch.compile` and its own CUDA graphs). Both servers were capped at 32 sequences in flight, below vLLM's default. Every request generated exactly 64 tokens, and this is one session. The fair claim is that the engine matches vLLM here, not that it beats vLLM in general.

**Where the scheduler's time goes** (the new `/stats` split, one 20 s run each):

| Rate | Decode ms/step: eager / graphs | Prefill: eager / graphs | Bookkeeping |
|---|---|---|---|
| 4 | 5.59 / 2.39 | 0.9 / 0.6 s | 0.1 s |
| 16 | 6.18 / 2.74 | 2.6 / 2.3 s | 0.3 s |
| 64 | 6.80 / 5.69 | 8.2 / 8.4 s | 0.7–0.8 s |

- Bookkeeping (admitting, publishing, finishing, compacting) is negligible: at most 0.8 s in a 25 s run.
- **Prefill is now the largest remaining cost.** It is ~34% of a saturated run with graphs. Each admitted request is prefilled on its own, ending in a device sync. The next step is to batch the prefills admitted in one step, or chunk them.

**Correctness.** Graph-replayed logits match the eager step on the same cache in fp32 and fp16 (same top token), with a padded batch (3 rows run as 4) and a padded length. Through the scheduler, all 8 requests (4 prompts × fp32 and fp16, 24 tokens each) produced exactly their solo-run output; none needed the near-tie allowance.

**The harness bug behind the first vLLM run (2026-10-07).** vLLM froze 19 s into the 32 req/s run and 7 s into the 64 req/s run; every later request hung until the client's 300 s timeout (28 and 865 failures). The scripts sent server output to a pipe that nothing read. Now that logs go to files, their sizes confirm it: vLLM wrote 25, 31 and 43 KB at 4, 8 and 16 req/s, and 66 and 109 KB at 32 and 64. It froze in exactly the two runs that outgrew the ~64 KB pipe. vLLM's log has one `ERROR` line per start (FlashAttention 2 needs compute capability 8.0); it falls back to Triton attention and is otherwise healthy.

**Phase 6 result (Kaggle T4, 2026-09-19): 75/75 tests pass**, including six GPU tests. The first run failed two of them; both were tolerances guessed on CPU, not engine faults (see below).

**The fp16 divergence question, settled with numbers.** Where batched fp16 output differs from a solo run, the gap between the top two logits at that step was **0** for one request (an exact tie in fp16 — the choice was arbitrary) and **0.0625** for the other, which is exactly one fp16 step at that magnitude. Neither is a scheduling bug; fp32 on the same path matches solo output exactly.

| | CPU fp32 (laptop) | T4 fp32 | T4 fp16 |
|---|---|---|---|
| prefill 1024 tokens | 3387 ms | 83.2 ms | **28.1 ms** (36k tok/s) |
| decode, batch 1 | 65.0 ms | 5.28 ms | 4.99 ms |
| decode, batch 32 (len 128) | — | 6.22 ms (5144 tok/s) | **5.43 ms (5890 tok/s)** |
| decode, batch 32 (len 1023) | — | 14.9 ms (2148 tok/s) | **9.77 ms (3277 tok/s)** |
| KV cache, 32 slots | — | 2304 MiB | 1152 MiB |

**FP16 buys almost nothing at batch 1** (4.99 vs 5.28 ms) and 1.5× at batch 32 with a long cache — exactly the prediction in §10: a decode step launches ~200 small kernels, so at low batch the CPU is the bottleneck and the GPU idles. It is the strongest argument for CUDA graphs in Phase 8. Prefill, which is compute-bound, gets the full tensor-core win (3×).

**Server under load (T4 fp16, 32 slots, 8 req/s, 30 s):** 266/266 requests served, 0 rejected, **557 output tokens/s**, TTFT p50 83 ms / p99 461 ms, TPOT p50 6.3 ms. Peak batch was only 14 of 32 slots, so 8 req/s does not saturate this configuration — the Phase 7 rate sweep needs to push much harder.

**Benchmark hazard found here:** the second run's `/stats` reported 400 requests when the load generator had sent 266. A server from the previous cell was still alive, and because cpp-httplib sets `SO_REUSEPORT`, the kernel happily split connections between two engines sharing port 8099 — numbers from a mix of two processes, with no error anywhere. Two causes, both fixed in `scripts/kaggle_run.py`: `Popen(shell=True)` meant `terminate()` killed the shell and left the server running, and nothing checked the port first. The script now starts the server without a shell in its own process group, kills the group, refuses to run if the port already answers, and warns when the server's request count disagrees with the load generator's.

**Why the two tests failed, and what the right bar is:**
1. `Fp16AgreesOnTheTopTokenAndStaysClose` demanded log-probabilities within 0.05 of Hugging Face; the run showed 0.237 with **100% top-1 agreement**. Near a GPT-2 logit (~100) consecutive fp16 values are 0.06–0.125 apart, so the threshold asked for finer agreement than fp16 can represent. Raised to 0.5; ordering is what the top-1 check guards.
2. `SchedulerOnGpuMatchesSoloRuns` required batched fp16 output to track the solo run for at least 5 tokens; two requests diverged at tokens 4 and 8. Batching changes the reduction order, and fp16 cannot resolve a close race — the effect §9 predicted. Replaced by two tests: **fp32 must match exactly** (the real invariant), while fp16 may diverge **only where the top-2 logit gap is tiny**, with the gap printed either way.

CPU baseline from `gpt2_bench` (fp32, laptop under WSL), useful as the "before" column:

| decode batch | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| ms/step (cache len 128) | 65.0 | 63.3 | 85.0 | 133.0 |
| output tokens/s | 15.4 | 31.6 | 47.1 | 60.2 |

Batching 8 requests costs ~2× the time of one and returns ~4× the tokens. Prefill runs at 300–430 tokens/s and is roughly linear up to 512 tokens, then superlinear (attention is quadratic): 1024 tokens takes 3.4 s.

**Phase 5 result (CPU, 2026-09-18):** first end-to-end run — server with 8 slots, open-loop load at 2 req/s for 20 s, prompts of 16–64 tokens, 12 tokens out each. 39/39 requests succeeded, no 429s: **TTFT p50 955 ms / p99 1209 ms, TPOT p50 54.7 ms, 23.8 output tokens/s**, all 8 slots busy at the peak, 468 tokens in 141 decode steps (3.3 tokens/step). These are CPU fp32 numbers on a laptop under WSL — the point is the pipeline works end to end, not the values. CSV and charts in `results/`.

**Two non-obvious server details:** (1) cpp-httplib's default pool of ~8 threads would cap streaming concurrency regardless of slot count, because a streaming response holds its worker for its whole lifetime — the pool is sized `slots + max_queue + 8`; (2) a disconnected client is only noticed when a write fails *or* `sink.is_writable()` goes false, and an HTTP keep-alive client that stops reading does **not** close the socket, so the test has to destroy the client to exercise cancellation.

**Phase 4 result (CPU, 2026-09-18):** 100 staggered requests from 12 client threads, random prompts (3–24 tokens) and lengths (4–14 new tokens), through 6 slots: **every output identical to its solo run**, no near-tie divergence. Slots filled completely (max batch 6/6) and were recycled by compaction throughout — 887 tokens in 135 decode steps, i.e. **6.6 tokens per step** instead of one. TSan is clean (0 warnings) both for the threading tests and for the real scheduler thread with the model loaded.

**TSan gotcha:** a `called_from_lib` pattern must match exactly one loaded library. Plain `libtorch` also matches `libtorch_cpu.so`, and TSan then exits with an error *before running any test* — which looks exactly like a failure. Patterns in `.tsan-suppressions` are therefore written with the `.so` suffix.

**Phase 3 result (CPU, 2026-09-18):** batching does not change answers — prompts of 1, 10, 26 and 31 tokens produce identical text alone and batched together, and the padded batched prefill matches per-prompt logits within tolerance. That batch was **45.2% padding**, which is the cost Phase 4 removes. No near-tie divergence yet on CPU fp32.

**Phase 2 result (CPU, 2026-09-18):** cached and uncached greedy output is identical to Hugging Face for all 5 prompts with greedy references (50 tokens each), and teacher-forced logits match at every step. No near-tie divergence has appeared yet on CPU fp32. Cached decode is **2.9× faster** (54.3 vs 157.9 ms/token, 60 tokens from a 7-token prompt); the gap widens with longer outputs.

**Phase 1 result (CPU, 2026-09-17):** logits are bit-identical to HF for prompts of 26–1024 tokens. Very short prompts differ slightly (6.1e-5 at T=10, 2.6e-4 at T=1) because BLAS picks different kernels for tiny matrices. They pass on the relative tolerance, and the top token always matches. Keep using `rtol` + `atol` (never `atol` alone) because logits sit around −100.

---

## 10. Feasibility Notes & Known Constraints

Checked against the current dev machine (Windows 11, Ryzen 7 250, 13.7 GB RAM, Radeon 780M, w64devkit MinGW g++, WSL2 with only `docker-desktop`).

### Not feasible as-is (needs a change)
| Item | Why | Resolution |
|---|---|---|
| Building with the installed **MinGW g++** on Windows | Prebuilt LibTorch Windows binaries are MSVC-only; MinGW can't link them | Use **WSL2 Ubuntu** (recommended) or install MSVC Build Tools |
| **ThreadSanitizer** on Windows | Not supported by MSVC or MinGW | Run TSan in WSL2 / Linux CI; test scheduler with a mock model to avoid LibTorch/OpenMP false positives |
| Any **local GPU** work | No NVIDIA GPU (Radeon iGPU; LibTorch ROCm doesn't target it on Windows/WSL) | CPU locally, all GPU work on Kaggle (as planned) |
| **FlashAttention** on T4 | Needs compute capability ≥ 8.0; T4 is 7.5 | Use SDPA memory-efficient/math kernels or manual attention; say so in the README |
| **Kaggle P100** | Recent PyTorch builds dropped Pascal (sm 6.0) | Always select T4 |
| **GPU tests in GitHub Actions** | Free runners have no GPU | CI is CPU only; GPU correctness runs in the Kaggle job |
| **Public live demo from Kaggle** | Notebooks can't expose a public HTTP endpoint | Record the demo locally on CPU (GPT-2 small streams fine on CPU) |

### Feasible but must be verified early (Phase 0 spike)
- LibTorch CUDA build on Kaggle: C++ ABI flag (`torch._C._GLIBCXX_USE_CXX11_ABI`), CUDA toolkit/nvcc presence for Torch's CMake config.
- Rust toolchain install on Kaggle (needs Internet on) — or prebuild tokenizers static libs and attach as a Kaggle dataset.
- vLLM baseline: current release still supports Turing (T4) and must run in a **separate** job/venv (it replaces Kaggle's torch).

### Feasible with caveats
- **CUDA graphs:** need static shapes → bucket batch size and attention length.
- **Paged KV cache:** pure-LibTorch gather is likely slower than contiguous slots; worthwhile only with a custom kernel.
- **FP16 speedup at low batch:** GPT-2 small decode is CPU-launch-bound (~200 kernel launches/step), so expect small gains until batch grows. This is a finding, not a bug.
- **Kaggle CPUs (2–4 vCPU):** loadgen, HTTP threads, and scheduler share cores → label charts with CPU and GPU, repeat runs ×3, report median.
- **Local RAM 13.7 GB:** fine for GPT-2 small/medium; limit build parallelism (`-j4`) if linking LibTorch runs out of memory.
- **Kaggle quota:** ~30 GPU hours/week, 12 h/session. Keep each benchmark job short.

---

## 11. Phases (summary)

| # | Phase | Est. | Done when |
|---|---|---|---|
| 0 | Setup + Kaggle CUDA spike | 3–4 d | Hello-tensor runs locally (WSL2), in CI (green), and on Kaggle T4 with CUDA |
| 1 | Weights + forward pass | 1–1.5 wk | Logits match HF for ≥ 5 prompts |
| 2 | Generation + KV cache + tokenizer | 1.5 wk | Uncached = cached = HF greedy for 50 tokens; cached much faster; CLI takes text |
| 3 | Static padded batching | 1 wk | Batch-invariance test passes for mixed lengths |
| 4 | Slot cache + continuous batching | 2–2.5 wk | 100 staggered random requests match their solo runs; TSan clean |
| 5 | Server + loadgen | 1–1.5 wk | Local CPU run produces a sensible CSV |
| 6 | Kaggle GPU + FP16 | 1 wk | Kaggle job builds, passes GPU tests, saves CSV |
| 7 | Benchmarks | 1 wk | 4–5 labelled charts |
| 8 | Stretch (CUDA graphs, custom kernel, paged cache) | CUDA graphs done | per item |
| 9 | Polish | 3–4 d | README, charts, demo video, Dockerfile |

Realistic total without stretch goals: **~12–14 weeks part-time.**
