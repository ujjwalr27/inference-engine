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
| PyTorch / LibTorch | LibTorch **2.10.0+cpu** (`libtorch-shared-with-deps-2.10.0%2Bcpu.zip`, cxx11 ABI only) | pip `torch 2.10.0+cu128`, `_GLIBCXX_USE_CXX11_ABI = True` |
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
- Later (CUDA graphs): replace dynamic `B`/`T_eff` with buckets (e.g. B ∈ {1,2,4,8,16,32}, T ∈ {128,256,512,1024}).

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
Wait until `N` requests or a timeout, left-pad, run the batch to completion, only then admit more. Reuses the Phase 3 padded-batch path (`scheduler/static_batch.cpp`), which already reports the two costs the Phase 7 chart is about: `padding_waste` (share of the padded prompt block that is padding) and `wasted_decode_rows` (row-steps spent on rows that already finished).

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
- One instance per HTTP thread (`thread_local`) unless thread safety of a shared handle is verified.
- **IncrementalDecoder** for streaming: keep all generated IDs, decode the full list each step, emit only the new suffix; hold back if the text ends in `U+FFFD` (incomplete UTF-8 sequence).

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
| 8 | Stretch (CUDA graphs, custom kernel, paged cache) | open | per item |
| 9 | Polish | 3–4 d | README, charts, demo video, Dockerfile |

Realistic total without stretch goals: **~12–14 weeks part-time.**
