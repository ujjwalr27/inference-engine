# TODO

Design and rationale: [IMPLEMENTATION.md](IMPLEMENTATION.md). Tick items as they land.

---

## Phase 0 — Setup + Kaggle CUDA spike (3–4 days)

### Local environment
- [x] Install WSL2 Ubuntu (got 26.04.1)
- [x] Run `bash scripts/setup_wsl.sh` inside WSL: apt tools, LibTorch 2.10.0 CPU, Rust, uv + Python 3.12 venv
- [x] Pin LibTorch to 2.10.0 (matches Kaggle `torch 2.10.0+cu128`)
- [x] Decision: source on `/mnt/c`, build dir on Linux fs (`scripts/build.sh`)

### Repo skeleton
- [x] `git init`, `.gitignore`, `.gitattributes` (LF)
- [x] Root `CMakeLists.txt`: C++17, `find_package(Torch)`, warnings, `GPT2_ENABLE_ASAN` / `GPT2_ENABLE_TSAN`
- [x] `FetchContent`: GoogleTest (nlohmann/json in Phase 1, cpp-httplib in Phase 5)
- [x] `src/common/device.{h,cpp}`: `--device` / `--dtype` parsing
- [x] `tools/hello_tensor.cpp`: create and print a tensor, report device, time a matmul
- [x] `tests/test_smoke.cpp` wired into `ctest`
- [x] `bash scripts/build.sh` passes locally (GCC 15.2, 5 tests, CUDA test skipped)

### CI
- [x] `.github/workflows/ci.yml`: ubuntu-latest, cached LibTorch, configure, build, ctest, hello_tensor
- [x] Repo pushed to https://github.com/ujjwalr27/inference-engine (29 commits, submodule included)
- [x] First green run: `build-test-cpu` + `tsan-scheduler` both pass
      (needed one fix: cpp-httplib's optional OpenSSL/zlib/brotli/zstd backends turned off — see IMPLEMENTATION.md §2)
- [ ] CI currently skips every model test (no weights). Add a job that caches the HF download and runs both scripts

### Kaggle spike (main risk)
- [x] Kaggle T4 notebook: torch 2.10.0+cu128, CUDA 12.8, Tesla T4
- [x] Run `scripts/kaggle_spike.py`: ABI=cxx11, nvcc 12.8 present, 4 vCPU, 2× T4, no cargo; CUDA C++ fp16 matmul + SDPA pass
- [x] Paste spike output into IMPLEMENTATION.md §2
- [ ] Rust missing on Kaggle → in Phase 2, try `rustup` in the job; fall back to prebuilt tokenizers libs as a dataset

**Done when:** hello-tensor runs in WSL2, CI is green, and a CUDA tensor is printed from C++ on a Kaggle T4.

---

## Phase 1 — Weights + forward pass (1–1.5 weeks)

### Export
- [x] `scripts/export_weights.py`: load `gpt2`, transpose Conv1D weights, split `c_attn` → q/k/v, drop `lm_head` (tied)
- [x] Write `weights/model.safetensors` (196 tensors, 124.4M params) + `weights/config.json`
- [x] Save `tokenizer.json` (for Phase 2)
- [x] Sanity check in Python: plain-PyTorch forward over exported tensors matches HF (max diff 0)

### Reference data
- [x] `scripts/reference.py`: HF `eager` attention, fp32, 8 threads
- [x] 7 prompts: 1, 10, 26, 31, 39 (unicode/emoji), 500, 1024 tokens
- [x] Logits at 5 positions per prompt + all 13 hidden states for prompt 0 → `weights/reference.safetensors` (+ `reference.json`)

### C++
- [x] `src/io/safetensors`: validated header parse, dtype map, reads bytes straight into the tensor
- [x] `GPT2Config` loaded from `config.json`
- [x] Embedding (token + per-row position)
- [x] `LayerNorm` (eps 1e-5)
- [x] `Attention` (uncached, bool mask + dtype-safe fill value, padding-mask ready)
- [x] `MLP` with explicit `gelu_new`
- [x] `Block`, `GPT2Model::forward(ids, positions, padding_mask)`, tied head, `hidden_states()`
- [x] `torch::InferenceMode` in forward and load
- [x] `tools/forward.cpp` (`gpt2_forward --ids ...`): load + forward timing, top-k

### Tests (24 total, all pass on CPU; CUDA test skipped)
- [x] Safetensors loader: values, metadata, truncated data, size mismatch, bad header length/JSON/dtype, missing file
- [x] Layers: gelu_new vs exact GELU, fp16 mask value, mask shape, causality, padded keys ignored, block shape, config
- [x] Per-layer hidden states vs HF (all 13 within tolerance)
- [x] Logits `allclose(rtol=1e-4, atol=1e-4)` + argmax for all 7 prompts. Max |diff|: 0 for T ≥ 26, 6.1e-5 (T=10), 2.6e-4 (T=1, passes on rtol; logits are ~−100)
- [x] Batch of 3 identical rows == single row
- [ ] CI: reference tests currently **skip** on GitHub (no weights). Add a job that caches the HF download and runs both scripts

**Done when:** C++ logits match HF for ≥ 5 prompts.

---

## Phase 2 — Generation + KV cache + tokenizer (1.5 weeks)

### Uncached generation
- [x] `generate_uncached(prompt_ids, n)`: full forward each step, greedy
- [x] Stop on EOS (50256) / `n` / `n_ctx`

### KV cache
- [x] `KVCache [L, S, H, n_ctx, 64]` ×2, allocated once, slot convention 0..B-1 (Phase 4 ready)
- [x] `prefill(ids, slot, cache)`: writes K/V `[0, T)`, returns last-position logits only
- [x] `decode(ids[B], positions[B], length, cache)`: `index_put_` write, view (not copy) read, per-row mask
- [x] `length` passed in by the caller so decode needs no GPU→CPU sync (Phase 6 groundwork)
- [x] `generate_cached` / `generate_greedy` with token callback for streaming

### Tokenizer
- [x] `third_party/tokenizers-cpp` submodule, `tokenizers_cpp` linked, SentencePiece off
- [x] `Tokenizer` wrapper (pimpl): `encode`, `decode`, `vocab_size`
- [x] `IncrementalDecoder`: emits new suffix only, holds back trailing `U+FFFD`
- [x] Test: 2000 cases match Python exactly (unicode, emoji, ZWJ sequences, code, whitespace, URLs); decode round-trips
- [x] Test: emoji/CJK streaming never emits `U+FFFD`, and reassembles to the full text

### Tests / tooling
- [x] `reference.py`: HF greedy 50 tokens for prompts 0–4, plus 2000 tokenizer cases
- [x] Teacher-forced: cached logits vs uncached at every step (allclose)
- [x] Greedy text: uncached == cached == HF for all 5 prompts (near-tie rule ready, not needed yet)
- [x] `tools/generate.cpp` CLI: `--prompt "..." --max-tokens N --cache on|off`, streams tokens, prints ms/token
- [x] Timing: 60 tokens from a 7-token prompt — cached 54.3 ms/token vs uncached 157.9 ms/token (**2.9×**)

**Done when:** all three greedy outputs agree, the cached path is clearly faster, and the CLI takes text. ✅

---

## Phase 3 — Static padded batching (1 week)

- [x] Left padding + attention mask `[B, T]` (`scheduler/padding.cpp`)
- [x] Per-row position IDs = `cumsum(mask) - 1` (clamped at 0 for pads)
- [x] Combined causal + padding mask; no NaN on pad rows (covered by `Layers.PaddedKeysAreIgnored`)
- [x] Batched prefill into slots 0..B-1 (`prefill_batch`, `KVCache::write_prefill_batch`)
- [x] Batched decode where cache column ≠ position id (`decode_padded` takes `cache_index` + explicit `key_mask`)
- [x] Per-row stopping; finished rows stay in the batch and are counted as `wasted_decode_rows`
- [x] One CPU transfer of the chosen tokens per step (Phase 6 groundwork)
- [x] Test: batched prefill logits == per-prompt prefill logits (strict, allclose)
- [x] Test: batch invariance — prompts of 1/10/26/31 tokens, alone vs batched, **identical text** (45.2% of the padded block was padding)
- [x] Test: batch of 1 == unbatched path; per-row streaming callback; input validation
- [x] Keep this path as the static policy for Phase 7 (`generate_batch_static`)

**Done when:** batch-invariance test passes for a mix of short and long prompts. ✅ (49 tests, all pass)

---

## Phase 4 — Slot KV cache + continuous batching (2–2.5 weeks)

### KV cache manager
- [x] Allocated once in the constructor (done in Phase 2)
- [x] Active count `B` owned by the scheduler; slots stay packed at the front (`slot == index`)
- [x] Compaction on release: `KVCache::copy_slot(last, freed, length)` copies only the cached prefix
- [x] Batched write with `index_put_` at per-row cache indices
- [x] Read as a view: `slice(0, 0, B)` + slice time to `max(pos)+1`
- [x] Unit tests: compaction keeps data intact and leaves later positions untouched; size formula

### Model integration
- [x] `prefill(ids, slot, cache)` (Phase 2)
- [x] `decode(ids[B], pos[B], length, cache)` with per-row key mask `j <= pos[i]` (Phase 2)
- [ ] Compare manual attention vs `scaled_dot_product_attention` (correctness + speed) — moved to Phase 6

### Scheduler
- [x] `Request` with timestamps, `finish_reason`, atomic `cancelled`
- [x] `TokenChannel` (mutex + condvar; token / done / error)
- [x] Bounded `RequestQueue` with `try_push` (false when full)
- [x] `Scheduler` thread: admit (prefill budget) → decode → publish → evict+compact; waits on a condvar when idle
- [x] Stop conditions: EOS, max tokens, context full, cancelled
- [x] Graceful shutdown: fails in-flight requests, drains the queue (works whether or not it was started)
- [x] `SchedulerStats`: submitted/rejected/admitted/finished/cancelled, decode steps, tokens, max batch
- [ ] Policy interface (continuous vs static in one server) — Phase 5, when the server picks one

### Tests
- [x] **100 staggered requests** from 12 client threads, random prompts and lengths → every output identical to its solo run (max batch 6/6, 887 tokens in 135 decode steps = 6.6 tokens/step)
- [x] Requests finishing mid-batch while others join (slots recycled, compaction exercised)
- [x] Cancellation frees the slot mid-flight and leaves the other request's output unchanged
- [x] Queue full → `submit` returns nullptr, counted in stats
- [x] Threading tests under **TSan**: 0 warnings (`.tsan-suppressions` + CI job `tsan-scheduler`)
- [x] Real scheduler thread + model under TSan (single request, cancellation): 0 warnings, run locally
      (not in CI: needs the weights, and TSan + LibTorch is memory hungry)
- [ ] ASan build of the full test suite, clean

**Done when:** 100 staggered random requests match their solo outputs and TSan is clean. ✅

---

## Phase 5 — Server + load generator (1–1.5 weeks)

### Server
- [x] `tools/serve.cpp`: flags `--weights --device --dtype --slots --max-queue --prefill-budget --host --port --threads`
- [x] Thread pool sized `slots + max_queue + 8` via `new_task_queue` (default ~8 would cap streaming concurrency)
- [x] `POST /v1/generate`: `prompt` or `token_ids`, `max_tokens`, `stream`, `stop_on_eos`
- [x] SSE streaming via chunked content provider + `IncrementalDecoder`
- [x] Non-streaming JSON with `text`, `token_ids`, `finish_reason`, timings (queue/TTFT/total)
- [x] Errors: 400 (bad JSON, empty prompt, bad token id, too long), 429 + `Retry-After` (queue full), 503 (shutdown)
- [x] Client disconnect → `cancel()`, checked with `sink.is_writable()` **and** on write failure
- [x] `GET /health`, `GET /stats` (scheduler counters)
- [x] Per-thread tokenizer (`thread_local`), so the scheduler thread only sees token IDs
- [ ] Sampling: `temperature`, `top_k` (on device, seeded); tests stay greedy — deferred to Phase 8
- [ ] Policy switch `--policy continuous|static` for the Phase 7 chart

### Load generator
- [x] `tools/loadgen.cpp`: **open-loop** Poisson arrivals (`--rate --duration --seed`), fires on schedule regardless of outstanding requests
- [x] Prompt length range (`--prompt-min/--prompt-max`), `--max-tokens`, `--stream on|off`
- [x] Sends `token_ids`, so tokenizer cost is excluded from the measurement
- [x] Per-request CSV: `id,prompt_tokens,output_tokens,send_ms,ttft_ms,done_ms,status`
- [x] Summary printout: TTFT p50/p90/p99, TPOT p50/p99, tokens/s, req/s, 429 count
- [x] Does not link the engine — it only speaks HTTP
- [ ] Server-side timing CSV (queue wait vs prefill) — currently returned per request in the JSON only

### Plots
- [x] `scripts/plot_results.py`: `latency` (CDF + TTFT over time), `sweep` (p50/p99 vs rate), `compare` (bar charts)

### Tests (69 total, all pass)
- [x] Health/stats, generate matches solo generation, streaming text == non-streaming text
- [x] 5 kinds of bad request → 400
- [x] Overload → 429 from 8 concurrent clients against 1 slot + 1 queue slot
- [x] Client disconnect cancels the request and frees the slot

**Done when:** a local CPU server + loadgen run produces a CSV with sensible numbers. ✅

---

## Phase 6 — Kaggle GPU + FP16 (1 week)

- [x] `--device cuda`, `--dtype fp16` wired through weights, cache, and mask values (dtype-safe since Phase 1)
- [x] `scripts/kaggle_run.py`: environment → weights → CUDA build → tests → bench (fp32 + fp16) → server + loadgen → `results/`
- [x] Pins the engine to one GPU (`CUDA_VISIBLE_DEVICES=0`); Kaggle hands out two T4s
- [x] Greedy pick stays on the device; one `.to(cpu)` per decode step (since Phase 4)
- [x] Pinned CPU staging buffers for `ids`/`positions`, `non_blocking=true` (allocated once per scheduler)
- [x] `tools/bench.cpp`: prefill vs prompt length, decode vs batch size at 3 cache lengths, `torch::cuda::synchronize()` around every timed region, CSV out
- [x] `tests/test_gpu.cpp`: fp32 vs HF (tight), fp16 top-1 agreement + log-prob drift, cached==uncached on GPU, fp16 greedy drift reported, scheduler on GPU vs solo runs — all skip without CUDA
- [x] **Ran on a Kaggle T4**: CUDA build, 72/74 tests passed, benchmarks + server load CSVs written
- [x] Needed one fix: Rust is absent on Kaggle, and configuring without it cached `CARGO_EXECUTABLE-NOTFOUND`
- [x] Retuned two fp16 tolerances that were too strict (see IMPLEMENTATION.md) and split the scheduler GPU
      test into a strict fp32 one and a near-tie fp16 one
- [x] Re-run on Kaggle: **75/75 pass**; fp16 divergences confirmed as top-2 gaps of 0 and 0.0625
- [x] Fixed a benchmark hazard: a leftover server shared the port via `SO_REUSEPORT` and silently split the load
- [ ] Profile one decode step (kernel launch count, CPU vs GPU time) — the batch-1 fp16 result says it is launch-bound
- [ ] Trim the build: SentencePiece and Abseil are compiled but unused (~10 min of the Kaggle build)

**Done when:** the Kaggle job builds, passes GPU tests, and saves a benchmark CSV.

---

## Phase 7 — Benchmarks (1 week)

Each item is a separate short Kaggle job, 3 repeats, median reported, charts labelled with GPU + vCPU count.

- [ ] Request-rate sweep → p99 TTFT vs rate (find the knee)
- [ ] Static vs continuous batching, same load
- [ ] FP32 vs FP16: tokens/s + peak memory
- [ ] Slots vs throughput (1, 2, 4, 8, 16, 32, 64)
- [ ] Prefill budget vs TTFT/TPOT trade-off (bonus)
- [ ] Baseline in a separate job/venv: vLLM (verify T4 support first) or HF `generate` fallback
- [ ] GPT-2 medium run (bonus, config-driven)
- [ ] Commit CSVs + PNGs to `results/<date>_<gpu>/`

**Done when:** 4–5 clean, labelled charts.

---

## Phase 8 — Stretch (optional)

- [ ] CUDA graphs: bucket `B` and `T_eff`, capture decode per bucket, measure overhead cut
- [ ] Custom CUDA decode-attention kernel (needs nvcc on Kaggle), compare vs LibTorch
- [ ] Paged KV cache (block table + allocator), best combined with the custom kernel
- [ ] Chunked prefill
- [ ] Top-p sampling, repetition penalty

---

## Phase 9 — Polish (3–4 days)

- [ ] README: architecture diagram, build/run (WSL2 + Docker + Kaggle), API, charts
- [ ] "What I learned": each optimisation → measured effect
- [ ] Honest notes: fp16 divergence, T4 limits (no FlashAttention), CPU-bound decode at low batch
- [ ] Dockerfile (CPU, multi-stage with Rust + LibTorch builder), `docker run` instructions
- [ ] 2–3 min demo video: streaming under load (recorded locally on CPU)
- [ ] License, clean up TODOs, tag `v1.0`
