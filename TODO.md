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
- [x] CI runs the model tests: a `weights` job exports GPT-2 + reference data once (cached on the two scripts' hash)

### Kaggle spike (main risk)
- [x] Kaggle T4 notebook: torch 2.10.0+cu128, CUDA 12.8, Tesla T4
- [x] Run `scripts/kaggle_spike.py`: ABI=cxx11, nvcc 12.8 present, 4 vCPU, 2× T4, no cargo; CUDA C++ fp16 matmul + SDPA pass
- [x] Paste spike output into IMPLEMENTATION.md §2
- [x] Rust missing on Kaggle → `kaggle_run.py` installs it with `rustup` before configuring

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
- [x] CI: reference tests run on GitHub against weights exported in CI

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
- [x] Policy switch: `--policy continuous|static` (+ `--static-batch`, `--static-wait`), reported in `/stats`,
      with tests that static produces the same answers and that late arrivals wait for the next batch

### Tests
- [x] **100 staggered requests** from 12 client threads, random prompts and lengths → every output identical to its solo run (max batch 6/6, 887 tokens in 135 decode steps = 6.6 tokens/step)
- [x] Requests finishing mid-batch while others join (slots recycled, compaction exercised)
- [x] Cancellation frees the slot mid-flight and leaves the other request's output unchanged
- [x] Queue full → `submit` returns nullptr, counted in stats
- [x] Threading tests under **TSan**: 0 warnings (`.tsan-suppressions` + CI job `tsan-scheduler`)
- [x] Real scheduler thread + model under TSan (single request, cancellation): 0 warnings, run locally
      (not in CI: needs the weights, and TSan + LibTorch is memory hungry)
- [x] ASan + UBSan build of the full test suite, model tests included: clean (79 pass, 9 GPU tests skip; one UBSan finding fixed, a memcpy from an empty vector in a test helper). CI job `asan`

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
- [x] Policy switch `--policy continuous|static` for the Phase 7 chart

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
- [x] Trim the build: SentencePiece, protobuf-lite and Abseil are no longer compiled (only the tokenizers crate
      and its C++ wrapper)

**Done when:** the Kaggle job builds, passes GPU tests, and saves a benchmark CSV.

---

## Phase 7 — Benchmarks (1 week)

`scripts/kaggle_sweep.py` runs all four sweeps in one job, each configuration on its own server
process (started in its own process group, killed afterwards, port checked first).

Re-run after the tokenizer fix (2026-10-05), results in IMPLEMENTATION.md and `results/2026-10-05_t4/`:
capacity ~50 req/s (~3200 tok/s), continuous batching -95% first-token latency vs static, 4.7x Hugging Face.

- [x] Request-rate sweep -> capacity between 32 and 64 req/s (~50 measured)
- [x] Static vs continuous batching, same load (latency gap only; see output lengths below)
- [x] Slots vs throughput (1, 4, 8, 16, 32): 16x, still rising at 32
- [x] FP32 vs FP16 below capacity: same throughput, fp16 24% faster per request
- [x] Fix: false "port may be shared" warning (it was counting connection failures)
- [x] Fix: `plot_results.py` throughput ~30% low (wrong wall-time formula)
- [x] Baseline vs plain Hugging Face on the T4: 4.7x throughput at 8 req/s, 1.4-1.7x cheaper tokens,
      first token in 7-9 ms at every load (vs 16 ms idle and 37 s at 8 req/s)
- [x] It exposed an engine bug: per-thread tokenizer loads (~200 ms each). Fixed with one shared tokenizer
- [x] Correction: the earlier 2.3-2.6x per-token claim was an artifact of that bug (TPOT below one decode step)
- [x] Re-run the full sweep: served throughput 730 -> 3239 tok/s; overload now answers 429, no dropped connections
- [x] Commit CSVs + PNGs to `results/2026-10-05_t4/`, with an engine-vs-Hugging-Face chart; chart titles computed from the data
- [x] Load generator: `--max-tokens-min` for mixed output lengths; `kaggle_sweep.py` section 5 compares the
      policies with 8-128 output tokens (to run on Kaggle)
- [x] Time prefill and decode inside the scheduler (`/stats`): bookkeeping is negligible; prefill is ~34% of a
      saturated run once decode uses CUDA graphs
- [ ] Prefill budget vs TTFT/TPOT trade-off (bonus)
- [x] vLLM baseline (0.31, own venv, T4 via Triton attention): the engine with CUDA graphs matches it on this
      workload (TPOT 2-10% lower, first token 2.7-3.5x sooner, +5% throughput past capacity)
- [ ] GPT-2 medium run (bonus, config-driven)

**Done when:** 4–5 clean, labelled charts.

---

## Phase 8 — Stretch (optional)

- [x] CUDA graphs: bucket `B` and `T_eff`, capture decode per bucket, measure overhead cut
      (decode step 4.8 -> 2.2 ms at batch 1; served TPOT 6.0 -> 2.5 ms; capacity +12%)
- [x] Finer length buckets: 64 by default (`--graph-length-step`), for batch 32 with a 512-token cache, which was
      slower with graphs (7.3 vs 6.3 ms) because 513 padded to 640. To measure on Kaggle (`bench_fp16_graphs*.csv`)
- [x] Batch the prefills admitted in one step: one right-padded batch, one read-back (`--solo-prefill` turns it
      off). Packing prompts into one row was tried first and lost on CPU (quadratic attention over the row:
      2.3x slower than one pass each at 16 prompts, padded 1.05x faster). Matches per-prompt prefill on CPU;
      to measure on Kaggle (`engine_full`, `prefill_*` bench rows)
- [x] CUDA graphs are the default for `gpt2_serve` on CUDA (`--no-cuda-graphs` to turn off). The next Kaggle run
      re-measures eager vs graphs in a second session; revert the default if it does not hold
- [x] Streaming decode re-decoded the whole output per token (O(n^2), under a shared lock): now O(1) per token
- [ ] Custom CUDA decode-attention kernel (needs nvcc on Kaggle), compare vs LibTorch
- [ ] Paged KV cache (block table + allocator), best combined with the custom kernel
- [ ] Chunked prefill
- [ ] Top-p sampling, repetition penalty

---

## Phase 9 — Polish (3–4 days)

- [x] README: architecture diagram, build/run (Docker, Linux/WSL2, Kaggle), API, flags, charts
- [x] "What each optimisation bought": each optimisation → measured effect
- [x] Honest notes: fp16 divergence, T4 limits (no FlashAttention), launch-bound decode at low batch, no paging
- [x] Dockerfile (CPU, four stages: toolchain, build + tests, weights export, runtime), `docker run` instructions
- [ ] 2–3 min demo video: streaming under load (recorded locally on CPU) — Ujjwal
- [x] License (MIT), TODOs cleaned up
- [ ] Tag `v1.0` once the next Kaggle run has measured batched prefill and the 64-token buckets
