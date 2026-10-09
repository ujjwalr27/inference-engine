"""Our engine vs plain Hugging Face transformers and vLLM, on the same T4, under the same load.

Both servers speak the same HTTP API, so the same gpt2_loadgen drives both with identical
prompts (same seed), rates and output lengths, and both are measured the same way. Each
server gets one warm-up request before measuring, so neither pays CUDA start-up inside a run.

Assumes scripts/kaggle_run.py has already built the engine. Usage in a notebook cell:
    %run /kaggle/working/gpt2-engine/scripts/kaggle_baseline.py

vLLM runs from its own environment (scripts/vllm_check.py creates it) with at most --slots
sequences in flight, the same concurrency budget as the engine, and is driven through its
OpenAI-compatible API with the same token-id prompts.

The engine is measured in up to three configurations, so each optimisation is compared within
one session:
  engine          no CUDA graphs, one prefill pass per request (the Phase 7 engine)
  engine_graphs   + CUDA graphs at 128-token length buckets (2026-10-08)
  engine_full     the defaults: CUDA graphs at 64-token buckets + batched prefill

Flags: --rates 1,2,4,8 --duration 20 --servers engine,engine_graphs,engine_full,huggingface,vllm
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# %run in a notebook reuses modules an earlier cell imported, even after a git pull;
# drop them so this run sees the helpers that sit next to it on disk.
for stale in ("kaggle_sweep", "vllm_check", "plot_results"):
    sys.modules.pop(stale, None)
from kaggle_sweep import run_load, serving_cmd, sh  # noqa: E402
import vllm_check  # noqa: E402

# Which HTTP API each server speaks (gpt2_loadgen --api).
APIS = {"engine": "engine", "engine_graphs": "engine", "engine_full": "engine", "huggingface": "engine",
        "vllm": "openai"}


def warm_up(port: int, env: dict, api: str) -> None:
    if api == "openai":
        path, body = "/v1/completions", {"model": "gpt2", "prompt": [15496, 995], "max_tokens": 8}
    else:
        path, body = "/v1/generate", {"token_ids": [15496, 995], "max_tokens": 8}
    sh(f"curl -s -X POST http://127.0.0.1:{port}{path} -H 'Content-Type: application/json' "
       f"-d '{json.dumps(body)}'", env=env, quiet=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/kaggle/working/gpt2-engine")
    ap.add_argument("--results", default="/kaggle/working/results")
    ap.add_argument("--rates", default="1,2,4,8")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--vllm-env", default="/kaggle/working/vllm-env")
    ap.add_argument("--vllm-model", default="openai-community/gpt2")
    ap.add_argument("--servers", default="engine,huggingface",
                    help="which servers to (re)measure; the side-by-side table reads every CSV present")
    args = ap.parse_args()

    repo, results = Path(args.repo), Path(args.results)
    build, weights = repo / "build-cuda", repo / "weights"
    if not (build / "gpt2_serve").exists():
        raise SystemExit(f"{build}/gpt2_serve not found - run scripts/kaggle_run.py first")
    results.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    rates = [float(r) for r in args.rates.split(",")]

    # Each server gets its own port. Rebinding one port straight after a different server released
    # it failed on Kaggle with "Address already in use", even though nothing answered on it any more.
    ports = {"engine": args.port, "huggingface": args.port + 100, "vllm": args.port + 200,
             "engine_graphs": args.port + 300, "engine_full": args.port + 400}
    engine = [str(build / "gpt2_serve"), "--weights", str(weights), "--device", "cuda", "--dtype", "fp16",
              "--slots", str(args.slots), "--max-queue", "256"]
    servers = {
        "engine": engine + ["--port", str(ports["engine"]), "--no-cuda-graphs", "--solo-prefill"],
        "engine_graphs": engine + ["--port", str(ports["engine_graphs"]), "--cuda-graphs",
                                   "--graph-length-step", "128", "--solo-prefill"],
        "engine_full": engine + ["--port", str(ports["engine_full"])],
        "huggingface": [sys.executable, str(repo / "scripts/hf_server.py"), "--dtype", "fp16",
                        "--port", str(ports["huggingface"]), "--max-queue", "256"],
    }
    selected = [s.strip() for s in args.servers.split(",") if s.strip()]
    unknown = set(selected) - set(servers) - {"vllm"}
    if unknown:
        raise SystemExit(f"unknown server(s): {', '.join(sorted(unknown))}")
    envs = {name: env for name in servers}
    startup = {"engine": 120, "engine_graphs": 120, "engine_full": 120, "huggingface": 120, "vllm": 600}  # vLLM compiles on start-up
    if "vllm" in selected:
        venv = Path(args.vllm_env)
        if not (venv / "bin" / "vllm").exists():
            vllm_check.install(venv, "")
        model = vllm_check.download(venv, args.vllm_model)
        servers["vllm"] = [str(venv / "bin" / "vllm"), "serve", str(model), "--served-model-name", "gpt2",
                           "--dtype", "float16", "--port", str(ports["vllm"]), "--max-model-len", "1024",
                           "--gpu-memory-utilization", "0.5", "--max-num-seqs", str(args.slots)]
        envs["vllm"] = dict(vllm_check.offline_env(), CUDA_VISIBLE_DEVICES="0")

    summary = []
    for name in selected:
        print(f"\n=== {name} ===")
        port = ports[name]
        for rate in rates:
            log = results / "logs" / f"baseline_{name}_rate{rate:g}.log"
            with serving_cmd(servers[name], envs[name], port, f"{name}, fp16", startup[name], log):
                warm_up(port, env, APIS[name])
                out = results / f"baseline_{name}_rate{rate:g}.csv"
                stats = run_load(build, env, port, rate, args.duration, out, api=APIS[name])
                summary.append({"server": name, "rate": rate, **stats})
    (results / f"baseline_summary_{'_'.join(selected)}.json").write_text(json.dumps(summary, indent=2))

    print("\n=== side by side ===")
    sys.path.insert(0, str(repo / "scripts"))
    from plot_results import summary as csv_summary  # noqa: E402

    print(f"{'rate':>6} | {'server':<12} | {'ok':>9} | {'TTFT p50':>9} | {'TTFT p99':>9} | "
          f"{'TPOT p50':>8} | {'tok/s':>7}")
    for rate in rates:
        for name in APIS:
            path = results / f"baseline_{name}_rate{rate:g}.csv"
            if not path.exists():
                continue
            s = csv_summary(path)
            print(f"{rate:>6g} | {name:<12} | {s['ok']:>4}/{s['requests']:<4} | {s['ttft_p50']:>7.0f}ms | "
                  f"{s['ttft_p99']:>7.0f}ms | {s['tpot_p50']:>6.1f}ms | {s['tokens_per_s']:>7.0f}")


if __name__ == "__main__":
    main()
