"""Our engine vs plain Hugging Face transformers, on the same T4, under the same load.

Both servers speak the same HTTP API, so the same gpt2_loadgen drives both with identical
prompts (same seed), rates and output lengths, and both are measured the same way. Each
server gets one warm-up request before measuring, so neither pays CUDA start-up inside a run.

Assumes scripts/kaggle_run.py has already built the engine. Usage in a notebook cell:
    %run /kaggle/working/gpt2-engine/scripts/kaggle_baseline.py

Flags: --rates 1,2,4,8 --duration 20 --servers engine,huggingface
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_sweep import run_load, serving_cmd, sh  # noqa: E402


def warm_up(port: int, env: dict) -> None:
    body = json.dumps({"token_ids": [15496, 995], "max_tokens": 8})
    sh(f"curl -s -X POST http://127.0.0.1:{port}/v1/generate -H 'Content-Type: application/json' -d '{body}'",
       env=env, quiet=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/kaggle/working/gpt2-engine")
    ap.add_argument("--results", default="/kaggle/working/results")
    ap.add_argument("--rates", default="1,2,4,8")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--port", type=int, default=8099)
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
    ports = {"engine": args.port, "huggingface": args.port + 100}
    servers = {
        "engine": [str(build / "gpt2_serve"), "--weights", str(weights), "--device", "cuda", "--dtype", "fp16",
                   "--slots", str(args.slots), "--max-queue", "256", "--port", str(ports["engine"])],
        "huggingface": [sys.executable, str(repo / "scripts/hf_server.py"), "--dtype", "fp16",
                        "--port", str(ports["huggingface"]), "--max-queue", "256"],
    }
    selected = [s.strip() for s in args.servers.split(",") if s.strip()]
    unknown = set(selected) - set(servers)
    if unknown:
        raise SystemExit(f"unknown server(s): {', '.join(sorted(unknown))}")

    summary = []
    for name in selected:
        print(f"\n=== {name} ===")
        port = ports[name]
        for rate in rates:
            with serving_cmd(servers[name], env, port, f"{name}, fp16"):
                warm_up(port, env)
                out = results / f"baseline_{name}_rate{rate:g}.csv"
                stats = run_load(build, env, port, rate, args.duration, out)
                summary.append({"server": name, "rate": rate, **stats})
    (results / f"baseline_summary_{'_'.join(selected)}.json").write_text(json.dumps(summary, indent=2))

    print("\n=== side by side ===")
    sys.path.insert(0, str(repo / "scripts"))
    from plot_results import summary as csv_summary  # noqa: E402

    print(f"{'rate':>6} | {'server':<12} | {'ok':>9} | {'TTFT p50':>9} | {'TTFT p99':>9} | "
          f"{'TPOT p50':>8} | {'tok/s':>7}")
    for rate in rates:
        for name in servers:
            path = results / f"baseline_{name}_rate{rate:g}.csv"
            if not path.exists():
                continue
            s = csv_summary(path)
            print(f"{rate:>6g} | {name:<12} | {s['ok']:>4}/{s['requests']:<4} | {s['ttft_p50']:>7.0f}ms | "
                  f"{s['ttft_p99']:>7.0f}ms | {s['tpot_p50']:>6.1f}ms | {s['tokens_per_s']:>7.0f}")


if __name__ == "__main__":
    main()
