"""Phase 7 benchmark sweep on a Kaggle T4. Assumes scripts/kaggle_run.py has already built.

Runs, in order:
  1. request-rate sweep, continuous batching   -> where latency breaks down
  2. request-rate sweep, static batching       -> the comparison that motivates the project
  3. fp32 vs fp16 at one rate                  -> precision cost
  4. slot-count sweep                          -> how throughput scales with concurrency
then writes the charts with scripts/plot_results.py.

Each configuration gets its own server process, started in its own process group and killed
afterwards: a leftover server would share the port (cpp-httplib sets SO_REUSEPORT) and silently
blend two engines into one measurement.

Usage in a notebook cell:
    %run /kaggle/working/gpt2-engine/scripts/kaggle_sweep.py

Flags: --rates 2,8,16,32,64 --duration 20 --slots 32 --quick --repeat N
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


def sh(cmd, check=True, env=None, quiet=False, tail=3000):
    if not quiet:
        print(f"$ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True, env=env)
    lines = [ln.split("\r")[-1] for ln in (result.stdout + result.stderr).splitlines()]
    output = "\n".join(ln for ln in lines if ln.strip()).strip()
    if output and not quiet:
        print(output[-tail:], flush=True)
    if check and result.returncode != 0:
        raise SystemExit(f"failed ({result.returncode}): {cmd}")
    return result


@contextmanager
def serving(build: Path, weights: Path, env: dict, port: int, dtype: str, slots: int, policy: str,
            max_queue: int = 256):
    if subprocess.run(f"curl -sf http://127.0.0.1:{port}/health", shell=True,
                      capture_output=True).returncode == 0:
        raise SystemExit(f"port {port} is already serving; restart the kernel before measuring")

    cmd = [str(build / "gpt2_serve"), "--weights", str(weights), "--device", "cuda", "--dtype", dtype,
           "--slots", str(slots), "--max-queue", str(max_queue), "--policy", policy, "--port", str(port)]
    print(f"--- server: {dtype}, {slots} slots, {policy} batching")
    server = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              start_new_session=True)
    try:
        for _ in range(120):
            if subprocess.run(f"curl -sf http://127.0.0.1:{port}/health", shell=True,
                              capture_output=True).returncode == 0:
                break
            if server.poll() is not None:
                raise SystemExit(f"server exited early:\n{server.stdout.read()[-2000:]}")
            time.sleep(1)
        else:
            raise SystemExit("server did not come up")
        yield
    finally:
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(server.pid), signal.SIGKILL)
        time.sleep(1)  # let the port close before the next configuration


def run_load(build: Path, env: dict, port: int, rate: float, duration: float, out: Path,
             prompt_min=32, prompt_max=256, max_tokens=64) -> dict:
    sh(f"{build}/gpt2_loadgen --url http://127.0.0.1:{port} --rate {rate} --duration {duration} "
       f"--prompt-min {prompt_min} --prompt-max {prompt_max} --max-tokens {max_tokens} --out {out}", env=env)
    stats = json.loads(sh(f"curl -s http://127.0.0.1:{port}/stats", env=env, quiet=True).stdout)
    sent = sum(1 for _ in open(out)) - 1
    if stats.get("submitted") != sent:
        print(f"WARNING: sent {sent} requests, server counted {stats.get('submitted')} - port may be shared")
    print(f"    max_batch {stats['max_batch']}/{stats['slots']} | decode steps {stats['decode_steps']} | "
          f"tokens {stats['generated_tokens']} | rejected {stats['rejected']}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/kaggle/working/gpt2-engine")
    ap.add_argument("--results", default="/kaggle/working/results")
    ap.add_argument("--rates", default="2,8,16,32,64")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--slot-sweep", default="1,4,8,16,32")
    ap.add_argument("--compare-rate", type=float, default=16.0)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--quick", action="store_true", help="fewer points, 10 s each")
    args = ap.parse_args()

    repo, results = Path(args.repo), Path(args.results)
    build, weights = repo / "build-cuda", repo / "weights"
    if not (build / "gpt2_serve").exists():
        raise SystemExit(f"{build}/gpt2_serve not found - run scripts/kaggle_run.py first")
    results.mkdir(parents=True, exist_ok=True)

    rates = [float(r) for r in args.rates.split(",")]
    slot_counts = [int(s) for s in args.slot_sweep.split(",")]
    duration = args.duration
    if args.quick:
        rates, slot_counts, duration = rates[:3], slot_counts[:3], 10.0

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    port = args.port
    summary = []

    print("\n=== 1. request-rate sweep, continuous batching (fp16) ===")
    for rate in rates:
        with serving(build, weights, env, port, "fp16", args.slots, "continuous"):
            out = results / f"continuous_rate{rate:g}.csv"
            stats = run_load(build, env, port, rate, duration, out)
            summary.append({"kind": "rate", "policy": "continuous", "rate": rate, **stats})

    print("\n=== 2. request-rate sweep, static batching (fp16) ===")
    for rate in rates:
        with serving(build, weights, env, port, "fp16", args.slots, "static"):
            out = results / f"static_rate{rate:g}.csv"
            stats = run_load(build, env, port, rate, duration, out)
            summary.append({"kind": "rate", "policy": "static", "rate": rate, **stats})

    print(f"\n=== 3. fp32 vs fp16 at {args.compare_rate:g} req/s ===")
    for dtype in ("fp32", "fp16"):
        with serving(build, weights, env, port, dtype, args.slots, "continuous"):
            out = results / f"dtype_{dtype}.csv"
            stats = run_load(build, env, port, args.compare_rate, duration, out)
            summary.append({"kind": "dtype", "dtype": dtype, **stats})

    print("\n=== 4. slots vs throughput (fp16, saturating load) ===")
    for slots in slot_counts:
        with serving(build, weights, env, port, "fp16", slots, "continuous"):
            out = results / f"slots_{slots}.csv"
            stats = run_load(build, env, port, max(rates), duration, out)
            summary.append({"kind": "slots", "slots_config": slots, **stats})

    (results / "sweep_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== charts ===")
    plots = repo / "scripts/plot_results.py"
    sh(f"{sys.executable} {plots} sweep {results}/continuous_rate*.csv --x-from-name "
       f"--title 'Continuous batching: TTFT vs request rate' --out {results}/ttft_vs_rate_continuous.png",
       check=False)
    sh(f"{sys.executable} {plots} sweep {results}/static_rate*.csv --x-from-name "
       f"--title 'Static batching: TTFT vs request rate' --out {results}/ttft_vs_rate_static.png", check=False)
    sh(f"{sys.executable} {plots} compare {results}/continuous_rate{args.compare_rate:g}.csv "
       f"{results}/static_rate{args.compare_rate:g}.csv --out {results}/continuous_vs_static.png", check=False)
    sh(f"{sys.executable} {plots} compare {results}/dtype_fp32.csv {results}/dtype_fp16.csv "
       f"--out {results}/fp32_vs_fp16.png", check=False)
    sh(f"{sys.executable} {plots} compare " + " ".join(f"{results}/slots_{s}.csv" for s in slot_counts) +
       f" --out {results}/slots_vs_throughput.png", check=False)

    print("\n=== done ===")
    sh(f"ls -la {results}")


if __name__ == "__main__":
    main()
