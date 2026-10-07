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
import csv
import json
import os
import signal
import socket
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
    cmd = [str(build / "gpt2_serve"), "--weights", str(weights), "--device", "cuda", "--dtype", dtype,
           "--slots", str(slots), "--max-queue", str(max_queue), "--policy", policy, "--port", str(port)]
    with serving_cmd(cmd, env, port, f"{dtype}, {slots} slots, {policy} batching"):
        yield


@contextmanager
def serving_cmd(cmd: list, env: dict, port: int, label: str, startup_timeout: int = 120, log: Path = None):
    """Starts any server that answers /health, waits for it, and kills its process group after.

    Server output goes to a log file, never to a pipe: nothing reads a pipe while the server runs,
    so once its ~64 KB buffer filled, the server's next log write blocked and froze it mid-run.
    That froze vLLM (much chattier than the engine) at 32 and 64 req/s, every later request
    hanging until the client's 300 s timeout.
    """
    if subprocess.run(f"curl -sf http://127.0.0.1:{port}/health", shell=True,
                      capture_output=True).returncode == 0:
        raise SystemExit(f"port {port} is already serving; restart the kernel before measuring")
    wait_for_free_port(port)

    if log is None:
        slug = "".join(c if c.isalnum() else "_" for c in label).strip("_")
        log = Path("/kaggle/working/server_logs") / f"{slug}_{int(time.time())}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"--- server: {label} (log: {log})")
    with open(log, "w") as out:
        server = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(startup_timeout):
            if subprocess.run(f"curl -sf http://127.0.0.1:{port}/health", shell=True,
                              capture_output=True).returncode == 0:
                break
            if server.poll() is not None:
                raise SystemExit(f"server exited early:\n{log.read_text(errors='replace')[-2000:]}")
            time.sleep(1)
        else:
            raise SystemExit(f"server did not come up:\n{log.read_text(errors='replace')[-2000:]}")
        yield
    finally:
        stop_group(server, signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            stop_group(server, signal.SIGKILL)
            server.wait()
        text = log.read_text(errors="replace")
        if "Traceback" in text or " ERROR " in text:
            print(f"    server log reports errors (full log: {log}):\n{text[-1500:]}")


def stop_group(server: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(os.getpgid(server.pid), sig)
    except ProcessLookupError:
        pass  # it already exited, e.g. because it failed to start


def wait_for_free_port(port: int, timeout: float = 90.0) -> None:
    """Nothing answering /health is not enough: a port can refuse a new bind for a while after its
    previous server exits. Wait until a socket can actually bind it, as the next server will."""
    deadline = time.time() + timeout
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
                return
            except OSError:
                if time.time() > deadline:
                    raise SystemExit(f"port {port} still cannot be bound after {timeout:.0f} s")
                time.sleep(1)


def server_stats(port: int, env: dict) -> dict:
    return json.loads(sh(f"curl -s http://127.0.0.1:{port}/stats", env=env, quiet=True).stdout)


def run_load(build: Path, env: dict, port: int, rate: float, duration: float, out: Path,
             prompt_min=32, prompt_max=256, max_tokens=64, api: str = "engine") -> dict:
    load = (f"{build}/gpt2_loadgen --url http://127.0.0.1:{port} --rate {rate} --duration {duration} "
            f"--prompt-min {prompt_min} --prompt-max {prompt_max} --max-tokens {max_tokens} --out {out}")
    if api != "engine":  # an OpenAI-compatible server (vLLM) has no /stats to cross-check against
        sh(f"{load} --api {api}", env=env)
        with open(out) as f:
            rows = list(csv.DictReader(f))
        transport_failures = sum(1 for r in rows if r["status"] == "0")
        print(f"    connection failures {transport_failures}")
        return {"transport_failures": transport_failures}

    # Count only what this run submits: a warm-up request beforehand is not part of the load.
    before = server_stats(port, env).get("submitted", 0)
    sh(load, env=env)
    stats = server_stats(port, env)
    submitted = stats.get("submitted", 0) - before
    with open(out) as f:
        rows = list(csv.DictReader(f))
    # Status 0 means the connection itself failed, so the server never saw that request.
    # Only requests that got an HTTP answer should match the server's count.
    transport_failures = sum(1 for r in rows if r["status"] == "0")
    reached = len(rows) - transport_failures
    if submitted != reached:
        print(f"WARNING: {reached} requests got an HTTP answer but the server counted {submitted} "
              "during this run - another process may be sharing the port")
    print(f"    max_batch {stats['max_batch']}/{stats['slots']} | decode steps {stats['decode_steps']} | "
          f"tokens {stats['generated_tokens']} | rejected {stats['rejected']} | "
          f"connection failures {transport_failures}")
    if stats.get("decode_steps") and "decode_ms" in stats:
        # Where the scheduler's time went (cumulative since the server started, warm-up included).
        other = stats["busy_ms"] - stats["prefill_ms"] - stats["decode_ms"]
        print(f"    time: decode {stats['decode_ms'] / stats['decode_steps']:.2f} ms/step "
              f"({stats['decode_ms'] / 1000:.1f} s) | prefill {stats['prefill_ms'] / 1000:.1f} s for "
              f"{stats['prefill_tokens']} tokens | bookkeeping {other / 1000:.1f} s")
    return {**stats, "transport_failures": transport_failures}


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
    # The policy comparison is only meaningful below capacity, where both policies keep up and
    # the difference is latency; saturated, it is mostly queueing. Use the highest rate that
    # served every request under both policies.
    below = [s["rate"] for s in summary if s["kind"] == "rate" and s.get("rejected", 0) == 0
             and s.get("transport_failures", 0) == 0]
    compare_rate = max((r for r in below if sum(x == r for x in below) == 2), default=rates[0])
    sh(f"{sys.executable} {repo}/scripts/plot_results.py readme {results} --rate {compare_rate:g}", check=False)

    print("\n=== done ===")
    sh(f"ls -la {results}")


if __name__ == "__main__":
    main()
