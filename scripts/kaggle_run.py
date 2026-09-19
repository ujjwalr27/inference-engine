"""Build and run the engine on a Kaggle T4 in one notebook cell.

Paste into a Kaggle notebook (GPU T4, Internet on):

    !git clone -q --recursive https://github.com/<you>/gpt2-engine /kaggle/working/gpt2-engine
    %run /kaggle/working/gpt2-engine/scripts/kaggle_run.py

What it does, in order:
  1. records the environment (torch, CUDA, GPU, CPU count)
  2. installs cmake/ninja if missing and exports the GPT-2 weights + reference data
  3. builds with CUDA against Kaggle's pip torch
  4. runs the test suite (CPU fp32 + GPU fp32/fp16 correctness)
  5. runs micro-benchmarks for fp32 and fp16
  6. runs the server with an open-loop load and saves the CSV

Everything lands in /kaggle/working/results, which Kaggle keeps as job output.

Flags: --skip-tests, --skip-bench, --skip-serve, --repo PATH, --jobs N
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

# Keep the log readable: these scripts otherwise print a progress bar line per weight tensor.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def sh(cmd, check=True, env=None, cwd=None, quiet=False, tail=4000):
    if not quiet:
        print(f"$ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True, env=env, cwd=cwd)
    # Progress bars (git, pip, tqdm) redraw with \r and would otherwise bury the real output.
    lines = [ln.split("\r")[-1] for ln in (result.stdout + result.stderr).splitlines()]
    output = "\n".join(ln for ln in lines if ln.strip()).strip()
    if output and not quiet:
        print(output[-tail:], flush=True)
    if check and result.returncode != 0:
        raise SystemExit(f"failed ({result.returncode}): {cmd}")
    return result


def ensure_rust() -> str:
    """The tokenizer wraps a Rust crate, and Kaggle images ship without cargo.

    This must run BEFORE cmake configures: a configure done without cargo caches
    CARGO_EXECUTABLE-NOTFOUND and the build fails much later, in the middle of the tokenizer.
    """
    cargo_bin = Path.home() / ".cargo" / "bin"
    if shutil.which("cargo") is None and not (cargo_bin / "cargo").exists():
        print("installing Rust (needed by tokenizers-cpp)")
        sh("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
           "sh -s -- -y --profile minimal --default-toolchain stable", tail=1500)
    os.environ["PATH"] = f"{cargo_bin}{os.pathsep}{os.environ['PATH']}"
    cargo = shutil.which("cargo")
    if cargo is None:
        raise SystemExit("cargo still not on PATH after installing Rust")
    sh(f"{cargo} --version", tail=200)
    return cargo


def environment() -> None:
    print("=== environment ===")
    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
    print("cxx11 abi:", torch._C._GLIBCXX_USE_CXX11_ABI)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
    else:
        raise SystemExit("no GPU: enable the T4 accelerator")
    print("cpu count:", os.cpu_count())
    sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv", check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/kaggle/working/gpt2-engine")
    ap.add_argument("--results", default="/kaggle/working/results")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--skip-serve", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="discard the build directory first")
    ap.add_argument("--rate", type=float, default=8.0)
    ap.add_argument("--duration", type=float, default=30.0)
    args = ap.parse_args()

    repo = Path(args.repo)
    if not (repo / "CMakeLists.txt").exists():
        raise SystemExit(f"{repo} does not look like the engine checkout")
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    build = repo / "build-cuda"
    weights = repo / "weights"

    environment()

    print("\n=== dependencies ===")
    if shutil.which("cmake") is None or shutil.which("ninja") is None:
        sh(f"{sys.executable} -m pip install -q cmake ninja")
    sh(f"{sys.executable} -m pip install -q 'transformers>=5' safetensors")

    print("\n=== weights + reference data ===")
    if not (weights / "model.safetensors").exists():
        sh(f"{sys.executable} scripts/export_weights.py", cwd=repo)
    if not (weights / "reference.safetensors").exists():
        sh(f"{sys.executable} scripts/reference.py", cwd=repo)

    print("\n=== build (CUDA) ===")
    ensure_rust()
    # A cache from a run where cargo was missing keeps the not-found path forever.
    cache = build / "CMakeCache.txt"
    if args.fresh or (cache.exists() and "CARGO_EXECUTABLE-NOTFOUND" in cache.read_text(errors="ignore")):
        print(f"discarding stale build directory {build}")
        shutil.rmtree(build, ignore_errors=True)
    env = dict(os.environ, TORCH_CUDA_ARCH_LIST="7.5")  # T4 is sm_75
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    cuda_args = f"-DCMAKE_CUDA_COMPILER={nvcc}" if Path(nvcc).exists() else ""
    sh(
        f"cmake -S {repo} -B {build} -G Ninja -DCMAKE_BUILD_TYPE=Release "
        f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path} {cuda_args}",
        env=env,
    )
    sh(f"cmake --build {build} -j {args.jobs}", env=env)

    # Keep the engine on a single GPU: Kaggle hands out two T4s and benchmarks should be single-device.
    run_env = dict(env, CUDA_VISIBLE_DEVICES="0", GPT2_DATA_DIR=str(weights))

    if not args.skip_tests:
        print("\n=== tests ===")
        sh(f"ctest --test-dir {build} --output-on-failure", env=run_env, check=False)
        sh(f"{build}/gpt2_tests --gtest_filter='GpuTest.*'", env=run_env, check=False)

    if not args.skip_bench:
        print("\n=== micro-benchmarks ===")
        for dtype in ("fp32", "fp16"):
            sh(
                f"{build}/gpt2_bench --weights {weights} --device cuda --dtype {dtype} "
                f"--slots 32 --out {results}/bench_{dtype}.csv",
                env=run_env,
            )

    if not args.skip_serve:
        print("\n=== server + open-loop load ===")
        for dtype in ("fp16",):
            server = subprocess.Popen(
                f"{build}/gpt2_serve --weights {weights} --device cuda --dtype {dtype} "
                f"--slots 32 --max-queue 128 --port 8099",
                shell=True, env=run_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            try:
                for _ in range(120):
                    if subprocess.run("curl -sf http://127.0.0.1:8099/health", shell=True).returncode == 0:
                        break
                    time.sleep(1)
                else:
                    raise SystemExit("server did not come up")
                sh(
                    f"{build}/gpt2_loadgen --url http://127.0.0.1:8099 --rate {args.rate} "
                    f"--duration {args.duration} --prompt-min 32 --prompt-max 256 --max-tokens 64 "
                    f"--out {results}/serve_{dtype}_rate{args.rate:g}.csv",
                    env=run_env,
                )
                sh("curl -s http://127.0.0.1:8099/stats", env=run_env)
            finally:
                server.terminate()
                server.wait(timeout=30)

    print("\n=== done ===")
    sh(f"ls -la {results}")


if __name__ == "__main__":
    main()
