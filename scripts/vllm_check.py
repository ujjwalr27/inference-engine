"""Can vLLM serve GPT-2 on this Kaggle T4? Answers that before a vLLM baseline is worth building.

vLLM pins its own torch. Installing it into Kaggle's Python would replace the torch our engine
is built against and links at run time, so it goes into a separate environment
(/kaggle/working/vllm-env, Python 3.12 via uv) and Kaggle's own Python is left untouched.

Checks, in order, stopping at the first failure:
  1. install   vLLM installs and sees the GPU
  2. download  GPT-2 into a local folder; vLLM then loads from that folder with the Hub offline
  3. offline   it loads GPT-2 in fp16 and generates; a rough batch throughput figure
  4. server    `vllm serve` starts and streams a completion for token ids, which is how a
               baseline would drive it (same prompts as our load generator, no re-tokenizing)

Usage in a notebook cell:
    %run /kaggle/working/gpt2-engine/scripts/vllm_check.py

Flags: --venv PATH --port 8299 --model REPO --version X.Y.Z (pin vLLM; default: latest)
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# %run in a notebook reuses modules an earlier cell imported, even after a git pull;
# drop them so this run sees the helpers that sit next to it on disk.
for stale in ("kaggle_sweep",):
    sys.modules.pop(stale, None)
from kaggle_sweep import sh, stop_group, wait_for_free_port  # noqa: E402

OFFLINE_TEST = r"""
import sys, time, torch, vllm
from vllm import LLM, SamplingParams
print(f"vllm {vllm.__version__} | torch {torch.__version__} | gpu {torch.cuda.get_device_name(0)}", flush=True)
llm = LLM(model=sys.argv[1], dtype="float16", gpu_memory_utilization=0.5, max_model_len=1024, seed=0)
greedy = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)

out = llm.generate(["The capital of France is"], greedy)[0]
print("sample:", repr(out.outputs[0].text[:120]), flush=True)

prompts = [{"prompt_token_ids": [15496 + i] * (32 + 7 * i)} for i in range(32)]  # 32-249 tokens
llm.generate(prompts, greedy)  # warm-up
start = time.perf_counter()
outs = llm.generate(prompts, greedy)
elapsed = time.perf_counter() - start
tokens = sum(len(o.outputs[0].token_ids) for o in outs)
print(f"batch: 32 requests x 64 tokens in {elapsed * 1000:.0f} ms -> {tokens / elapsed:.0f} output tok/s", flush=True)
"""


def venv_python(venv: Path) -> Path:
    return venv / "bin" / "python"


def install(venv: Path, version: str) -> None:
    py = venv_python(venv)
    if not py.exists():
        sh(f"{sys.executable} -m pip install -q uv")
        sh(f"{sys.executable} -m uv venv -q --python 3.12 {venv}")
    spec = f"vllm=={version}" if version else "vllm"
    start = time.time()
    sh(f"{sys.executable} -m uv pip install -q --python {py} {spec}", tail=1500)
    print(f"installed in {time.time() - start:.0f} s")
    sh(f"{py} -c \"import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, "
       f"'| cuda', torch.cuda.is_available())\"")


# The full repo id, not the legacy alias "gpt2": recent huggingface_hub fetches through Xet, and
# Xet's read-token request for the alias returned 404 on Kaggle (2026-10-07). Plain HTTP is
# plenty for a 500 MB model and takes that whole failure mode away.
DOWNLOAD = r"""
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(sys.argv[1], allow_patterns=[
    "config.json", "generation_config.json", "model.safetensors",
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"])
print(path)
"""


def download(venv: Path, repo: str) -> Path:
    env = dict(os.environ, HF_HUB_DISABLE_XET="1")
    result = subprocess.run([str(venv_python(venv)), "-c", DOWNLOAD, repo], text=True, capture_output=True, env=env)
    if result.returncode != 0:
        print(result.stderr[-6000:])
        raise SystemExit(f"could not download {repo}")
    path = Path(result.stdout.strip().splitlines()[-1])
    print(f"{repo} -> {path}")
    return path


def offline_env() -> dict:
    # From here on vLLM reads only the local folder; a hidden Hub request would fail loudly.
    return dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def first_error(stderr: str) -> str:
    """vLLM's engine runs in a child process, so its root cause sits well above the final
    traceback. Show the first error-looking line as well as the tail."""
    for line in stderr.splitlines():
        if "Error" in line or "error:" in line:
            return line.strip()
    return ""


def offline(venv: Path, model: Path) -> None:
    result = subprocess.run([str(venv_python(venv)), "-c", OFFLINE_TEST, str(model)], text=True,
                            capture_output=True, env=offline_env())
    lines = [ln for ln in result.stdout.splitlines() if ln.startswith(("vllm ", "sample:", "batch:"))]
    print("\n".join(lines))
    if result.returncode != 0:
        print(result.stderr[-6000:])
        print(f"\nfirst error: {first_error(result.stderr)}")
        raise SystemExit("offline generation failed")


def get(url: str, timeout: float = 5.0) -> int:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status


def server(venv: Path, model: Path, port: int) -> None:
    wait_for_free_port(port)
    log = Path("/kaggle/working/vllm_server.log")
    cmd = [str(venv / "bin" / "vllm"), "serve", str(model), "--served-model-name", "gpt2",
           "--dtype", "float16", "--port", str(port),
           "--gpu-memory-utilization", "0.5", "--max-model-len", "1024"]
    print("$ " + " ".join(cmd))
    with open(log, "w") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
                                env=offline_env())
    try:
        start = time.time()
        while True:  # first start compiles and captures CUDA graphs: minutes, not seconds
            if proc.poll() is not None:
                print(log.read_text()[-4000:])
                raise SystemExit("vllm serve exited during start-up")
            try:
                if get(f"http://127.0.0.1:{port}/health") == 200:
                    break
            except OSError:
                pass
            if time.time() - start > 600:
                print(log.read_text()[-4000:])
                raise SystemExit("vllm serve not healthy after 10 minutes")
            time.sleep(2)
        print(f"healthy after {time.time() - start:.0f} s")

        body = json.dumps({"model": "gpt2", "prompt": [15496, 995], "max_tokens": 16, "temperature": 0,
                           "stream": True, "ignore_eos": True}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        sent = time.perf_counter()
        first, pieces = None, []
        with urllib.request.urlopen(req, timeout=60) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                if first is None:
                    first = time.perf_counter() - sent
                pieces.append(json.loads(line[6:])["choices"][0]["text"])
        total = time.perf_counter() - sent
        print(f"streamed {len(pieces)} chunks for token ids [15496, 995] ('Hello world'): {''.join(pieces)!r}")
        print(f"first chunk {first * 1000:.0f} ms, whole response {total * 1000:.0f} ms")
    finally:
        stop_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            stop_group(proc, signal.SIGKILL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", default="/kaggle/working/vllm-env")
    ap.add_argument("--port", type=int, default=8299)
    ap.add_argument("--model", default="openai-community/gpt2")
    ap.add_argument("--version", default="", help="pin a vLLM version, e.g. one that still supports the T4")
    args = ap.parse_args()
    venv = Path(args.venv)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    print("=== 1. install ===")
    install(venv, args.version)
    print("\n=== 2. download ===")
    model = download(venv, args.model)
    print("\n=== 3. offline generation (fp16) ===")
    offline(venv, model)
    print("\n=== 4. OpenAI-compatible server, streaming ===")
    server(venv, model, args.port)
    print("\n=== verdict: vLLM runs GPT-2 on this GPU; a baseline is feasible ===")


if __name__ == "__main__":
    main()
