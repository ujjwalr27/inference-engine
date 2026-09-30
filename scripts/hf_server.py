"""Baseline: plain Hugging Face transformers behind the engine's own HTTP API.

It speaks exactly what gpt2_serve speaks - POST /v1/generate with token_ids, max_tokens and
stream, answered with the same server-sent events - so the unchanged gpt2_loadgen can drive
either server and the two are measured identically.

What it represents: the usual way a transformers model gets served. Requests run one at a
time through model.generate() (with its KV cache, in fp16 on the GPU); there is no batching
across requests. It is a baseline for "no serving engine", not the best transformers can do.

Usage: python scripts/hf_server.py [--model gpt2] [--port 8099] [--dtype fp16] [--max-queue 256]
                                   [--device cuda]
"""
import argparse
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()


class TokenQueue(BaseStreamer):
    """Receives each generated token from generate() and hands it to the HTTP thread."""

    def __init__(self):
        self.tokens: "queue.Queue[int | None]" = queue.Queue()
        self._prompt_seen = False

    def put(self, value):
        if not self._prompt_seen:  # generate() first passes the whole prompt
            self._prompt_seen = True
            return
        for token in value.reshape(-1).tolist():
            self.tokens.put(int(token))

    def end(self):
        self.tokens.put(None)


class Engine:
    def __init__(self, model_name: str, dtype: torch.dtype, max_queue: int, device: str):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
        # Streamed events carry decoded text, as the engine's do, so both pay for detokenizing.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.eos = self.model.config.eos_token_id
        self.lock = threading.Lock()  # one generate() at a time: no batching across requests
        self.slots = threading.BoundedSemaphore(max_queue + 1)
        self.stats_lock = threading.Lock()
        self.stats = {"submitted": 0, "rejected": 0, "finished": 0, "generated_tokens": 0,
                      "decode_steps": 0, "max_batch": 1, "slots": 1, "max_queue": max_queue,
                      "policy": "huggingface-serial"}

    def bump(self, **counts):
        with self.stats_lock:
            for key, value in counts.items():
                self.stats[key] += value

    def warm_up(self):
        self.generate([15496], 4, TokenQueue(), stop_on_eos=False)

    @torch.inference_mode()
    def generate(self, prompt, max_tokens, streamer, stop_on_eos=True):
        ids = torch.tensor([prompt], device=self.device)
        with self.lock:
            self.model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=self.eos if stop_on_eos else None,
                pad_token_id=self.eos,
                streamer=streamer,
            )


def make_handler(engine: Engine, vocab: int):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # one line per request would swamp the notebook
            pass

        def send_json(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.send_json(200, {"status": "ok"})
            elif self.path == "/stats":
                with engine.stats_lock:
                    self.send_json(200, dict(engine.stats))
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/generate":
                self.send_json(404, {"error": "not found"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                prompt = [int(t) for t in body["token_ids"]]
                max_tokens = int(body.get("max_tokens", 64))
                stream = bool(body.get("stream", False))
                stop_on_eos = bool(body.get("stop_on_eos", True))
                if not prompt or any(t < 0 or t >= vocab for t in prompt) or max_tokens < 1:
                    raise ValueError("bad request")
                if len(prompt) + max_tokens > engine.model.config.n_positions:
                    raise ValueError("prompt + max_tokens exceeds the context")
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                self.send_json(400, {"error": str(e)})
                return

            engine.bump(submitted=1)
            if not engine.slots.acquire(blocking=False):  # same overload rule as the engine
                engine.bump(rejected=1)
                self.send_json(429, {"error": "server busy: request queue is full"})
                return
            try:
                streamer = TokenQueue()
                worker = threading.Thread(target=engine.generate,
                                          args=(prompt, max_tokens, streamer, stop_on_eos))
                worker.start()
                tokens = []
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                while (token := streamer.tokens.get()) is not None:
                    tokens.append(token)
                    if stream:
                        text = engine.tokenizer.decode([token])
                        self.chunk(f"data: {json.dumps({'text': text, 'token_id': token})}\n\n")
                worker.join()
                engine.bump(finished=1, generated_tokens=len(tokens), decode_steps=len(tokens))
                if stream:
                    self.chunk("data: [DONE]\n\n")
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    self.send_json(200, {"token_ids": tokens, "finish_reason": "max_tokens"})
            except (BrokenPipeError, ConnectionResetError):
                pass  # client went away; generate() still finishes, like a server without cancellation
            finally:
                engine.slots.release()

        def chunk(self, text: str):
            data = text.encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--max-queue", type=int, default=256)
    ap.add_argument("--device", default="cuda", help="cuda for measurements; cpu only for smoke tests")
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    engine = Engine(args.model, dtype, args.max_queue, args.device)
    engine.warm_up()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine, engine.model.config.vocab_size))
    server.daemon_threads = True
    print(f"huggingface baseline on http://127.0.0.1:{args.port} | {args.model} {args.dtype} | serial generate()",
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
