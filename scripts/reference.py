"""Produce reference outputs from Hugging Face transformers for the C++ correctness tests.

Writes <out>/reference.safetensors with, for each prompt index i:
  p{i}.input_ids     int64 [T]
  p{i}.positions     int64 [P]      token positions whose logits are stored
  p{i}.logits        float32 [P, V] logits at those positions
  p{i}.greedy        int64 [G]      greedy continuation (short prompts only)
and for prompt 0 only (short, so it stays small):
  p0.hidden.{l}      float32 [T, n_embd]  HF hidden_states[l], l = 0..n_layer
                     (l = 0 embeddings, 1..n_layer-1 block outputs, n_layer = after ln_f)
plus <out>/reference.json with the prompt texts and settings, and
<out>/tokenizer_cases.json with encode/decode cases for the C++ tokenizer.

Usage: python scripts/reference.py [--model gpt2] [--out weights]
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, GPT2LMHeadModel

LONG_SOURCE = (
    "An inference engine turns a trained model into a service. It loads weights once, keeps a cache of "
    "attention keys and values for every active request, and decides at each step which requests run "
    "together. Latency matters to people waiting for the first word; throughput matters to whoever pays "
    "for the hardware. Continuous batching lets new requests join between decode steps instead of "
    "waiting for a whole batch to finish. "
)


def build_prompts(tokenizer) -> list[tuple[str, list[int]]]:
    texts = [
        "The quick brown fox jumps over the lazy dog because",
        "Hello",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
        "Café naïve résumé — 日本語のテキスト 🚀🔥 emojis, accents & symbols: ∑ ∫ √",
        "In 2026, researchers measured time to first token (p50 = 42 ms, p99 = 180 ms) and found that",
    ]
    prompts = [(t, tokenizer.encode(t)) for t in texts]

    long_ids = tokenizer.encode(LONG_SOURCE * 40)
    for n in (500, 1024):
        ids = long_ids[:n]
        assert len(ids) == n, f"long source too short for {n} tokens"
        prompts.append((f"<LONG_SOURCE truncated to {n} tokens>", ids))
    return prompts


TOKENIZER_FRAGMENTS = [
    "the", " the", "The", " inference", " engine", "GPT-2", "  double  spaces", "\ttab", "\n\nnewlines",
    "123", " 3.14159", " -42", "0x1F", "don't", "it's", "can't", "naïve", "café", "Straße", "日本語",
    "中文测试", "Русский текст", "العربية", "🚀", "👨‍👩‍👧‍👦", "🇯🇵", "e\u0301", "é", "—", "…", "«quotes»",
    " def foo(x: int) -> str:", " return [i for i in range(10)]", " // comment", " <div class=\"x\">",
    " https://example.com/path?q=1&r=2", " user@example.com", " C:\\Users\\path\\file.txt", " ", "  ", "",
    " aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", " supercalifragilisticexpialidocious", " 一二三四五六七八九十",
]


def tokenizer_cases(tokenizer, seed: int = 7, n: int = 2000) -> list[dict]:
    import random

    rng = random.Random(seed)
    texts = list(TOKENIZER_FRAGMENTS) + [t for t, _ in build_prompts(tokenizer)[:5]] + [LONG_SOURCE]
    while len(texts) < n:
        k = rng.randint(1, 6)
        texts.append("".join(rng.choice(TOKENIZER_FRAGMENTS) for _ in range(k)))
    cases = []
    for text in texts:
        ids = tokenizer.encode(text)
        cases.append({"text": text, "ids": ids, "decoded": tokenizer.decode(ids)})
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--out", default="weights")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--greedy-tokens", type=int, default=50)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = GPT2LMHeadModel.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    n_ctx = model.config.n_positions

    tensors: dict[str, torch.Tensor] = {}
    meta = {"model": args.model, "threads": args.threads, "attn_implementation": "eager", "prompts": []}

    for i, (text, ids) in enumerate(build_prompts(tokenizer)):
        assert 1 <= len(ids) <= n_ctx
        T = len(ids)
        input_ids = torch.tensor([ids], dtype=torch.int64)
        with torch.inference_mode():
            out = model(input_ids, output_hidden_states=(i == 0))

        positions = sorted({0, T // 2, max(T - 3, 0), max(T - 2, 0), T - 1})
        tensors[f"p{i}.input_ids"] = input_ids[0].clone()
        tensors[f"p{i}.positions"] = torch.tensor(positions, dtype=torch.int64)
        tensors[f"p{i}.logits"] = out.logits[0, positions].contiguous().clone()
        if i == 0:
            for l, h in enumerate(out.hidden_states):
                tensors[f"p0.hidden.{l}"] = h[0].contiguous().clone()

        top = out.logits[0, -1].argmax().item()
        entry = {"index": i, "text": text, "num_tokens": T, "positions": positions, "last_argmax": top}

        # Greedy continuation for short prompts: the Phase 2 target for both the cached and uncached paths.
        if T + args.greedy_tokens <= n_ctx and T <= 64:
            with torch.inference_mode():
                gen = model.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=args.greedy_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=model.config.eos_token_id,
                )
            new_ids = gen[0, T:].contiguous().clone()
            tensors[f"p{i}.greedy"] = new_ids
            entry["greedy_text"] = tokenizer.decode(new_ids.tolist())

        meta["prompts"].append(entry)
        print(f"p{i}: {T:5d} tokens, next-token argmax {top!r} = {tokenizer.decode([top])!r}")

    meta["num_prompts"] = len(meta["prompts"])
    meta["greedy_tokens"] = args.greedy_tokens
    save_file(tensors, str(out_dir / "reference.safetensors"))
    (out_dir / "reference.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    cases = tokenizer_cases(tokenizer)
    (out_dir / "tokenizer_cases.json").write_text(json.dumps(cases, ensure_ascii=False) + "\n")
    print(f"wrote {out_dir / 'reference.safetensors'} and {len(cases)} tokenizer cases")


if __name__ == "__main__":
    main()
