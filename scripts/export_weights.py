"""Export Hugging Face GPT-2 weights into a layout the C++ engine can load directly.

Output (default: weights/):
  model.safetensors   float32 tensors, names below
  config.json         model hyperparameters
  tokenizer.json      HF fast tokenizer (used from Phase 2)

Tensor names:
  wte                 [vocab, n_embd]   token embeddings (also the tied LM head)
  wpe                 [n_ctx, n_embd]   position embeddings
  h.{i}.ln_1.{weight,bias}
  h.{i}.attn.{q_proj,k_proj,v_proj,c_proj}.{weight,bias}
  h.{i}.ln_2.{weight,bias}
  h.{i}.mlp.{c_fc,c_proj}.{weight,bias}
  ln_f.{weight,bias}

All linear weights are [out, in] (Conv1D weights transposed), so C++ uses torch::linear directly.

Usage: python scripts/export_weights.py [--model gpt2] [--out weights]
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()  # one progress line per tensor is noise in a job log


def export(model: GPT2LMHeadModel) -> dict[str, torch.Tensor]:
    cfg = model.config
    sd = model.state_dict()
    n_embd = cfg.n_embd
    out: dict[str, torch.Tensor] = {
        "wte": sd["transformer.wte.weight"],
        "wpe": sd["transformer.wpe.weight"],
        "ln_f.weight": sd["transformer.ln_f.weight"],
        "ln_f.bias": sd["transformer.ln_f.bias"],
    }
    for i in range(cfg.n_layer):
        p = f"transformer.h.{i}."
        o = f"h.{i}."
        for ln in ("ln_1", "ln_2"):
            out[o + ln + ".weight"] = sd[p + ln + ".weight"]
            out[o + ln + ".bias"] = sd[p + ln + ".bias"]

        # Conv1D stores weight as [in, out]; torch::linear wants [out, in].
        c_attn_w = sd[p + "attn.c_attn.weight"].t()  # [3*n_embd, n_embd]
        c_attn_b = sd[p + "attn.c_attn.bias"]  # [3*n_embd]
        for name, w, b in zip(
            ("q_proj", "k_proj", "v_proj"),
            c_attn_w.split(n_embd, dim=0),
            c_attn_b.split(n_embd, dim=0),
        ):
            out[o + f"attn.{name}.weight"] = w
            out[o + f"attn.{name}.bias"] = b

        out[o + "attn.c_proj.weight"] = sd[p + "attn.c_proj.weight"].t()
        out[o + "attn.c_proj.bias"] = sd[p + "attn.c_proj.bias"]
        out[o + "mlp.c_fc.weight"] = sd[p + "mlp.c_fc.weight"].t()
        out[o + "mlp.c_fc.bias"] = sd[p + "mlp.c_fc.bias"]
        out[o + "mlp.c_proj.weight"] = sd[p + "mlp.c_proj.weight"].t()
        out[o + "mlp.c_proj.bias"] = sd[p + "mlp.c_proj.bias"]

    return {k: v.detach().to(torch.float32).contiguous().clone() for k, v in out.items()}


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


@torch.inference_mode()
def reference_forward(w: dict[str, torch.Tensor], cfg: dict, ids: torch.Tensor) -> torch.Tensor:
    """Plain-PyTorch GPT-2 over the exported tensors. This is the spec the C++ model follows."""
    B, T = ids.shape
    H, D, eps = cfg["n_head"], cfg["n_embd"], cfg["layer_norm_epsilon"]
    hd = D // H
    x = w["wte"][ids] + w["wpe"][torch.arange(T)]
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    for i in range(cfg["n_layer"]):
        o = f"h.{i}."
        h = F.layer_norm(x, (D,), w[o + "ln_1.weight"], w[o + "ln_1.bias"], eps)
        q, k, v = (
            F.linear(h, w[o + f"attn.{n}.weight"], w[o + f"attn.{n}.bias"]).view(B, T, H, hd).transpose(1, 2)
            for n in ("q_proj", "k_proj", "v_proj")
        )
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        a = (scores.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + F.linear(a, w[o + "attn.c_proj.weight"], w[o + "attn.c_proj.bias"])
        h = F.layer_norm(x, (D,), w[o + "ln_2.weight"], w[o + "ln_2.bias"], eps)
        h = gelu_new(F.linear(h, w[o + "mlp.c_fc.weight"], w[o + "mlp.c_fc.bias"]))
        x = x + F.linear(h, w[o + "mlp.c_proj.weight"], w[o + "mlp.c_proj.bias"])
    x = F.layer_norm(x, (D,), w["ln_f.weight"], w["ln_f.bias"], eps)
    return F.linear(x, w["wte"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--out", default="weights")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model}")
    model = GPT2LMHeadModel.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf = model.config

    if hf.activation_function != "gelu_new":
        raise SystemExit(f"unexpected activation {hf.activation_function}")
    if not torch.equal(model.lm_head.weight, model.transformer.wte.weight):
        raise SystemExit("lm_head is not tied to wte; exporter assumes tied weights")

    tensors = export(model)
    save_file(tensors, str(out_dir / "model.safetensors"), metadata={"source": args.model, "format": "gpt2-engine-v1"})

    cfg = {
        "model": args.model,
        "n_layer": hf.n_layer,
        "n_head": hf.n_head,
        "n_embd": hf.n_embd,
        "n_ctx": hf.n_positions,
        "vocab_size": hf.vocab_size,
        "layer_norm_epsilon": hf.layer_norm_epsilon,
        "eos_token_id": hf.eos_token_id,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    tokenizer.save_pretrained(str(out_dir))
    n_params = sum(t.numel() for t in tensors.values())
    print(f"wrote {len(tensors)} tensors ({n_params / 1e6:.1f}M params) to {out_dir}")

    # Sanity check: the exported tensors, run through the plain-PyTorch spec, must reproduce HF logits.
    ids = torch.tensor([tokenizer.encode("Exported weights should reproduce Hugging Face logits exactly.")])
    with torch.inference_mode():
        expected = model(ids).logits
    got = reference_forward(load_file(str(out_dir / "model.safetensors")), cfg, ids)
    max_diff = (expected - got).abs().max().item()
    print(f"sanity check: max |logit diff| = {max_diff:.3e}")
    if max_diff > 1e-4:
        raise SystemExit("sanity check FAILED")
    print("sanity check passed")


if __name__ == "__main__":
    main()
