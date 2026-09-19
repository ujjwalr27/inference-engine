"""Turn load-generator CSVs into the Phase 7 charts.

Each CSV comes from `gpt2_loadgen --out <file>.csv` and holds one row per request:
  id,prompt_tokens,output_tokens,send_ms,ttft_ms,done_ms,status

Usage:
  # latency profile of a single run
  python scripts/plot_results.py latency results/run.csv --label "8 slots, fp32"

  # p99 TTFT against request rate (files named like rate_2.csv, rate_4.csv, ...)
  python scripts/plot_results.py sweep results/rate_*.csv --x-from-name

  # compare runs side by side (e.g. static vs continuous, fp32 vs fp16)
  python scripts/plot_results.py compare results/static.csv results/continuous.csv
"""
import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ok"] = df["status"] == 200
    df["busy"] = df["status"] == 429
    # Time per output token, excluding the first (that one is time to first token).
    multi = df["ok"] & (df["output_tokens"] > 1) & (df["ttft_ms"] >= 0)
    df["tpot_ms"] = pd.NA
    df.loc[multi, "tpot_ms"] = (df.loc[multi, "done_ms"] - df.loc[multi, "ttft_ms"]) / (
        df.loc[multi, "output_tokens"] - 1
    )
    return df


def summary(path: Path) -> dict:
    df = load(path)
    ok = df[df["ok"]]
    wall_s = (df["send_ms"].max() + df["done_ms"].max()) / 1000.0 if len(df) else 0.0
    return {
        "name": path.stem,
        "requests": len(df),
        "ok": int(df["ok"].sum()),
        "busy_429": int(df["busy"].sum()),
        "failed": int((~df["ok"] & ~df["busy"]).sum()),
        "ttft_p50": ok["ttft_ms"].quantile(0.50) if len(ok) else float("nan"),
        "ttft_p99": ok["ttft_ms"].quantile(0.99) if len(ok) else float("nan"),
        "tpot_p50": ok["tpot_ms"].dropna().quantile(0.50) if len(ok) else float("nan"),
        "output_tokens": int(ok["output_tokens"].sum()),
        "tokens_per_s": ok["output_tokens"].sum() / wall_s if wall_s > 0 else float("nan"),
    }


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def cmd_latency(args) -> None:
    path = Path(args.files[0])
    df = load(path)
    ok = df[df["ok"]]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))

    for column, label in (("ttft_ms", "time to first token"), ("done_ms", "end to end")):
        values = ok[column].sort_values().to_numpy()
        left.plot(values, np.arange(1, len(values) + 1) / len(values), label=label)
    style(left, f"Latency distribution — {args.label or path.stem}", "milliseconds", "fraction of requests")
    left.legend(frameon=False)

    right.scatter(ok["send_ms"] / 1000.0, ok["ttft_ms"], s=8, alpha=0.6)
    style(right, "Time to first token over the run", "seconds since start", "TTFT (ms)")
    save(fig, Path(args.out or path.with_suffix(".png")))


def cmd_sweep(args) -> None:
    rows = []
    for name in args.files:
        path = Path(name)
        s = summary(path)
        if args.x_from_name:
            match = re.search(r"(\d+(?:\.\d+)?)", path.stem)
            s["x"] = float(match.group(1)) if match else float("nan")
        else:
            s["x"] = s["requests"]
        rows.append(s)
    data = pd.DataFrame(rows).sort_values("x")
    print(data.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(data["x"], data["ttft_p50"], marker="o", label="p50")
    ax.plot(data["x"], data["ttft_p99"], marker="o", label="p99")
    style(ax, args.title or "Time to first token vs request rate", args.xlabel, "TTFT (ms)")
    ax.legend(frameon=False)
    save(fig, Path(args.out or "results/ttft_vs_rate.png"))


def cmd_compare(args) -> None:
    data = pd.DataFrame([summary(Path(f)) for f in args.files])
    print(data.to_string(index=False))

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.bar(data["name"], data["ttft_p99"], color="#4C72B0")
    style(left, "p99 time to first token", "", "milliseconds")
    right.bar(data["name"], data["tokens_per_s"], color="#DD8452")
    style(right, "Output throughput", "", "tokens / second")
    save(fig, Path(args.out or "results/comparison.png"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    for name, handler in (("latency", cmd_latency), ("sweep", cmd_sweep), ("compare", cmd_compare)):
        p = sub.add_parser(name)
        p.add_argument("files", nargs="+")
        p.add_argument("--out")
        p.add_argument("--label")
        p.add_argument("--title")
        p.add_argument("--xlabel", default="requests per second")
        p.add_argument("--x-from-name", action="store_true", help="take the x value from the digits in the filename")
        p.set_defaults(handler=handler)

    args = ap.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
