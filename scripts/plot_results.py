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

  # the four README charts from a kaggle_sweep.py results directory
  python scripts/plot_results.py readme results/2026-09-30_t4
"""
import argparse
import re
from pathlib import Path

import matplotlib
import matplotlib.ticker
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
    # Wall time is when the last request finished, measured from the run start. Adding the two
    # column maxima instead would pair the latest send with the slowest request and overstate it.
    finished = df[df["done_ms"] >= 0]
    wall_s = (finished["send_ms"] + finished["done_ms"]).max() / 1000.0 if len(finished) else 0.0
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


# ---------------------------------------------------------------------------------------------
# README charts. Colors are the first two slots of a categorical palette validated for colour
# blindness (worst adjacent CVD delta-E 24.7) and >= 3:1 against the surface. Colour always
# follows the entity: continuous / fp16 are slot 1, static / fp32 slot 2, in every chart.
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SLOT_1, SLOT_2 = "#2a78d6", "#eb6834"


def readme_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": INK_2, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2, "lines.markersize": 7,
        "legend.frameon": False,
    })


def tidy(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(length=0)


def footnote(fig, text: str) -> None:
    fig.text(0.01, 0.01, text, fontsize=8, color=INK_2, ha="left", va="bottom")


def rate_points(results: Path, policy: str) -> pd.DataFrame:
    rows = []
    for path in results.glob(f"{policy}_rate*.csv"):
        s = summary(path)
        s["rate"] = float(re.search(r"rate(\d+(?:\.\d+)?)", path.stem).group(1))
        rows.append(s)
    return pd.DataFrame(rows).sort_values("rate")


def chart_ttft_vs_rate(results: Path, out: Path, context: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    handles = []
    for policy, color in (("continuous", SLOT_1), ("static", SLOT_2)):
        data = rate_points(results, policy)
        if data.empty:
            continue
        (line,) = ax.plot(data["rate"], data["ttft_p50"] / 1000, color=color, marker="o", label=policy)
        ax.plot(data["rate"], data["ttft_p99"] / 1000, color=color, linestyle=(0, (4, 3)), marker="o",
                markerfacecolor=SURFACE)
        handles.append(line)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    rates = sorted(set(rate_points(results, "continuous")["rate"]))
    ax.set_xticks(rates, [f"{r:g}" for r in rates])
    seconds = [0.2, 0.5, 1, 2, 5, 10, 20, 50]
    ax.set_yticks(seconds, [f"{s:g}" for s in seconds])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("offered load (requests per second)")
    ax.set_ylabel("time to first token (s, log scale)")
    ax.set_title("Latency stays flat until capacity, then queueing takes over")
    ax.axvspan(8, 16, color=GRID, alpha=0.6, zorder=0, linewidth=0)
    ax.text(11.3, ax.get_ylim()[1] * 0.6, "capacity\n~11 req/s", ha="center", va="top", fontsize=8, color=INK_2)
    style_handles = [plt.Line2D([], [], color=INK_2, marker="o", label="p50"),
                     plt.Line2D([], [], color=INK_2, linestyle=(0, (4, 3)), marker="o",
                                markerfacecolor=SURFACE, label="p99")]
    ax.legend(handles=handles + style_handles, loc="upper left", ncols=2)
    tidy(ax)
    footnote(fig, context + " · above ~16 req/s the queue fills and requests are rejected")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out)


def chart_policy_at_rate(results: Path, rate: float, out: Path, context: str) -> None:
    cont = summary(results / f"continuous_rate{rate:g}.csv")
    stat = summary(results / f"static_rate{rate:g}.csv")
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4.2), gridspec_kw={"width_ratios": [2, 1]})

    metrics = [("ttft_p50", "p50"), ("ttft_p99", "p99")]
    width = 0.36
    for i, (key, label) in enumerate(metrics):
        for j, (s, color, name) in enumerate(((cont, SLOT_1, "continuous"), (stat, SLOT_2, "static"))):
            x = i + (j - 0.5) * width
            left.bar(x, s[key], width=width, color=color, edgecolor=SURFACE, linewidth=2,
                     label=name if i == 0 else None)
            left.text(x, s[key], f"{s[key]:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
    left.set_xticks(range(len(metrics)), [m[1] for m in metrics])
    left.set_ylabel("milliseconds")
    left.set_title("Time to first token")
    left.legend(loc="upper left")

    for j, (s, color) in enumerate(((cont, SLOT_1), (stat, SLOT_2))):
        right.bar(j, s["tpot_p50"], width=0.6, color=color, edgecolor=SURFACE, linewidth=2)
        right.text(j, s["tpot_p50"], f"{s['tpot_p50']:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    right.set_xticks([0, 1], ["continuous", "static"])
    right.set_ylabel("milliseconds")
    right.set_title("Time per output token (p50)")

    for ax in (left, right):
        tidy(ax)
    reduction = 1 - cont["ttft_p50"] / stat["ttft_p50"]
    fig.suptitle(f"At {rate:g} req/s, continuous batching cuts first-token latency {reduction:.0%} "
                 f"at the same throughput", x=0.01, ha="left", fontsize=12, fontweight="bold")
    footnote(fig, context + f" · throughput {cont['tokens_per_s']:.0f} vs {stat['tokens_per_s']:.0f} output tok/s")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save(fig, out)


def chart_slots(results: Path, out: Path, context: str) -> None:
    rows = []
    for path in results.glob("slots_*.csv"):
        s = summary(path)
        s["slots"] = int(re.search(r"slots_(\d+)", path.stem).group(1))
        rows.append(s)
    if not rows:
        return
    data = pd.DataFrame(rows).sort_values("slots")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(data["slots"], data["tokens_per_s"], color=SLOT_1, marker="o")
    for _, r in data.iterrows():
        ax.annotate(f"{r['tokens_per_s']:.0f}", (r["slots"], r["tokens_per_s"]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(data["slots"]), [str(s) for s in data["slots"]])
    ax.set_ylim(0, data["tokens_per_s"].max() * 1.2)
    ax.set_xlabel("KV-cache slots (concurrent requests)")
    ax.set_ylabel("output tokens per second")
    gain = data["tokens_per_s"].iloc[-1] / data["tokens_per_s"].iloc[0]
    ax.set_title(f"Batching raises throughput {gain:.0f}x, flattening past 16 slots")
    tidy(ax)
    footnote(fig, context.replace(" · 32 slots", "") + " · saturating load (64 req/s offered)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out)


def chart_decode_step(results: Path, out: Path, context: str) -> None:
    frames = []
    for dtype in ("fp16", "fp32"):
        path = results / f"bench_{dtype}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return
    bench = pd.concat(frames)
    decode = bench[bench["phase"] == "decode"]
    lengths = sorted(decode["length"].unique())
    fig, axes = plt.subplots(1, len(lengths), figsize=(4 * len(lengths), 4.2), sharey=True)
    top = decode["ms"].max() * 1.1  # shared axis must fit every panel, not just the first
    for ax, length in zip(axes, lengths):
        for dtype, color in (("fp16", SLOT_1), ("fp32", SLOT_2)):
            d = decode[(decode["dtype"] == dtype) & (decode["length"] == length)].sort_values("batch")
            ax.plot(d["batch"], d["ms"], color=color, marker="o", label=dtype)
        ax.set_xscale("log", base=2)
        batches = sorted(decode["batch"].unique())
        ax.set_xticks(batches, [str(b) for b in batches])
        ax.set_xlabel("batch size")
        ax.set_title(f"cache length {length}", fontsize=10, fontweight="normal")
        ax.set_ylim(0, top)
        tidy(ax)
    axes[0].set_ylabel("ms per decode step")
    axes[0].legend(loc="upper left")
    fig.suptitle("In fp16, one decode step costs ~5 ms for 1 request or for 16",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    footnote(fig, context.split(" · ")[0] + " · gpt2_bench: 20 timed steps after 5 warm-up, device synchronised")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    save(fig, out)


def cmd_readme(args) -> None:
    results = Path(args.files[0])
    out_dir = Path(args.out) if args.out else results
    context = args.label or "Tesla T4 · fp16 · 32 slots · prompts 32–256 tokens · 64 output tokens"
    readme_style()
    chart_ttft_vs_rate(results, out_dir / "ttft_vs_rate.png", context)
    chart_policy_at_rate(results, args.rate, out_dir / "continuous_vs_static.png", context)
    chart_slots(results, out_dir / "slots_vs_throughput.png", context)
    chart_decode_step(results, out_dir / "decode_step_vs_batch.png", context)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    readme = sub.add_parser("readme", help="the four README charts from one results directory")
    readme.add_argument("files", nargs=1, metavar="RESULTS_DIR")
    readme.add_argument("--out")
    readme.add_argument("--label")
    readme.add_argument("--rate", type=float, default=8.0, help="rate for the policy comparison (below capacity)")
    readme.set_defaults(handler=cmd_readme)

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
