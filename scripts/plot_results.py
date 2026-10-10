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

  # the README charts from a kaggle_sweep.py (+ kaggle_baseline.py) results directory
  python scripts/plot_results.py readme results/2026-10-05_t4 --rate 32
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
        "requests_per_s": len(ok) / wall_s if wall_s > 0 else float("nan"),
        "tokens_per_s": ok["output_tokens"].sum() / wall_s if wall_s > 0 else float("nan"),
    }


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, out: Path, relayout: bool = True) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if relayout:  # pass False to keep margins a chart reserved itself (e.g. for a two-line footnote)
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
SLOT_3 = "#1baf7a"  # aqua, Hugging Face; under 3:1 on the surface, so its values are always labelled
SLOT_5 = "#e87ba4"  # magenta, vLLM; under 3:1 on the surface, so its lines are always labelled
SLOT_7 = "#4a3aa7"  # violet, the engine with CUDA graphs (and, when measured, everything else on)
PREVIOUS = "#8b8fa3"  # grey, an earlier configuration shown for reference; always labelled directly


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


MS_TICKS = (2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)


def ms_label(ms: float) -> str:
    return f"{ms:g} ms" if ms < 1000 else f"{ms / 1000:g} s"


def log_ms_axis(ax, values, pad_low: float, pad_high: float) -> None:
    low, high = min(values) * pad_low, max(values) * pad_high
    ticks = [t for t in MS_TICKS if low <= t <= high]
    ax.set_yscale("log")
    ax.set_ylim(low, high)
    ax.set_yticks(ticks, [ms_label(t) for t in ticks])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def chart_ttft_vs_rate(results: Path, out: Path, context: str) -> None:
    if not list(results.glob("continuous_rate*.csv")):
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    handles = []
    values = []
    for policy, color in (("continuous", SLOT_1), ("static", SLOT_2)):
        data = rate_points(results, policy)
        if data.empty:
            continue
        (line,) = ax.plot(data["rate"], data["ttft_p50"], color=color, marker="o", label=policy)
        ax.plot(data["rate"], data["ttft_p99"], color=color, linestyle=(0, (4, 3)), marker="o",
                markerfacecolor=SURFACE)
        handles.append(line)
        values += list(data["ttft_p50"]) + list(data["ttft_p99"])
    cont = rate_points(results, "continuous")
    ax.set_xscale("log", base=2)
    rates = sorted(set(cont["rate"]))
    ax.set_xticks(rates, [f"{r:g}" for r in rates])
    log_ms_axis(ax, values, 0.6, 2.5)
    ax.set_xlabel("offered load (requests per second)")
    ax.set_ylabel("time to first token (log scale)")
    ax.set_title("Latency stays flat until capacity, then queueing takes over")

    # Capacity lies between the last rate served without rejections and the first one with them;
    # at the first saturated rate, the completed request rate is the measured capacity itself.
    served = cont[cont["busy_429"] == 0]["rate"]
    saturated = cont[cont["busy_429"] > 0].sort_values("rate")
    reject_note = ""
    if len(served) and len(saturated):
        lo, hi = served.max(), saturated["rate"].iloc[0]
        capacity = saturated["requests_per_s"].iloc[0]
        ax.axvspan(lo, hi, color=GRID, alpha=0.6, zorder=0, linewidth=0)
        ax.text((lo * hi) ** 0.5, ax.get_ylim()[1] * 0.6, f"capacity\n~{capacity:.0f} req/s", ha="center",
                va="top", fontsize=8, color=INK_2)
        reject_note = f" · at {hi:g} req/s the queue fills and requests are rejected"
    style_handles = [plt.Line2D([], [], color=INK_2, marker="o", label="p50"),
                     plt.Line2D([], [], color=INK_2, linestyle=(0, (4, 3)), marker="o",
                                markerfacecolor=SURFACE, label="p99")]
    ax.legend(handles=handles + style_handles, loc="upper left", ncols=2)
    tidy(ax)
    footnote(fig, context + reject_note)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out)


def chart_policy_at_rate(results: Path, rate: float, out: Path, context: str) -> None:
    chart_policy(results / f"continuous_rate{rate:g}.csv", results / f"static_rate{rate:g}.csv", out, context,
                 f"At {rate:g} req/s")


def chart_policy_varied(results: Path, out: Path, context: str) -> None:
    """kaggle_sweep.py section 5: the same comparison with output lengths drawn from a range."""
    cont, stat = results / "varied_continuous.csv", results / "varied_static.csv"
    if not cont.exists() or not stat.exists():
        return
    lengths = pd.concat([load(cont), load(stat)])
    lengths = lengths[lengths["ok"]]["output_tokens"]
    # Described by mean and maximum: before 2026-10-10 the load generator undercounted ~1% of
    # streamed requests, so the minimum in older CSVs is below what was requested.
    span = f"mixed output lengths (mean {lengths.mean():.0f}, up to {lengths.max()} tokens)"
    chart_policy(cont, stat, out, context.replace("64 output tokens", span), "With mixed output lengths")


def chart_policy(cont_path: Path, stat_path: Path, out: Path, context: str, lead: str) -> None:
    if not cont_path.exists() or not stat_path.exists():
        return
    cont = summary(cont_path)
    stat = summary(stat_path)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [2, 1]})

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
    ratio = cont["tokens_per_s"] / stat["tokens_per_s"]
    rest = "at the same throughput" if abs(ratio - 1) < 0.05 else f"and serves {ratio:.2f}x the tokens per second"
    fig.suptitle(f"{lead}, continuous batching cuts first-token latency {reduction:.0%} {rest}",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
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
    tps, slots = data["tokens_per_s"].to_numpy(), data["slots"].to_numpy()
    gain = tps[-1] / tps[0]
    # If the last doubling of slots still bought >25% more throughput, it has not levelled off.
    trend = (f"still rising at {slots[-1]}" if len(tps) > 1 and tps[-1] / tps[-2] > 1.25
             else f"levelling off past {slots[-2]}")
    ax.set_title(f"{slots[-1]} slots serve {gain:.0f}x the tokens of 1 slot, {trend}")
    tidy(ax)
    footnote(fig, context.replace(" · 32 slots", "") + " · saturating load (64 req/s offered)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out)


def chart_decode_step(results: Path, out: Path, context: str) -> None:
    frames = []
    for run in ("fp16", "fp32", "fp16_graphs"):
        path = results / f"bench_{run}.csv"
        if path.exists():
            frames.append(pd.read_csv(path).assign(run=run))
    if not frames:
        return
    bench = pd.concat(frames)
    decode = bench[bench["phase"] == "decode"]
    lengths = sorted(decode["length"].unique())
    fig, axes = plt.subplots(1, len(lengths), figsize=(4 * len(lengths), 4.2), sharey=True)
    top = decode["ms"].max() * 1.1  # shared axis must fit every panel, not just the first
    for ax, length in zip(axes, lengths):
        for run, color, label in (("fp16_graphs", SLOT_7, "fp16 + CUDA graphs"), ("fp16", SLOT_1, "fp16"),
                                  ("fp32", SLOT_2, "fp32")):
            d = decode[(decode["run"] == run) & (decode["length"] == length)].sort_values("batch")
            if not d.empty:
                ax.plot(d["batch"], d["ms"], color=color, marker="o", label=label)
        ax.set_xscale("log", base=2)
        batches = sorted(decode["batch"].unique())
        ax.set_xticks(batches, [str(b) for b in batches])
        ax.set_xlabel("batch size")
        ax.set_title(f"cache length {length}", fontsize=10, fontweight="normal")
        ax.set_ylim(0, top)
        tidy(ax)
    axes[0].set_ylabel("ms per decode step")
    axes[0].legend(loc="upper left")
    short = decode[(decode["run"] == "fp16") & (decode["length"] == lengths[0])].set_index("batch")["ms"]
    graphed = decode[(decode["run"] == "fp16_graphs") & (decode["length"] == lengths[0])].set_index("batch")["ms"]
    if 1 in short.index and 1 in graphed.index:
        title = (f"CUDA graphs cut a decode step from {short[1]:.1f} to {graphed[1]:.1f} ms for one request "
                 f"(fp16, cache length {lengths[0]})")
    elif 1 in short.index and 16 in short.index:
        title = (f"In fp16, a decode step costs {short[1]:.1f} ms for 1 request and {short[16]:.1f} ms for 16 "
                 f"(cache length {lengths[0]})")
    else:
        title = "Decode step time by batch size"
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    footnote(fig, context.split(" · ")[0] + " · gpt2_bench: 20 timed steps after 5 warm-up, device synchronised")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    save(fig, out)


def chart_prefill_batching(results: Path, out: Path, context: str) -> None:
    """gpt2_bench: n prompts prefilled one pass each (read back after each), right-padded into one
    batch (the scheduler's layout), or packed end to end in one row."""
    path = next((results / f"bench_{run}.csv" for run in ("fp16_graphs", "fp16")
                 if (results / f"bench_{run}.csv").exists()), None)
    if path is None:
        return
    bench = pd.read_csv(path)
    phases = {phase: bench[bench["phase"] == phase].set_index("batch")["ms"]
              for phase in ("prefill_solo", "prefill_padded", "prefill_packed")}
    counts = sorted(set.intersection(*(set(v.index) for v in phases.values())))
    if not counts:
        return
    length = int(bench[bench["phase"] == "prefill_padded"]["length"].iloc[0])
    fig, ax = plt.subplots(figsize=(9, 4.4))
    width = 0.27
    # Aqua is under 3:1 on the surface, but every bar carries its value.
    bars = (("prefill_solo", PREVIOUS, "one pass per prompt"), ("prefill_padded", SLOT_1, "one padded batch"),
            ("prefill_packed", SLOT_3, "packed into one row"))
    for j, (phase, color, label) in enumerate(bars):
        xs = [i + (j - 1) * width for i in range(len(counts))]
        values = [phases[phase][n] for n in counts]
        ax.bar(xs, values, width=width, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        for x, v in zip(xs, values):
            ax.text(x, v, f"{v:.0f}" if v >= 100 else f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(counts)), [str(n) for n in counts])
    ax.set_xlabel(f"prompts admitted in the same step ({length} tokens each)")
    ax.set_ylabel("ms until every first token is on the host")
    ax.legend(loc="upper left")
    tidy(ax)
    n = counts[-1]
    solo, padded = phases["prefill_solo"][n], phases["prefill_padded"][n]
    fig.suptitle(f"Prefilling {n} prompts as one padded batch: {padded:.1f} ms instead of {solo:.1f} ms "
                 f"({solo / padded:.1f}x)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    footnote(fig, context.split(" · ")[0] + f" · {bench['dtype'].iloc[0]} · gpt2_bench, device synchronised")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    save(fig, out)


def chart_baseline(results: Path, out: Path, context: str) -> None:
    """Engine vs plain Hugging Face transformers (scripts/kaggle_baseline.py) under the same load."""
    servers = (("engine", "this engine", SLOT_1), ("huggingface", "Hugging Face generate()", SLOT_3))
    points = {}
    for key, _, _ in servers:
        rows = []
        for path in results.glob(f"baseline_{key}_rate*.csv"):
            s = summary(path)
            s["rate"] = float(re.search(r"rate(\d+(?:\.\d+)?)", path.stem).group(1))
            rows.append(s)
        if rows:
            points[key] = pd.DataFrame(rows).set_index("rate").sort_index()
    if len(points) < 2:
        return
    rates = sorted(set(points["engine"].index) & set(points["huggingface"].index))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))

    width = 0.38
    for j, (key, label, color) in enumerate(servers):
        d = points[key].loc[rates]
        x = np.arange(len(rates)) + (j - 0.5) * width
        left.bar(x, d["tokens_per_s"], width=width, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        for xi, v in zip(x, d["tokens_per_s"]):
            left.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
        right.plot(rates, d["ttft_p50"], color=color, marker="o", label=label)
        for r, v in zip(rates, d["ttft_p50"]):
            shown = float(f"{v:.2g}") if v >= 1000 else round(v)
            # The engine sits below Hugging Face everywhere, so its labels go under its line.
            below = key == "engine"
            right.annotate(ms_label(shown), (r, v), xytext=(0, -14 if below else 8), textcoords="offset points",
                           ha="center", va="top" if below else "bottom", fontsize=9, color=INK)
    left.set_xticks(range(len(rates)), [f"{r:g}" for r in rates])
    left.set_xlabel("offered load (requests per second)")
    left.set_ylabel("output tokens per second")
    left.set_title("Throughput")
    left.legend(loc="upper left")

    right.set_xscale("log", base=2)
    right.set_xticks(rates, [f"{r:g}" for r in rates])
    values = list(points["engine"].loc[rates, "ttft_p50"]) + list(points["huggingface"].loc[rates, "ttft_p50"])
    log_ms_axis(right, values, 0.25, 4)
    right.set_xlabel("offered load (requests per second)")
    right.set_ylabel("time to first token, p50 (log scale)")
    right.set_title("Time to first token")
    for ax in (left, right):
        tidy(ax)

    top = rates[-1]
    ratio = points["engine"].loc[top, "tokens_per_s"] / points["huggingface"].loc[top, "tokens_per_s"]
    hf_cap = points["huggingface"]["requests_per_s"].max()
    fig.suptitle(f"At {top:g} req/s the engine serves {ratio:.1f}x the tokens of a plain Hugging Face server, "
                 f"which tops out near {hf_cap:.1f} req/s", x=0.01, ha="left", fontsize=12, fontweight="bold")
    footnote(fig, context.replace(" · 32 slots", "") + " · engine: 32 slots, continuous batching · "
             "Hugging Face: transformers generate(), one request at a time")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save(fig, out)


def baseline_points(results: Path, server: str) -> pd.DataFrame:
    rows = []
    for path in results.glob(f"baseline_{server}_rate*.csv"):
        s = summary(path)
        s["rate"] = float(re.search(r"_rate(\d+(?:\.\d+)?)$", path.stem).group(1))
        rows.append(s)
    return pd.DataFrame(rows).set_index("rate").sort_index() if rows else pd.DataFrame()


def chart_engine_vs_vllm(results: Path, out: Path, context: str) -> None:
    """The engine, eager and at its best, against vLLM (kaggle_baseline.py, one session)."""
    full = not baseline_points(results, "engine_full").empty
    best = "engine_full" if full else "engine_graphs"
    best_label = "engine + CUDA graphs + batched prefill" if full else "engine + CUDA graphs"
    # (key, label, colour, vertical offset of its value label: the two low lines sit close together)
    servers = (("engine", "engine, no graphs", SLOT_1, 0), (best, best_label, SLOT_7, -7),
               ("vllm", "vLLM", SLOT_5, 7))
    points = {key: baseline_points(results, key) for key, *_ in servers}
    if any(points[key].empty for key, *_ in servers):
        return
    rates = sorted(set.intersection(*(set(points[key].index) for key, *_ in servers)))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
    ttft_values = []
    for key, label, color, dy in servers:
        d = points[key].loc[rates]
        for ax, column in ((left, "tpot_p50"), (right, "ttft_p50")):
            ax.plot(rates, d[column], color=color, marker="o", label=label)
        ttft_values += list(d["ttft_p50"])
        # Direct label at the lowest rate, where the lines are furthest apart.
        left.annotate(f"{d['tpot_p50'].iloc[0]:.1f}", (rates[0], d["tpot_p50"].iloc[0]), xytext=(-10, dy),
                      textcoords="offset points", ha="right", va="center", fontsize=9, color=INK)
    for ax in (left, right):
        ax.set_xscale("log", base=2)
        ax.set_xticks(rates, [f"{r:g}" for r in rates])
        ax.set_xlabel("offered load (requests per second)")
        tidy(ax)
    left.set_xlim(rates[0] / 1.6, rates[-1] * 1.2)
    left.set_ylim(0, max(points[key].loc[rates, "tpot_p50"].max() for key, *_ in servers) * 1.15)
    left.set_ylabel("milliseconds")
    left.set_title("Time per output token (p50)")
    left.legend(loc="upper left")
    log_ms_axis(right, ttft_values, 0.6, 2.5)
    right.set_ylabel("milliseconds (log scale)")
    right.set_title("Time to first token (p50)")

    low, top = rates[0], rates[-1]
    eager, ours, vllm = (points[key].loc[low, "tpot_p50"] for key in ("engine", best, "vllm"))
    tps = {key: points[key].loc[top, "tokens_per_s"] for key, *_ in servers}
    if full:
        title = (f"At {top:g} req/s the engine serves {tps[best] / tps['vllm'] - 1:.0%} more tokens/s than vLLM; "
                 f"at {low:g} req/s, {ours:.1f} vs {vllm:.1f} ms per token")
    else:
        title = (f"CUDA graphs cut the engine's time per token {eager / ours:.1f}x at {low:g} req/s: "
                 f"{ours:.1f} ms against vLLM's {vllm:.1f} ms")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    footnote(fig, context.replace(" · 32 slots", "") + " · at most 32 sequences in flight on both servers · "
             "vLLM 0.31 (Triton attention: FlashAttention needs a newer GPU)\n"
             f"Throughput at {top:g} req/s: {tps[best]:.0f} (engine at its best), {tps['vllm']:.0f} (vLLM), "
             f"{tps['engine']:.0f} (engine, no graphs) output tokens/s")
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    save(fig, out, relayout=False)


def chart_batched_prefill(results: Path, out: Path, context: str) -> None:
    """engine_graphs (one prefill pass per request) against engine_full (batched prefill), one session."""
    before, after = baseline_points(results, "engine_graphs"), baseline_points(results, "engine_full")
    if before.empty or after.empty:
        return
    rates = sorted(set(before.index) & set(after.index))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
    series = ((before, "one prefill pass per request", PREVIOUS), (after, "batched prefill", SLOT_7))
    ttft_values = []
    ends = [d.loc[rates[-1], "tokens_per_s"] for d, *_ in series]
    for (d, label, color), end in zip(series, ends):
        d = d.loc[rates]
        left.plot(rates, d["ttft_p99"], color=color, marker="o", label=label)
        right.plot(rates, d["tokens_per_s"], color=color, marker="o", label=label)
        ttft_values += list(d["ttft_p99"])
        # The two lines often end close together: the higher label goes above, the lower below.
        dy = 7 if end >= max(ends) else -7
        right.annotate(f"{end:.0f}", (rates[-1], end), xytext=(8, dy), textcoords="offset points",
                       ha="left", va="center", fontsize=9, color=INK)
    for ax in (left, right):
        ax.set_xscale("log", base=2)
        ax.set_xticks(rates, [f"{r:g}" for r in rates])
        ax.set_xlabel("offered load (requests per second)")
        tidy(ax)
    log_ms_axis(left, ttft_values, 0.6, 2.5)
    left.set_ylabel("milliseconds (log scale)")
    left.set_title("Time to first token (p99)")
    left.legend(loc="upper left")
    right.set_xlim(rates[0] / 1.3, rates[-1] * 1.5)
    right.set_ylim(0, max(before["tokens_per_s"].max(), after["tokens_per_s"].max()) * 1.15)
    right.set_ylabel("output tokens per second")
    right.set_title("Throughput")

    top = rates[-1]
    gain = after.loc[top, "tokens_per_s"] / before.loc[top, "tokens_per_s"]
    def short(ms: float) -> str:
        return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"

    fig.suptitle(f"Batched prefill at {top:g} req/s: p99 first token {short(before.loc[top, 'ttft_p99'])} → "
                 f"{short(after.loc[top, 'ttft_p99'])}, throughput {gain:.2f}x",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    footnote(fig, context + " · both with CUDA graphs (128-token length buckets before, 64 after)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save(fig, out)


def cmd_readme(args) -> None:
    results = Path(args.files[0])
    out_dir = Path(args.out) if args.out else results
    context = args.label or "Tesla T4 · fp16 · 32 slots · prompts 32–256 tokens · 64 output tokens"
    readme_style()
    chart_ttft_vs_rate(results, out_dir / "ttft_vs_rate.png", context)
    chart_policy_at_rate(results, args.rate, out_dir / "continuous_vs_static.png", context)
    chart_policy_varied(results, out_dir / "continuous_vs_static_varied.png", context)
    chart_slots(results, out_dir / "slots_vs_throughput.png", context)
    chart_decode_step(results, out_dir / "decode_step_vs_batch.png", context)
    chart_baseline(results, out_dir / "engine_vs_huggingface.png", context)
    chart_engine_vs_vllm(results, out_dir / "engine_vs_vllm.png", context)
    chart_prefill_batching(results, out_dir / "prefill_batching.png", context)
    chart_batched_prefill(results, out_dir / "batched_prefill.png", context)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    readme = sub.add_parser("readme", help="the README charts from one results directory")
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
