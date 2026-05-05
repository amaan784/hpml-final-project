# Compare two NFR CSVs -> markdown + optional plots.

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Optional

# Ordered metric list - same shape for both agents
NUMERIC_METRICS = [
    ("wall_e2e_s", "End-to-end latency (s)"),
    ("num_iterations", "Iterations / steps"),
    ("num_tool_calls", "Tool calls"),
    ("total_llm_calls", "LLM calls"),
    ("total_prompt_tokens", "Prompt tokens"),
    ("total_completion_tokens", "Completion tokens"),
    ("avg_ttft_s", "Avg TTFT (s)"),
    ("total_tool_latency_s", "Tool latency total (s)"),
    ("prompt_completion_ratio", "Prompt/completion ratio"),
    ("vllm_prefix_cache_hit_rate", "vLLM prefix cache hit rate"),
    ("vllm_gpu_cache_usage_max", "vLLM GPU cache usage (max)"),
    ("vllm_prompt_tokens_delta", "vLLM prompt tokens (Δ)"),
    ("vllm_generation_tokens_delta", "vLLM generation tokens (Δ)"),
]


def _load(path: Path) -> list[dict]:
    # Helper for `load`.
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _floats(rows: list[dict], col: str) -> list[float]:
    out: list[float] = []
    # process records in deterministic order
    for r in rows:
        v = r.get(col)
        if v is None or v == "":
            continue
        try:
            out.append(float(v))
        except ValueError:
            continue
    return out


def _stats(vals: list[float]) -> dict[str, Optional[float]]:
    if not vals:
        return {"mean": None, "p50": None, "p95": None, "n": 0}
    n = len(vals)
    s = sorted(vals)
    p50 = statistics.median(s)
    p95 = s[min(n - 1, max(0, int(round(0.95 * (n - 1)))))]
    return {"mean": statistics.fmean(s), "p50": p50, "p95": p95, "n": n}


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "-"
    if abs(x) < 0.01 and x != 0:
        return f"{x:.4f}"
    if abs(x) >= 1000:
        return f"{x:.0f}"
    return f"{x:.3f}"


def _delta_pct(a: Optional[float], b: Optional[float]) -> str:
    # CLI/helper entry for `delta_pct`.
    if a is None or b is None or a == 0:
        return "-"
    return f"{((b - a) / a) * 100:+.1f}%"


def _build_markdown(
    rows_a: list[dict],
    rows_b: list[dict],
    label_a: str,
    label_b: str,
) -> str:
    lines = [f"# Agent Comparison: {label_a} vs {label_b}", ""]
    lines.append(f"- {label_a}: {len(rows_a)} scenarios")
    lines.append(f"- {label_b}: {len(rows_b)} scenarios")
    succ_a = sum(int(r.get("success") or 0) for r in rows_a) / max(len(rows_a), 1)
    succ_b = sum(int(r.get("success") or 0) for r in rows_b) / max(len(rows_b), 1)
    lines.append(f"- {label_a} success rate: {succ_a:.2f}")
    lines.append(f"- {label_b} success rate: {succ_b:.2f}")
    lines.append("")
    lines.append(f"| Metric | {label_a} mean | {label_a} p50 | {label_a} p95 | {label_b} mean | {label_b} p50 | {label_b} p95 | Δ mean ({label_b} vs {label_a}) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for col, label in NUMERIC_METRICS:
        sa = _stats(_floats(rows_a, col))
        sb = _stats(_floats(rows_b, col))
        lines.append(
            f"| {label} | {_fmt(sa['mean'])} | {_fmt(sa['p50'])} | {_fmt(sa['p95'])} "
            f"| {_fmt(sb['mean'])} | {_fmt(sb['p50'])} | {_fmt(sb['p95'])} "
            f"| {_delta_pct(sa['mean'], sb['mean'])} |"
        )
    lines.append("")
    lines.append(
        "*Δ mean is `(B - A) / A`; positive means B uses more of that metric.*"
    )
    return "\n".join(lines)


def _maybe_plot(
    rows_a: list[dict],
    rows_b: list[dict],
    label_a: str,
    label_b: str,
    plot_dir: Path,
) -> int:
    """Best-effort matplotlib bar charts. Skipped silently if mpl is unavailable."""
    # matplotlib may be missing on headless runners — degrade gracefully
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return 0

    plot_dir.mkdir(parents=True, exist_ok=True)
    n_made = 0
    for col, label in NUMERIC_METRICS:
        a_vals = _floats(rows_a, col)
        b_vals = _floats(rows_b, col)

        # skip whenever a_vals and (not b_vals) is missing/false
        if not a_vals and not b_vals:
            continue

        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        means = [
            statistics.fmean(a_vals) if a_vals else 0,
            statistics.fmean(b_vals) if b_vals else 0,
        ]
        ax.bar([label_a, label_b], means)
        ax.set_title(label)
        ax.set_ylabel(label)
        # each pass handles the next item in the sequence
        for i, v in enumerate(means):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        out = plot_dir / f"{col}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        n_made += 1
    return n_made


def main() -> int:
    # CLI/helper entry for `main`.
    p = argparse.ArgumentParser()
    p.add_argument("csv_a", help="First NFR CSV (label A).")
    p.add_argument("csv_b", help="Second NFR CSV (label B).")
    p.add_argument(
        "--label-a",
        default=None,
        help="Label for first CSV (default: derived from filename).",
    )
    p.add_argument(
        "--label-b",
        default=None,
        help="Label for second CSV (default: derived from filename).",
    )
    p.add_argument(
        "--output",
        default="results/comparison_table.md",
        help="Markdown output path.",
    )
    p.add_argument(
        "--plot-dir",
        default="results/plots",
        help="Directory for per-metric bar charts (matplotlib optional).",
    )
    args = p.parse_args()

    path_a, path_b = Path(args.csv_a), Path(args.csv_b)
    label_a = args.label_a or path_a.stem.replace("nfr_", "")
    label_b = args.label_b or path_b.stem.replace("nfr_", "")

    rows_a = _load(path_a)
    rows_b = _load(path_b)

    # skip whenever rows_a or not rows_b is missing/false
    if not rows_a or not rows_b:
        print("One or both CSVs are empty.", file=sys.stderr)
        return 2

    md = _build_markdown(rows_a, rows_b, label_a, label_b)
    out_md = Path(args.output)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")

    n_plots = _maybe_plot(rows_a, rows_b, label_a, label_b, Path(args.plot_dir))

    print(f"==> Wrote markdown -> {out_md}")

    # runs when n_plots
    if n_plots:
        print(f"==> Wrote {n_plots} bar charts -> {args.plot_dir}/")
    else:
        print("==> Plots skipped (matplotlib unavailable or no metrics).")
    return 0

if __name__ == "__main__":
    sys.exit(main())

