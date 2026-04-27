"""Summarize one or more benchmark CSVs from benchmark.py.

Prints a side-by-side comparison of the labels (e.g. L0_fp16 vs L1_awq vs L2_tuned)
using mean/p50/p95 for TTFT, E2E latency, decode time, and output tokens-per-second.

Run:
    python vlm_serving/summarize_runs.py benchmarks/*.csv
    python vlm_serving/summarize_runs.py benchmarks/L0_fp16.csv benchmarks/L1_awq.csv

Optional --markdown prints a GitHub-flavored table instead of the fixed-width one.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path


METRICS = [
    ("ttft_s", "TTFT (s)", "lower"),
    ("e2e_s", "E2E (s)", "lower"),
    ("decode_s", "Decode (s)", "lower"),
    ("decode_tps", "Decode tok/s", "higher"),
]


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            for k in ("ttft_s", "e2e_s", "decode_s", "decode_tps",
                      "prompt_tokens", "completion_tokens"):
                v = r.get(k, "")
                if v == "" or v is None:
                    r[k] = None
                else:
                    try:
                        r[k] = float(v)
                    except ValueError:
                        r[k] = None
            rows.append(r)
        return rows


def stats(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"mean": None, "p50": None, "p95": None, "n": 0}
    vs = sorted(vals)
    n = len(vs)
    return {
        "mean": statistics.fmean(vs),
        "p50": statistics.median(vs),
        "p95": vs[min(n - 1, max(0, int(round(0.95 * (n - 1)))))],
        "n": n,
    }


def pct_change(a, b, direction: str) -> str:
    """a = baseline (first label), b = this label.
    Returns e.g. '-42%' (good) or '+15%' (bad) given direction preference."""
    if a in (None, 0) or b is None:
        return "  n/a"
    delta = (b - a) / a * 100
    arrow = "+" if delta >= 0 else ""
    return f"{arrow}{delta:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="CSV files from benchmark.py")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    runs: list[tuple[str, list[dict]]] = []
    for p in args.csvs:
        path = Path(p)
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 1
        rows = load_rows(path)
        label = rows[0].get("label") if rows else path.stem
        runs.append((label, rows))

    if not runs:
        print("no data", file=sys.stderr)
        return 1

    # Overall per-label summary
    print("\n=== Overall summary (all scenarios, all iterations) ===\n")
    headers = ["Metric"] + [r[0] for r in runs] + ([] if len(runs) == 1 else [f"vs {runs[0][0]}"])
    table: list[list[str]] = [headers]
    for key, name, direction in METRICS:
        for agg in ("mean", "p50", "p95"):
            row = [f"{name} {agg}"]
            base = None
            for i, (label, rows) in enumerate(runs):
                s = stats([r[key] for r in rows])
                val = s[agg]
                row.append(f"{val:.3f}" if val is not None else "n/a")
                if i == 0:
                    base = val
            if len(runs) > 1:
                # vs column: show delta for the *last* label so the reader gets
                # the end-to-end improvement at a glance.
                last_rows = runs[-1][1]
                last = stats([r[key] for r in last_rows])[agg]
                row.append(pct_change(base, last, direction))
            table.append(row)

    _print_table(table, markdown=args.markdown)

    # Per-scenario breakdown for the first metric that matters most (e2e mean)
    print("\n=== Per-scenario mean E2E latency (s) ===\n")
    scenarios = sorted({r["scenario_id"] for _, rows in runs for r in rows})
    hdr = ["scenario_id"] + [lbl for lbl, _ in runs]
    rows_out = [hdr]
    for sid in scenarios:
        row = [sid]
        for _, rows in runs:
            s = stats([r["e2e_s"] for r in rows if r["scenario_id"] == sid])
            row.append(f"{s['mean']:.3f}" if s["mean"] is not None else "n/a")
        rows_out.append(row)
    _print_table(rows_out, markdown=args.markdown)

    return 0


def _print_table(rows: list[list[str]], markdown: bool) -> None:
    if markdown:
        print("| " + " | ".join(rows[0]) + " |")
        print("|" + "|".join(["---"] * len(rows[0])) + "|")
        for r in rows[1:]:
            print("| " + " | ".join(r) + " |")
        return
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    for i, r in enumerate(rows):
        line = "  ".join(str(c).ljust(widths[j]) for j, c in enumerate(r))
        print(line)
        if i == 0:
            print("-" * len(line))


if __name__ == "__main__":
    sys.exit(main())
