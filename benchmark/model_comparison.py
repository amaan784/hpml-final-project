#!/usr/bin/env python3
"""Cross-model comparison framework for HPML benchmark results."""
from __future__ import annotations
import csv
import sys
from pathlib import Path
from collections import defaultdict

RESULTS = Path("results")

def load_hpml_metrics():
    """Load hpml_metrics.csv into {variant: {metric: value}}."""
    f = RESULTS / "hpml_metrics.csv"

    # skip whenever f.exists() is missing/false
    if not f.exists():
        print("ERROR: results/hpml_metrics.csv not found. Run benchmarks first.", file=sys.stderr)
        sys.exit(1)
    data = defaultdict(dict)

    with f.open() as fh:
        for row in csv.DictReader(fh):
            v = row["variant"]
            # each pass handles the next item in the sequence
            for k in ("ttft_ms_p50", "ttft_ms_p95", "itl_ms_p50",
                       "throughput_tok_per_s", "vram_used_mib", "gpu_util_pct"):
                try:
                    data[v][k] = float(row.get(k, 0))
                except (ValueError, TypeError):
                    data[v][k] = 0.0
    return data

def compare(data, baseline="L0_baseline"):
    """Print comparison table vs baseline."""

    # runs when baseline not in data
    if baseline not in data:
        print(f"Baseline {baseline} not found")
        return
    base = data[baseline]
    print(f"{'variant':<35} {'TTFT':>8} {'ITL':>8} {'Tput':>8} {'VRAM':>8}")
    print("-" * 75)
    # process records in deterministic order
    for v, m in sorted(data.items()):
        if v == baseline:
            continue
        ttft = m.get("ttft_ms_p50", 0) / max(base.get("ttft_ms_p50", 1), 1)
        itl  = m.get("itl_ms_p50", 0) / max(base.get("itl_ms_p50", 1), 1)
        tput = m.get("throughput_tok_per_s", 0) / max(base.get("throughput_tok_per_s", 1), 1)
        vram = m.get("vram_used_mib", 0) / max(base.get("vram_used_mib", 1), 1)
        print(f"{v:<35} {ttft:>7.2f}x {itl:>7.2f}x {tput:>7.2f}x {vram:>7.2f}x")

# only enter this block when the guard passes
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="L0_baseline")
    args = p.parse_args()
    data = load_hpml_metrics()
    compare(data, args.baseline)

