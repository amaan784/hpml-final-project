# HPML metrics: read summary.csv and print per-variant aggregates.

import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
SUMMARY_CSV = RESULTS / "summary.csv"


def _summary_for_variant(variant):
    # `summary_for_variant` lives here so benchmark steps read top-down.
    if not SUMMARY_CSV.exists():
        return {}
    rows = []

    with SUMMARY_CSV.open() as f:
        for row in csv.DictReader(f):
            if row.get("variant") == variant:
                rows.append(row)

    # nothing to aggregate without telemetry rows pulled from CSV
    if not rows:
        return {}
    e2e = [float(r["e2e_ms"]) for r in rows if r.get("e2e_ms")]
    correct = [int(r["correct"]) for r in rows if r.get("correct") in ("0", "1")]
    return {
        "n_scenarios": len(rows),
        "e2e_mean_ms": sum(e2e) / len(e2e) if e2e else 0.0,
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
    }


def main():
    # CLI/helper entry for `main`.
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    args = p.parse_args()

    summary = _summary_for_variant(args.variant)
    print(f"variant={args.variant}  n={summary.get('n_scenarios', 0)}  "
          f"e2e_mean={summary.get('e2e_mean_ms', 0):.0f}ms  "
          f"accuracy={summary.get('accuracy', 0):.2f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

