"""Wipe local benchmark results so a fresh run doesn't mix with old rows.

Why: ``benchmark/run_vlm_benchmark.py`` and ``benchmark/hpml_metrics.py``
both open their CSVs in append mode (``"a"``). Re-running the same variant
doubles its rows, contaminating plots and W&B aggregates.

Usage::

    python scripts/clean_results.py                                # dry-run
    python scripts/clean_results.py --yes                          # wipe everything
    python scripts/clean_results.py --yes --variants L0_baseline   # wipe rows for one variant
    python scripts/clean_results.py --yes --plots --tensorboard    # also clear regenerated outputs

Never touched: results/eric/, results/madhav/, results/aman/, results/checkpoint_artifacts/.
W&B server runs are not touched (delete via wandb UI).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

ACCUMULATING_CSVS = [
    RESULTS / "summary.csv",
    RESULTS / "hpml_metrics.csv",
    RESULTS / "react_summary.csv",
    RESULTS / "nfr_react.csv",
    RESULTS / "nfr_plan_execute.csv",
]

ACCUMULATING_OTHERS = [
    RESULTS / "react_traces.json",
    RESULTS / "comparison_table.md",
]

LOG_GLOBS = [
    "bench_*.log",
    "vllm_serve_logs/*.log",
    "nsys_*.qdrep",
    "nsys_*.qdstrm",
]

PLOT_DIRS = [RESULTS / "plots"]
TB_DIR = REPO / "tb"

PRESERVE_DIRS = {
    RESULTS / "eric",
    RESULTS / "madhav",
    RESULTS / "aman",
    RESULTS / "checkpoint_artifacts",
}


def _format_size(path: Path) -> str:
    # walk through `format_size`, kept separate so the main flow stays readable.
    if not path.exists():
        return "-"

    if path.is_file():
        return f"{path.stat().st_size:,} B"
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    n = sum(1 for _ in path.rglob("*"))
    return f"{total:,} B ({n} files)"


def _filter_csv_by_variant(csv_path: Path, variants_to_remove: set[str], dry: bool) -> tuple[int, int]:
    # walk through `filter_csv_by_variant`, kept separate so the main flow stays readable.
    if not csv_path.exists():
        return (0, 0)

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
        fieldnames = reader.fieldnames or []

    if "variant" not in fieldnames:
        return (0, len(all_rows))
    keep = [r for r in all_rows if r.get("variant") not in variants_to_remove]
    removed = len(all_rows) - len(keep)

    if not dry and removed > 0:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            # repeat for every element we need to touch
            for r in keep:
                w.writerow(r)
    return (removed, len(keep))


def _wipe(path: Path, dry: bool) -> bool:
    # does `wipe`, split out so we can reuse it from a few call sites.
    if not path.exists():
        return False

    if path in PRESERVE_DIRS or any(p in path.parents for p in PRESERVE_DIRS):
        print(f"  [preserve] {path}")
        return False

    if dry:
        print(f"  [would delete] {path}  ({_format_size(path)})")
        return True

    if path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)
    print(f"  [deleted]      {path}")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--yes", "-y", action="store_true",
                   help="Actually delete (otherwise dry-run).")
    p.add_argument("--variants", default=None,
                   help="Comma-separated variant names. Only removes ROWS for "
                        "these variants from CSVs; leaves other variants' rows.")
    p.add_argument("--plots", action="store_true",
                   help="Also wipe results/plots/.")
    p.add_argument("--tensorboard", action="store_true",
                   help="Also wipe ./tb/ (PyTorch Profiler output).")
    args = p.parse_args()

    dry = not args.yes
    mode_label = "DRY RUN (use --yes to actually delete)" if dry else "DELETING"
    print(f"==> Clean results - {mode_label}")
    print(f"    Repo:    {REPO}")
    print(f"    Results: {RESULTS}")
    print(f"    Preserve: {', '.join(str(p.relative_to(REPO)) for p in PRESERVE_DIRS)}")
    print()

    # only enter this block when the guard passes
    if args.variants:
        targets = {v.strip() for v in args.variants.split(",") if v.strip()}
        print(f"==> Filtering CSV rows for variants: {sorted(targets)}")
        # each pass handles the next item in the sequence
        for csv_path in ACCUMULATING_CSVS:
            removed, kept = _filter_csv_by_variant(csv_path, targets, dry)
            tag = "[would remove]" if dry else "[removed]"
            print(f"  {tag:<14} {removed:>4} rows from {csv_path.name}  ({kept} rows kept)")
        print()
        return 0

    print("==> Wiping accumulating CSVs:")
    # each pass handles the next item in the sequence
    for path in ACCUMULATING_CSVS:
        _wipe(path, dry)

    print()
    print("==> Wiping accumulating JSON/markdown:")
    # step through the batch one entry at a time
    for path in ACCUMULATING_OTHERS:
        _wipe(path, dry)

    print()
    print("==> Wiping per-variant logs:")
    n_logs = 0
    # repeat for every element we need to touch
    for pattern in LOG_GLOBS:
        # repeat for every element we need to touch
        for path in RESULTS.glob(pattern):
            _wipe(path, dry)
            n_logs += 1

    if n_logs == 0:
        print("  (no matching logs)")

    if args.plots:
        print()
        print("==> Wiping plots:")
        # each pass handles the next item in the sequence
        for d in PLOT_DIRS:
            _wipe(d, dry)

    if args.tensorboard:
        print()
        print("==> Wiping tensorboard traces:")
        _wipe(TB_DIR, dry)

    print()
    print("==> Reminder: W&B server runs are NOT touched.")
    print("    To clean those: wandb UI -> project page -> bulk-delete runs")
    print()

    if dry:
        print("==> Dry run complete. Re-run with --yes to actually delete.")
    else:
        print("==> Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

