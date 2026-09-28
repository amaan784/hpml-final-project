"""Single source of truth for every headline number quoted by the project.

Reads ``results/hpml_metrics.csv`` (latency / VRAM / TTFT / ITL / throughput),
``results/llm_judge.csv`` (per-pass judge rows), and optionally
``results/accuracy_canonical.csv`` (output of the multi-pass regeneration),
and prints each number in a sectioned report keyed to the document that
should quote it -- README sections, the Project Report's Table II / Section III.D,
and the headline slides in the .pptx deck.

The README, report .tex, and slide deck should quote ONLY what this script
emits. If a number elsewhere disagrees with this output, the document is
stale -- not the data.

Usage:
  python scripts/report_numbers.py                # print everything
  python scripts/report_numbers.py --section readme
  python scripts/report_numbers.py --section report
  python scripts/report_numbers.py --section slides
  python scripts/report_numbers.py --json         # machine-readable dump
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

HPML_CSV = RESULTS / "hpml_metrics.csv"
JUDGE_CSV = RESULTS / "llm_judge.csv"
ACCURACY_CANONICAL = RESULTS / "accuracy_canonical.csv"

PASS_THRESHOLD = 4  # score >= 4 counts as pass; matches benchmark/llm_judge.py default


def _load_hpml() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not HPML_CSV.exists():
        return out
    with HPML_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row.get("variant")
            if not v:
                continue
            out[v] = {k: _safe_float(row.get(k)) for k in row.keys() if k != "variant"}
            out[v]["_ts"] = row.get("ts", "")  # keep string fields too
    return out


def _safe_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _load_judge() -> dict[str, dict]:
    """Per-variant accuracy aggregation from llm_judge.csv.

    For multi-pass data: per-pass pass rate -> mean +/- std across passes.
    For single-pass legacy data (no ``pass_idx`` column): pass rate over all
    scored rows, with std reported as 0 (cannot estimate variance from one
    sample).
    """
    out: dict[str, dict] = {}
    if not JUDGE_CSV.exists():
        return out
    by_variant_pass: dict[str, dict[str, list[int]]] = {}
    by_variant_n_scenarios: dict[str, set[str]] = {}

    # Last-wins dedupe on (variant, scenario, pass): --force re-grades append
    # without removing prior rows; the newest row is authoritative.
    latest: dict[tuple[str, str, str], dict] = {}
    with JUDGE_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row.get("variant")
            if not v:
                continue
            pidx = row.get("pass_idx") or "0"
            latest[(v, row.get("scenario_id", ""), pidx)] = row

    for (v, sid, pidx), row in latest.items():
        try:
            score = int(row.get("llm_judge_score") or 0)
        except (TypeError, ValueError):
            continue
        if score == 0:
            continue  # judge error -- skip
        judge_pass = int(row.get("llm_judge_pass") or 0)
        by_variant_pass.setdefault(v, {}).setdefault(pidx, []).append(judge_pass)
        by_variant_n_scenarios.setdefault(v, set()).add(sid)

    for v, passes in by_variant_pass.items():
        rates = [sum(p) / len(p) for p in passes.values() if p]
        n_scenarios = len(by_variant_n_scenarios.get(v, set()))
        if not rates:
            continue
        mean = sum(rates) / len(rates)
        var = sum((x - mean) ** 2 for x in rates) / len(rates) if len(rates) > 1 else 0.0
        out[v] = {
            "n_passes": len(rates),
            "n_scenarios": n_scenarios,
            "passes_per_scenario": {p: len(rows) for p, rows in passes.items()},
            "mean_pass_rate": mean,
            "std_pass_rate": math.sqrt(var),
            "total_pass": sum(sum(p) for p in passes.values()),
            "total_scored": sum(len(p) for p in passes.values()),
        }
    return out


def _load_canonical_accuracy() -> dict[str, dict]:
    """Optional accuracy_canonical.csv overrides the on-the-fly aggregation."""
    if not ACCURACY_CANONICAL.exists():
        return {}
    out: dict[str, dict] = {}
    with ACCURACY_CANONICAL.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row.get("variant")
            if not v:
                continue
            out[v] = {
                "mean_pass_rate": _safe_float(row.get("mean_pass_rate")),
                "std_pass_rate": _safe_float(row.get("std_pass_rate")),
                "n_passes": int(_safe_float(row.get("n_passes") or 0)),
                "n_scenarios": int(_safe_float(row.get("n_scenarios") or 0)),
                "total_pass": int(_safe_float(row.get("total_pass") or 0)),
                "total_scored": int(_safe_float(row.get("total_scored") or 0)),
            }
    return out


def _fmt_acc(acc: dict | None) -> str:
    if not acc:
        return "n/a"
    n_scen = acc.get("n_scenarios", 0)
    n_pass = acc.get("n_passes", 0)
    denom = n_scen * n_pass
    pass_n = acc.get("total_pass", 0)
    mu = acc.get("mean_pass_rate", 0.0)
    sd = acc.get("std_pass_rate", 0.0)
    if n_pass > 1:
        return f"{mu:.1%} +/- {sd:.1%}  ({pass_n}/{denom} across {n_pass} passes)"
    return f"{mu:.1%}  ({pass_n}/{denom}, single pass)"


def _fmt_speedup(base: float, opt: float) -> str:
    if base <= 0 or opt <= 0:
        return "n/a"
    return f"{base / opt:.2f}x"


# Section renderers
def _section_readme(hpml: dict, acc: dict) -> str:
    qwen_l0 = hpml.get("L0_baseline", {})
    qwen_l1d = hpml.get("L1_awq_w4a16_domain", {})
    qwen_l1g = hpml.get("L1_awq_w4a16_generic", {})
    lla_l0 = hpml.get("L0_llama_baseline", {})
    lla_l1d = hpml.get("L1_llama_awq_w4a16_domain", {})
    lla_l1g = hpml.get("L1_llama_awq_w4a16_generic", {})

    lines = ["== README Section 3.1 -- Qwen2.5-VL-7B primary track ==", ""]
    lines.append(f"  Baseline (L0):         "
                 f"acc={_fmt_acc(acc.get('L0_baseline'))}  "
                 f"e2e_mean={qwen_l0.get('e2e_mean_ms', 0):.0f} ms  "
                 f"p50={qwen_l0.get('e2e_p50_ms', 0):.0f} ms")
    lines.append(f"  AWQ W4A16 domain (L1d):"
                 f" acc={_fmt_acc(acc.get('L1_awq_w4a16_domain'))}  "
                 f"e2e_mean={qwen_l1d.get('e2e_mean_ms', 0):.0f} ms  "
                 f"p50={qwen_l1d.get('e2e_p50_ms', 0):.0f} ms")
    lines.append(f"  Speedup (mean):        "
                 f"{_fmt_speedup(qwen_l0.get('e2e_mean_ms', 0), qwen_l1d.get('e2e_mean_ms', 0))}")
    lines.append(f"  Speedup (p50):         "
                 f"{_fmt_speedup(qwen_l0.get('e2e_p50_ms', 0), qwen_l1d.get('e2e_p50_ms', 0))}")
    lines.append("")
    lines.append("== README Section 3.2 -- Llama-3-LLaVA-NeXT-8B cross-family ==")
    lines.append("")
    lines.append(f"  Baseline (L0):         "
                 f"acc={_fmt_acc(acc.get('L0_llama_baseline'))}  "
                 f"e2e_mean={lla_l0.get('e2e_mean_ms', 0):.0f} ms  "
                 f"p50={lla_l0.get('e2e_p50_ms', 0):.0f} ms")
    lines.append(f"  AWQ W4A16 generic (L1g):"
                 f" acc={_fmt_acc(acc.get('L1_llama_awq_w4a16_generic'))}  "
                 f"e2e_mean={lla_l1g.get('e2e_mean_ms', 0):.0f} ms  "
                 f"p50={lla_l1g.get('e2e_p50_ms', 0):.0f} ms")
    lines.append(f"  AWQ W4A16 domain (L1d):"
                 f" acc={_fmt_acc(acc.get('L1_llama_awq_w4a16_domain'))}  "
                 f"e2e_mean={lla_l1d.get('e2e_mean_ms', 0):.0f} ms")
    lines.append(f"  Speedup L1g (mean):    "
                 f"{_fmt_speedup(lla_l0.get('e2e_mean_ms', 0), lla_l1g.get('e2e_mean_ms', 0))}")
    lines.append(f"  Speedup L1d (mean):    "
                 f"{_fmt_speedup(lla_l0.get('e2e_mean_ms', 0), lla_l1d.get('e2e_mean_ms', 0))}")
    return "\n".join(lines)


REPORT_VARIANT_ORDER = [
    "L0_baseline", "L1_awq_w4a16_domain", "L1_awq_w4a16_generic",
    "L2_full_bundle", "L3_image_512",
    "L0_llama_baseline", "L1_llama_awq_w4a16_domain", "L1_llama_awq_w4a16_generic",
    "L2_llama_full_bundle", "L3_llama_image_512",
]


def _section_report(hpml: dict, acc: dict) -> str:
    lines = ["== Report Table II -- all 10 variants ==", ""]
    header = (f"  {'variant':32s}  {'e2e_mean':>10}  {'p50':>8}  "
              f"{'iqr':>6}  {'VRAM':>8}  {'TTFT_p50':>9}  {'accuracy':>34}")
    lines.append(header)
    lines.append(f"  {'-'*32}  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*34}")
    for v in REPORT_VARIANT_ORDER:
        h = hpml.get(v, {})
        a = acc.get(v)
        lines.append(
            f"  {v:32s}  "
            f"{h.get('e2e_mean_ms', 0):>8.0f}ms  "
            f"{h.get('e2e_p50_ms', 0):>6.0f}ms  "
            f"{h.get('e2e_iqr_med_ms', 0):>4.0f}ms  "
            f"{h.get('vram_used_mib', 0):>5.0f}MiB  "
            f"{h.get('ttft_ms_p50', 0):>7.0f}ms  "
            f"{_fmt_acc(a):>34}"
        )
    lines.append("")
    lines.append("Notes for Section III.D: denominators = (scenarios * passes); L1g_llama")
    lines.append("may have a smaller scenario count if any rows were missing from")
    lines.append("summary.csv at judge time -- see passes_per_scenario in --json.")
    return "\n".join(lines)


def _section_slides(hpml: dict, acc: dict) -> str:
    qwen_l0 = hpml.get("L0_baseline", {})
    qwen_l1d = hpml.get("L1_awq_w4a16_domain", {})
    qwen_l2 = hpml.get("L2_full_bundle", {})
    lla_l0 = hpml.get("L0_llama_baseline", {})
    lla_l1g = hpml.get("L1_llama_awq_w4a16_generic", {})

    n_scen = 0
    for v in REPORT_VARIANT_ORDER:
        a = acc.get(v)
        if a:
            n_scen = a.get("n_scenarios", n_scen)
            break

    lines = ["== Slides -- headline cells ==", ""]
    lines.append(f"  Scenario count (use on ALL slides; today slide 6 vs 14 disagree)")
    lines.append(f"    scenarios_per_variant = {n_scen}")
    lines.append("")
    lines.append("  Slide 6 -- Qwen L1d headline (labels MUST match metric):")
    lines.append(f"    Qwen L0  e2e_mean = {qwen_l0.get('e2e_mean_ms', 0):.0f} ms"
                 f"   (NOT p50)")
    lines.append(f"    Qwen L1d e2e_mean = {qwen_l1d.get('e2e_mean_ms', 0):.0f} ms"
                 f"   (NOT p50)")
    lines.append(f"    speedup (mean)    = "
                 f"{_fmt_speedup(qwen_l0.get('e2e_mean_ms', 0), qwen_l1d.get('e2e_mean_ms', 0))}")
    lines.append(f"    Qwen L1d accuracy = {_fmt_acc(acc.get('L1_awq_w4a16_domain'))}")
    lines.append("")
    lines.append("  Slide 9 -- latency/accuracy tradeoff:")
    lines.append(f"    Llama L1g e2e_mean = {lla_l1g.get('e2e_mean_ms', 0):.0f} ms")
    lines.append(f"    Llama L1g accuracy = {_fmt_acc(acc.get('L1_llama_awq_w4a16_generic'))}")
    lines.append(f"    Qwen  L2  p50      = {qwen_l2.get('e2e_p50_ms', 0):.0f} ms")
    lines.append(f"    Llama L0  p50      = {lla_l0.get('e2e_p50_ms', 0):.0f} ms")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--section", choices=["readme", "report", "slides", "all"],
                   default="all")
    p.add_argument("--json", action="store_true",
                   help="Machine-readable dump (overrides --section).")
    args = p.parse_args()

    hpml = _load_hpml()
    canonical = _load_canonical_accuracy()
    judge = _load_judge()
    # Canonical takes precedence per-variant; fall back to on-the-fly judge agg.
    acc = {**judge, **canonical}

    if not hpml:
        print(f"WARN: {HPML_CSV} not found or empty.", file=sys.stderr)
    if not acc:
        print(f"WARN: no accuracy data ({JUDGE_CSV} or {ACCURACY_CANONICAL}).",
              file=sys.stderr)

    if args.json:
        json.dump({"hpml": hpml, "accuracy": acc}, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    sections = {
        "readme": lambda: _section_readme(hpml, acc),
        "report": lambda: _section_report(hpml, acc),
        "slides": lambda: _section_slides(hpml, acc),
    }
    order = ["readme", "report", "slides"] if args.section == "all" else [args.section]
    for i, name in enumerate(order):
        if i:
            print()
        print(sections[name]())
    return 0


if __name__ == "__main__":
    sys.exit(main())
