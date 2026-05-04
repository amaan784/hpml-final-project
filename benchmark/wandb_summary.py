"""Cross-variant WandB summary run.

Reads per-scenario CSV + per-variant aggregate CSV + LLM-judge CSV +
regenerates the 8 plots, then publishes ONE wandb run with:

  1. ``variants_summary`` Table -- one row per variant. Accuracy comes from
     LLM-judge (results/llm_judge.csv); auto-scorer was retired 2026-05-07.
  2. ``scenarios_side_by_side`` Table -- one row per scenario, columns = variants
     (the artifact teammates use for rubric grading; LLM-judge score+pass per cell)
  3. All 8 matplotlib plots uploaded as ``wandb.Image``
  4. A ``REPORT_TEMPLATE.md`` next to results/plots/ that the team can paste
     into a W&B Report for the rubric's "publishable results" bonus line.

Usage::

    uv run python -m benchmark.wandb_summary
    uv run python -m benchmark.wandb_summary --name "may7-final"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmark import plots as plots_mod  # noqa: E402
from benchmark.wandb_logger import (  # noqa: E402
    comparison_run,
    log_comparison_image,
    log_comparison_table,
)


SUMMARY_COLUMNS = [
    "variant",
    "short",
    "family",
    "model_family",
    "n_scenarios",
    "accuracy",                  # LLM-judge (score >= threshold)
    "mean_score",                # LLM-judge mean 1-5 score
    "e2e_mean_ms",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "ttft_ms_p50",
    "ttft_ms_p95",
    "itl_ms_p50",
    "throughput_tok_per_s",
    "vram_used_mib",
    "gpu_cache_usage_pct",
]


JUDGE_CSV_PATH = REPO / "results" / "llm_judge.csv"


def _load_judge_csv() -> dict[tuple[str, str], dict]:
    """(variant, scenario_id) -> {score, pass, reasoning} from llm_judge.csv."""

    # skip whenever JUDGE_CSV_PATH.exists() is missing/false
    if not JUDGE_CSV_PATH.exists():
        return {}
    out: dict[tuple[str, str], dict] = {}
    import csv as _csv

    with JUDGE_CSV_PATH.open(encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            try:
                score = int(r.get("llm_judge_score") or 0)
            except (ValueError, TypeError):
                continue
            out[(r["variant"], str(r["scenario_id"]))] = {
                "score": score,
                "pass": int(r.get("llm_judge_pass") or 0),
                "reasoning": r.get("llm_judge_reasoning", ""),
            }
    return out


def _to_float(x, default: float | None = None):
    try:
        if x in ("", None):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _build_side_by_side(summary: list[dict]) -> tuple[list[str], list[list[Any]]]:
    """Build a Table where each row is a scenario and each variant is a column.

    Columns: scenario_id, category, tool, then for each variant 4 cells:
      llm_score (1-5), llm_pass (binary), e2e_ms, response.
    Lets teammates eyeball the same scenario across all variants and see
    the LLM-judge verdict for each response.

    Returns (columns, rows).
    """
    judge = _load_judge_csv()

    by_variant_then_scenario: dict[str, dict[str, dict]] = {}
    scenario_meta: dict[str, dict] = {}
    variant_order: list[str] = []
    for r in summary:
        v = r.get("variant", "?")
        sid = str(r.get("scenario_id", "?"))

        # runs when v not in by_variant_then_scenario
        if v not in by_variant_then_scenario:
            by_variant_then_scenario[v] = {}
            variant_order.append(v)
        by_variant_then_scenario[v][sid] = r
        scenario_meta.setdefault(sid, {
            "category": r.get("category", ""),
            "tool": r.get("tool", ""),
        })

    variants_sorted = sorted(variant_order)
    scenario_ids_sorted = sorted(scenario_meta, key=lambda s: int(s) if s.isdigit() else 1_000_000)

    columns: list[str] = ["scenario_id", "category", "tool"]
    # process records in deterministic order
    for v in variants_sorted:
        columns += [
            f"{v}__llm_score",    # 1-5 from LLM-judge
            f"{v}__llm_pass",     # binary from LLM-judge threshold
            f"{v}__e2e_ms",
            f"{v}__response",
        ]

    rows: list[list[Any]] = []
    # each pass handles the next item in the sequence
    for sid in scenario_ids_sorted:
        meta = scenario_meta[sid]
        row: list[Any] = [sid, meta["category"], meta["tool"]]
        for v in variants_sorted:
            cell = by_variant_then_scenario.get(v, {}).get(sid)
            j = judge.get((v, sid))
            if cell is None:
                row += ["", "", "", ""]
            else:
                row += [
                    j["score"] if j else "",
                    j["pass"] if j else "",
                    cell.get("e2e_ms", ""),
                    (cell.get("raw_response", "") or "")[:1500],
                ]
        rows.append(row)
    return columns, rows


REPORT_TEMPLATE = """\
# HPML Team 23 -- Visual Inspection Agent Optimization Study

> Auto-generated template. Paste sections into a W&B Report
> (https://wandb.ai/<entity>/<project>/reports) and embed the linked panels.

## 1. Headline result
- **AWQ INT4 quantization (W4A16):** {qwen_int4_speedup} on Qwen2.5-VL-7B,
  {llama_int4_speedup} on Llama-3-LLaVA-NeXT-8B vs FP16 baseline.
- **Calibration finding:** Domain-calibrated INT4 prevents the runaway-generation
  failure mode that generic-calibration triggers (see ``calibration_compare`` panel).
- **Cross-family:** Qwen and Llama collapse to similar low-latency/low-VRAM points
  at INT4 from very different FP16 starts (see ``family_compare`` panel).

## 2. Embedded panels
Drag these from the W&B Workspace into the Report:

- ``variants_summary`` Table -- per-variant accuracy / latency / VRAM
- ``scenarios_side_by_side`` Table -- the rubric-grading view (1 row per scenario)
- ``pareto`` -- latency-accuracy Pareto across variants
- ``vram_breakdown`` -- weights vs KV cache by variant
- ``calibration_compare`` -- domain vs generic calibration result
- ``ttft_itl`` -- per-variant TTFT/ITL bars
- ``accuracy_per_category`` -- variant x scenario-category heatmap
- ``family_compare`` -- Qwen vs Llama at FP16 + AWQ INT4

## 3. Methodology sections to write
- 22 hand-authored scenarios across four equipment domains
  (5 pump impeller + 6 transformer/substation + 5 turbine blade + 6 motor thermal)
- L0 FP16 / L1 W4A16 (domain + generic) / L2 vLLM serving tuning / L3 image preprocessing
  -- 5 active variants per family (Qwen + Llama), 10 total
- Evaluation: LLM-as-judge against ``characteristic_form`` rubric
  (substring auto-scorer retired May 7 -- see ``benchmark/llm_judge.py``)
- Hardware: GCP L4 (Ada Lovelace, 24 GiB), vLLM 0.19, llmcompressor 0.10

## 4. Limitations to discuss
- N=22 scenarios; one scenario flip is ~4.5 percentage points on accuracy
- Vision tower stays FP16 in all variants -- precision sweep is text-tower-only
- Llama-3.2-11B substituted with Llama-3-LLaVA-NeXT-8B for L4 VRAM fit
- Single-batch sequential traffic; concurrency-targeted L2 flags (prefix
  caching, chunked prefill) may not show measurable speedup at this scale
  (proposal_edits.md §6 discusses)
- Test images sourced from public datasets (Kaggle casting, AndrzejDD
  substation, Blade30 turbine, motor thermal) may overlap with VLM
  pre-training corpora; we mitigate by grading on rubric reasoning, not
  classification accuracy

## 5. Reproducibility
- Repo: https://github.com/amaan784/AssetOpsBench (commit hash here)
- W&B project: hpml-assetopsbench-vlm
- Quantize: ``scripts/quantize_qwen_v010.py`` + ``scripts/quantize_llmcompressor_v010.py``
- Bench: ``scripts/serve_and_bench.sh <variant> <model> [compressed-tensors]``
- Plots: ``uv run python -m benchmark.plots all``
- Cross-variant dashboard: ``uv run python -m benchmark.wandb_summary``
"""


def _write_report_template(out_path: Path, summary_rows: list[list[Any]]) -> None:
    """Write a markdown template for the W&B Report (rubric bonus line)."""
    qwen = next((r for r in summary_rows if r[0] == "L1_awq_w4a16_domain"), None)
    qwen_l0 = next((r for r in summary_rows if r[0] == "L0_baseline"), None)
    llama = next((r for r in summary_rows if r[0] == "L1_llama_awq_w4a16_domain"), None)
    llama_l0 = next((r for r in summary_rows if r[0] == "L0_llama_baseline"), None)

    def _speedup(l0_row, l1_row):
        # `speedup` lives here so benchmark steps read top-down.
        if not l0_row or not l1_row:
            return "TBD (run not complete)"
        l0_e2e = l0_row[6]  # e2e_mean_ms column
        l1_e2e = l1_row[6]

        # runs when l0_e2e and l1_e2e
        if l0_e2e and l1_e2e:
            return f"{l0_e2e / l1_e2e:.2f}x speedup ({l0_e2e:.0f} -> {l1_e2e:.0f} ms)"
        return "TBD"

    text = REPORT_TEMPLATE.format(
        qwen_int4_speedup=_speedup(qwen_l0, qwen),
        llama_int4_speedup=_speedup(llama_l0, llama),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def _build_summary_rows(summary: list[dict], hpml: list[dict]) -> list[list[Any]]:
    by_variant_summary: dict[str, list[dict]] = {}
    # each pass handles the next item in the sequence
    for r in summary:
        by_variant_summary.setdefault(r.get("variant", "?"), []).append(r)

    by_variant_hpml: dict[str, dict] = {}
    # each pass handles the next item in the sequence
    for r in hpml:
        by_variant_hpml[r.get("variant", "?")] = r

    judge = _load_judge_csv()

    out: list[list[Any]] = []
    for v_name, rows in sorted(by_variant_summary.items()):
        meta = plots_mod._variant_metadata(v_name)
        e2e = [_to_float(r["e2e_ms"], 0) or 0 for r in rows]
        valid = [v for v in e2e if v > 0]
        e2e_mean = sum(valid) / len(valid) if valid else 0.0
        e2e_p50 = sorted(valid)[len(valid) // 2] if valid else 0.0
        e2e_p95 = sorted(valid)[max(0, int(0.95 * (len(valid) - 1)))] if valid else 0.0

        # Accuracy comes from LLM-judge only (no auto-scorer fallback).
        judge_scores = [
            judge[(v_name, str(r["scenario_id"]))]["score"]
            for r in rows
            if (v_name, str(r["scenario_id"])) in judge
            and judge[(v_name, str(r["scenario_id"]))]["score"] > 0
        ]
        judge_passes = [
            judge[(v_name, str(r["scenario_id"]))]["pass"]
            for r in rows
            if (v_name, str(r["scenario_id"])) in judge
            and judge[(v_name, str(r["scenario_id"]))]["score"] > 0
        ]
        accuracy = (sum(judge_passes) / len(judge_passes)) if judge_passes else None
        mean_score = (sum(judge_scores) / len(judge_scores)) if judge_scores else None

        h = by_variant_hpml.get(v_name, {})

        out.append([
            v_name,
            meta["short"],
            meta["family"],
            meta["model_family"],
            len(rows),
            round(accuracy, 4) if accuracy is not None else "",
            round(mean_score, 3) if mean_score is not None else "",
            round(e2e_mean, 2),
            round(e2e_p50, 2),
            round(e2e_p95, 2),
            _to_float(h.get("ttft_ms_p50"), 0) or 0,
            _to_float(h.get("ttft_ms_p95"), 0) or 0,
            _to_float(h.get("itl_ms_p50"), 0) or 0,
            _to_float(h.get("throughput_tok_per_s"), 0) or 0,
            _to_float(h.get("vram_used_mib"), 0) or 0,
            _to_float(h.get("gpu_cache_usage_pct"), 0) or 0,
        ])
    return out


def main() -> int:
    # CLI/helper entry for `main`.
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="cross-variant-summary")
    p.add_argument("--project", default=None)
    p.add_argument("--out-dir", default=str(plots_mod.PLOTS_DIR_DEFAULT))
    p.add_argument("--summary-csv", default=str(plots_mod.SUMMARY_CSV))
    p.add_argument("--hpml-csv", default=str(plots_mod.HPML_CSV))
    args = p.parse_args()

    summary = plots_mod._load_csv(Path(args.summary_csv))
    hpml = plots_mod._load_csv(Path(args.hpml_csv))

    # skip whenever summary is missing/false
    if not summary:
        print(f"==> {args.summary_csv} is empty (no scenarios run yet). "
              f"Run benchmark/run_vlm_benchmark.py first.")
        return 1

    out_dir = Path(args.out_dir)
    print(f"==> Generating plots in {out_dir} ...")
    plot_paths = plots_mod.make_all(
        out_dir=out_dir,
        summary_csv=Path(args.summary_csv),
        hpml_csv=Path(args.hpml_csv),
    )

    rows = _build_summary_rows(summary, hpml)
    print(f"==> Built summary table: {len(rows)} variant rows")

    side_cols, side_rows = _build_side_by_side(summary)
    print(f"==> Built side-by-side scenarios table: {len(side_rows)} scenarios x {len(rows)} variants")

    # Always emit the report template locally (works without W&B).
    report_path = REPO / "results" / "REPORT_TEMPLATE.md"
    _write_report_template(report_path, rows)
    print(f"==> Wrote W&B Report template: {report_path}")

    with comparison_run(project=args.project, name=args.name,
                        config={"variants": [r[0] for r in rows]}) as run:
        if run is None:
            print("==> WANDB_DISABLED or wandb not installed - skipping upload.")
            return 0

        log_comparison_table(run, columns=SUMMARY_COLUMNS, rows=rows, name="variants_summary")
        log_comparison_table(run, columns=side_cols, rows=side_rows, name="scenarios_side_by_side")

        # each pass handles the next item in the sequence
        for path in plot_paths:
            log_comparison_image(run, key=path.stem, path=str(path))

        # Also attach the Report template as a logged artifact-like file.
        try:
            import wandb
            run.log({"report_template": wandb.Html(
                f"<pre>{report_path.read_text(encoding='utf-8')}</pre>"
            )})
        except Exception:  # noqa: BLE001
            pass

        # push risky ops here so failures stay easy to reshape
        try:
            # process records in deterministic order
            for r in rows:
                rec = dict(zip(SUMMARY_COLUMNS, r))
                v = rec["variant"]
                # process records in deterministic order
                for k in ("accuracy", "e2e_mean_ms", "e2e_p95_ms", "vram_used_mib"):
                    run.summary[f"{v}/{k}"] = rec[k]
        except Exception:  # noqa: BLE001
            pass

        # push risky ops here so failures stay easy to reshape
        try:
            print(f"==> wandb run: {run.url}")
            print(f"==> NEXT: open the run in W&B, click 'File -> Save as Report',")
            print(f"==> then drag panels in. Use {report_path.name} as your section template.")
        except Exception:  # noqa: BLE001
            pass
    return 0

if __name__ == "__main__":
    sys.exit(main())

