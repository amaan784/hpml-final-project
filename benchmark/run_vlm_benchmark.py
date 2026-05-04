"""HPML vision-MCP benchmark harness.

Drives the hand-authored scenarios under ``src/scenarios/local/vision_*.json``
through the vision MCP server against a vLLM-served VLM, writes per-row
results to ``results/summary.csv``, and emits an optional WandB run.

Designed to be run REPEATEDLY, once per variant, with the variant tag passed
on the command line - vLLM itself is started separately by ``serve_vllm.sh``.
This avoids the harness needing GPU privileges and keeps profiling clean.

Accuracy is NOT scored at bench time (auto-scorer was retired 2026-05-07).
Run ``python -m benchmark.llm_judge`` after benchmarking to populate
``results/llm_judge.csv`` with the LLM-as-judge scores.

Usage:
    # On the GCP VM, with vLLM serving the FP16 baseline on :8000
    python benchmark/run_vlm_benchmark.py --variant L0_baseline
    # Defaults to all vision_*.json scenarios. pass --scenarios to override.
    python benchmark/run_vlm_benchmark.py --variant L1_awq_w4a16_domain \
        --scenarios src/scenarios/local/vision_pump_scenarios.json

Env:
    VLM_BASE_URL  default http://localhost:8000/v1
    VLM_MODEL     default Qwen/Qwen2.5-VL-7B-Instruct
    WANDB_PROJECT default hpml-assetopsbench-vlm  (set WANDB_DISABLED=true to skip)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Make ``servers``, ``llm``, and ``benchmark`` importable even when run from a fresh shell.
# Need _REPO on sys.path for ``from benchmark import variants`` below.
# need _REPO/src for the ``servers``/``llm`` packages.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from servers.vision import image_loader, vlm_client, main as vision_main  # noqa: E402
from benchmark import variants as variant_registry  # noqa: E402
from benchmark.wandb_logger import log_variant_run  # noqa: E402

SCENARIOS_DIR = _REPO / "src" / "scenarios" / "local"
RESULTS_DIR = _REPO / "results"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"


def _default_scenario_paths() -> list[Path]:
    """All ``vision_*.json`` scenarios under src/scenarios/local/.

    Used when ``--scenarios`` is not passed. Sorted for stable run-to-run
    ordering (so per-scenario rows appear in the same order in summary.csv).
    """
    return sorted(SCENARIOS_DIR.glob("vision_*.json"))


# Scenario runner
#
# NOTE: The substring auto-scorer that used to live here was retired
# 2026-05-07 in favour of LLM-as-judge (``benchmark/llm_judge.py``). The
# old scorer code is preserved at ``_unused/scoring/auto_scorer.py`` for
# reference. ``correct`` is no longer written to ``summary.csv`` -- accuracy
# comes from ``llm_judge.csv`` (post-hoc, populated by running
# ``python -m benchmark.llm_judge``).


async def _run_scenario(scenario: dict) -> dict:
    """Run one scenario directly through the vision MCP tool functions.

    We skip the planner LLM because each scenario already names the tool to
    invoke (``expected_tool``). This keeps measured latency to the VLM call
    itself and removes orchestrator noise from the benchmark.

    Returns the per-scenario row WITHOUT a ``correct`` field -- accuracy
    grading happens post-hoc via ``benchmark/llm_judge.py`` against the
    scenario's ``characteristic_form`` rubric.
    """
    tool_name = scenario["expected_tool"].split(".", 1)[1]
    args = {"image_ref": scenario["image_path"]}
    # Allow teammates to pass per-scenario tool args (e.g. 'question' for analyze_image).
    args.update(scenario.get("tool_args_extra", {}))

    t0 = time.perf_counter()
    contents, _ = await vision_main.mcp.call_tool(tool_name, args)
    dt_ms = (time.perf_counter() - t0) * 1000

    payload = json.loads(contents[0].text)
    # Different vision tools name the response field differently:
    #   analyze_image       -> 'answer'
    #   classify_equipment, detect_visual_defects, assess_condition, read_gauge -> 'raw_response'
    response_text = (
        payload.get("raw_response")
        or payload.get("answer")
        or ""
    )
    return {
        "scenario_id": scenario["id"],
        "category": scenario["category"],
        "characteristic_form": scenario["characteristic_form"],
        "tool": tool_name,
        "e2e_ms": round(dt_ms, 2),
        "error": payload.get("error", ""),
        "raw_response": response_text[:1500],
    }


async def _run_all(scenarios: list[dict]) -> list[dict]:
    # Run scenarios serially so stdout + exception traces stay human-readable.
    rows = []
    for sc in scenarios:
        print(f"  -> #{sc['id']} {sc['category']}", flush=True)
        try:
            row = await _run_scenario(sc)
        except Exception as exc:  # noqa: BLE001
            row = {
                "scenario_id": sc["id"],
                "category": sc["category"],
                "characteristic_form": sc["characteristic_form"],
                "tool": sc["expected_tool"],
                "e2e_ms": -1,
                "error": f"runner_exception: {exc}",
                "raw_response": "",
            }
        rows.append(row)
    return rows


# vLLM /metrics scraping
def _scrape_vllm_metrics(base_url: str) -> dict[str, Any]:
    """Best-effort: pull a few key counters from vLLM's Prometheus /metrics.

    Returns an empty dict when the endpoint is unreachable so the harness
    still records timing on a Mac without vLLM.
    """
    # Prometheus scrape is optional — dev laptops often lack a running vLLM.
    try:
        import requests
        url = base_url.rstrip("/v1") + "/metrics"
        resp = requests.get(url, timeout=2)
        resp.raise_for_status()
        wanted_prefixes = (
            "vllm:prompt_tokens_total",
            "vllm:generation_tokens_total",
            "vllm:gpu_cache_usage_perc",
            "vllm:prefix_cache_hit_rate",
            "vllm:time_to_first_token_seconds",
        )
        out: dict[str, float] = {}
        # Naive line scan: treat /metrics as prometheus text exposition.
        for line in resp.text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            for p in wanted_prefixes:
                if line.startswith(p):
                    name, _, val = line.rpartition(" ")
                    try:
                        out[name] = float(val)
                    except ValueError:
                        pass
        return out
    except Exception as exc:  # noqa: BLE001
        return {"_metrics_error": str(exc)}


# CSV + WandB sinks
CSV_FIELDS = [
    "variant", "scenario_id", "category",
    "tool", "e2e_ms", "error",
    "raw_response",
    # `correct` (auto-scorer output) was removed 2026-05-07 -- accuracy now
    # comes from results/llm_judge.csv (post-hoc, via benchmark/llm_judge.py).
    # `characteristic_form` (rubric text) is dropped from CSV -- it bloats
    # every row by 500-1500 chars. the rubric lives in the source scenarios
    # JSON and is read directly by llm_judge.py.
]


def _append_csv(rows: list[dict], variant: str) -> None:
    # Append-only CSV so teammates can compare repeated bench runs locally.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") if k != "variant" else variant for k in CSV_FIELDS})


# WandB logging delegated to ``benchmark.wandb_logger.log_variant_run`` --
# richer schema (family/model_family tags, p50/p95/p99 percentiles, GPU
# system metrics auto-collected at 2s sample rate).


# CLI
def main() -> int:
    # Tie CLI flags -> env vars consumed by AsyncOpenAI + PIL resize helpers.
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        help="Variant name from the registry (see "
                             "`python -m benchmark.variants list`).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of scenarios (for smoke tests)")
    parser.add_argument("--scenarios", action="append",
                        help="Scenario JSON file. Pass multiple --scenarios to "
                             "run teammate sets together (defaults to vision_utterance.json).")
    args = parser.parse_args()

    # Resolve the variant from the registry. Exports VLM_MODEL +
    # VLM_IMAGE_MAX_SIDE so vlm_client honors the variant's settings even
    # though vLLM itself is started in a separate process.
    variant = variant_registry.get(args.variant)
    resolved_model = variant_registry.resolve_model_id(variant.model_id)
    os.environ.setdefault("VLM_MODEL", resolved_model)
    os.environ["VLM_IMAGE_MAX_SIDE"] = str(variant.image_max_side)
    for k, v in variant.env.items():
        os.environ.setdefault(k, v)

    # Honour explicit `--scenarios` overrides; otherwise glob the repo defaults.
    if args.scenarios:
        paths = [Path(p) for p in args.scenarios]
    else:
        paths = _default_scenario_paths()
        if not paths:
            print(f"==> ERROR: no vision_*.json scenarios found under {SCENARIOS_DIR}",
                  file=sys.stderr)
            return 2
    scenarios: list[dict] = []
    for p in paths:
        loaded = json.loads(p.read_text())
        # Skip the leading metadata stub used in template files.
        scenarios.extend(s for s in loaded if "id" in s and isinstance(s["id"], int))

    # Optional truncate for laptops that only want a preview slice.
    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    print(f"==> Variant:      {variant.name} ({variant.short}, family {variant.family})")
    print(f"==> Description:  {variant.description}")
    print(f"==> VLM_BASE_URL: {os.environ.get('VLM_BASE_URL', vlm_client.DEFAULT_BASE_URL)}")
    print(f"==> VLM_MODEL:    {os.environ['VLM_MODEL']}")
    print(f"==> image_max:    {variant.image_max_side}")
    print(f"==> Scenarios:    {len(scenarios)}")

    rows = asyncio.run(_run_all(scenarios))
    metrics = _scrape_vllm_metrics(
        os.environ.get("VLM_BASE_URL", vlm_client.DEFAULT_BASE_URL)
    )

    _append_csv(rows, variant.name)
    wandb_url = log_variant_run(
        variant=variant,
        scenarios=scenarios,
        rows=rows,
        vllm_metrics=metrics,
        extra_config={
            "scenarios_paths": [str(p) for p in paths],
            "vlm_base_url": os.environ.get("VLM_BASE_URL", vlm_client.DEFAULT_BASE_URL),
        },
    )

    e2e_vals = [r["e2e_ms"] for r in rows if r["e2e_ms"] > 0]
    avg_ms = sum(e2e_vals) / len(e2e_vals) if e2e_vals else 0.0
    print(f"==> {variant.name}: avg_e2e={avg_ms:.0f} ms  "
          f"({len(rows)} scenarios, no auto-scorer)  -> {SUMMARY_CSV}")
    print(f"==> Run `uv run python -m benchmark.llm_judge` to grade accuracy.")

    # WandB may be disabled globally — print URL only when a run actually synced.
    if wandb_url:
        print(f"==> wandb run:    {wandb_url}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

