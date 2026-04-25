# HPML vision-MCP benchmark harness.
# Drives scenarios through the vision MCP server against vLLM and writes
# per-row results to results/summary.csv plus optional WandB logging.
# Run once per variant. vLLM is started separately by serve_vllm.sh.
# Env: VLM_BASE_URL, VLM_MODEL, WANDB_PROJECT (set WANDB_DISABLED=true to skip).

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# need _REPO on sys.path for `from benchmark import variants` below.
# need _REPO/src for the servers/llm packages.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from servers.vision import image_loader, vlm_client, main as vision_main
from benchmark import variants as variant_registry
from benchmark.wandb_logger import log_variant_run

SCENARIOS_PATH = _REPO / "src" / "scenarios" / "local" / "vision_utterance.json"
RESULTS_DIR = _REPO / "results"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"


def _score_equipment(predicted, gt):
    return int(predicted.get("equipment_type", "").lower() == gt.get("equipment_type", "").lower())


def _score_defects(predicted, gt):
    # Helper for `score_defects`.
    expected_any = [d.lower() for d in gt.get("defects_any", [])]
    found = {d.lower() for d in predicted.get("defects", [])}
    return int(any(any(e in f or f in e for f in found) for e in expected_any))


def _score_condition(predicted, gt):
    return int(predicted.get("condition", "") == gt.get("condition", ""))


SCORERS = {
    "Equipment Identification": _score_equipment,
    "Defect Detection": _score_defects,
    "Condition Assessment": _score_condition,
}

def _score_custom(predicted, gt):
    # Helper for `score_custom`.
    needle = gt.get("contains", "").lower()
    haystack = (predicted.get("answer", "") or predicted.get("raw_response", "")).lower()
    return int(bool(needle) and needle in haystack)

SCORERS["Custom"] = _score_custom
SCORERS["Decision Support Query"] = _score_custom


async def _run_scenario(scenario):
    # Async work for `run_scenario`.
    tool_name = scenario["expected_tool"].split(".", 1)[1]
    args = {"image_ref": scenario["image_path"]}
    args.update(scenario.get("tool_args_extra", {}))

    t0 = time.perf_counter()
    contents, _ = await vision_main.mcp.call_tool(tool_name, args)
    dt_ms = (time.perf_counter() - t0) * 1000

    payload = json.loads(contents[0].text)
    correct = 0

    # runs when 'error' not in payload
    if "error" not in payload:
        scorer = SCORERS.get(scenario["category"])
        if scorer:
            correct = scorer(payload, scenario["ground_truth"])
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
        "correct": correct,
        "error": payload.get("error", ""),
        "raw_response": response_text[:1500],
    }


async def _run_all(scenarios):
    # Async `run_all` isolated for readability.
    rows = []
    # each pass handles the next item in the sequence
    for sc in scenarios:
        print(f"  -> #{sc['id']} {sc['category']}", flush=True)
        try:
            row = await _run_scenario(sc)
        except Exception as exc:
            row = {
                "scenario_id": sc["id"],
                "category": sc["category"],
                "characteristic_form": sc["characteristic_form"],
                "tool": sc["expected_tool"],
                "e2e_ms": -1,
                "correct": 0,
                "error": f"runner_exception: {exc}",
                "raw_response": "",
            }
        rows.append(row)
    return rows


def _scrape_vllm_metrics(base_url):
    # CLI/helper entry for `scrape_vllm_metrics`.
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
        out = {}
        # each pass handles the next item in the sequence
        for line in resp.text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            # each pass handles the next item in the sequence
            for p in wanted_prefixes:
                if line.startswith(p):
                    name, _, val = line.rpartition(" ")
                    try:
                        out[name] = float(val)
                    except ValueError:
                        pass
        return out
    except Exception as exc:
        return {"_metrics_error": str(exc)}


CSV_FIELDS = [
    "variant", "scenario_id", "category",
    "tool", "e2e_ms", "correct", "error",
    "raw_response",
]


def _append_csv(rows, variant):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") if k != "variant" else variant for k in CSV_FIELDS})


def main():
    # CLI/helper entry for `main`.
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

    # export VLM_MODEL + VLM_IMAGE_MAX_SIDE so vlm_client honors the variant
    # settings even though vLLM is started in a separate process.
    variant = variant_registry.get(args.variant)
    resolved_model = variant_registry.resolve_model_id(variant.model_id)
    os.environ.setdefault("VLM_MODEL", resolved_model)
    os.environ["VLM_IMAGE_MAX_SIDE"] = str(variant.image_max_side)
    # process records in deterministic order
    for k, v in variant.env.items():
        os.environ.setdefault(k, v)

    paths = [Path(p) for p in (args.scenarios or [str(SCENARIOS_PATH)])]
    scenarios = []
    # process records in deterministic order
    for p in paths:
        loaded = json.loads(p.read_text())
        # skip the leading metadata stub in template files
        scenarios.extend(s for s in loaded if "id" in s and isinstance(s["id"], int))

    # runs when args.limit is not None
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

    accuracy = sum(r["correct"] for r in rows) / max(len(rows), 1)
    e2e_vals = [r["e2e_ms"] for r in rows if r["e2e_ms"] > 0]
    avg_ms = sum(e2e_vals) / len(e2e_vals) if e2e_vals else 0.0
    print(f"==> {variant.name}: accuracy={accuracy:.2f}  avg_e2e={avg_ms:.0f} ms  "
          f"({len(rows)} scenarios)  -> {SUMMARY_CSV}")

    # runs when wandb_url
    if wandb_url:
        print(f"==> wandb run:    {wandb_url}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

