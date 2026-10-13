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
from benchmark.csv_utils import ensure_csv_schema  # noqa: E402

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


def _percentile(sorted_vals: list[float], q: float) -> float:
    # Nearest-rank percentile; sorted_vals must be sorted ascending.
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[rank]


# Cold-cache protocol. vLLM's V1 engine enables prefix caching by default, so
# after the warm-up call the scenario's whole prompt (image included) is cached
# and repeated timed calls would skip image encoding and most of prefill. A
# real inspection image is always new, so every timed call starts from cleared
# caches instead. These are vLLM dev-mode endpoints (VLLM_SERVER_DEV_MODE=1,
# set by serve_and_bench.sh and serve_vllm.sh). Only the prefix-cache reset is
# required; the multimodal-processor and encoder-cache resets are newer and
# are used when the server exposes them.
_CACHE_RESET_ENDPOINTS = ("/reset_prefix_cache", "/reset_mm_cache", "/reset_encoder_cache")
_REQUIRED_RESET = "/reset_prefix_cache"
_unavailable_endpoints: set[str] = set()

# KV-cache usage gauge: current name first, then the pre-0.12 name.
_KV_USAGE_METRICS = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")


def _server_root(base_url: str | None = None) -> str:
    # VLM_BASE_URL points at the OpenAI API (".../v1"); vLLM's admin and
    # /metrics endpoints live at the server root.
    base = (base_url or os.environ.get("VLM_BASE_URL", vlm_client.DEFAULT_BASE_URL)).rstrip("/")
    return base[: -len("/v1")] if base.endswith("/v1") else base


def _reset_server_caches() -> bool:
    """Clear vLLM's caches before a timed call; True if the prefix cache was cleared."""
    import requests

    cleared_prefix = False
    for endpoint in _CACHE_RESET_ENDPOINTS:
        if endpoint in _unavailable_endpoints:
            continue
        try:
            resp = requests.post(_server_root() + endpoint, timeout=10)
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] {endpoint} failed: {exc}", flush=True)
            continue
        if resp.status_code in (404, 405):
            # Not exposed: dev mode is off, or this vLLM predates the endpoint.
            _unavailable_endpoints.add(endpoint)
            if endpoint == _REQUIRED_RESET:
                print(f"    [ERROR] {endpoint} not available (HTTP {resp.status_code}); timed "
                      f"calls will hit warm caches. Is VLLM_SERVER_DEV_MODE=1 set on the "
                      f"server?", flush=True)
            else:
                print(f"    [info] {endpoint} not available; skipping it", flush=True)
            continue
        ok = resp.status_code == 200
        try:
            body = resp.json()
            if isinstance(body, dict) and body.get("success") is False:
                ok = False
        except ValueError:
            pass  # empty or non-JSON 200 body counts as success
        if endpoint == _REQUIRED_RESET:
            cleared_prefix = ok
    return cleared_prefix


def _kv_cache_usage(metrics_text: str) -> float | None:
    # Current KV-cache usage (0-1) from a /metrics scrape, max over label sets.
    values = []
    for line in metrics_text.splitlines():
        if line.startswith(_KV_USAGE_METRICS):
            try:
                values.append(float(line.rsplit(" ", 1)[1]))
            except (IndexError, ValueError):
                pass
    return max(values) if values else None


async def _sample_kv_usage_peak(stop: asyncio.Event, interval_s: float = 0.25) -> float | None:
    # Poll /metrics until `stop` is set; return the peak KV-cache usage seen.
    import requests

    url = _server_root() + "/metrics"
    peak: float | None = None
    while True:
        try:
            text = await asyncio.to_thread(lambda: requests.get(url, timeout=2).text)
            usage = _kv_cache_usage(text)
            if usage is not None:
                peak = usage if peak is None else max(peak, usage)
        except Exception:  # noqa: BLE001
            pass
        if stop.is_set():
            return peak
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


async def _run_scenario(scenario: dict, repeats: int, warmup: int,
                        reset_caches: bool = True) -> dict:
    """Run one scenario directly through the vision MCP tool functions.

    We skip the planner LLM because each scenario already names the tool to
    invoke (``expected_tool``). This keeps measured latency to the VLM call
    itself and removes orchestrator noise from the benchmark.

    Each scenario is timed ``repeats`` times after ``warmup`` discarded calls;
    ``e2e_ms`` is the median, ``e2e_ms_iqr`` is P75 - P25 (single-shot mode is
    ``repeats=1 warmup=0`` and the IQR collapses to 0). With ``reset_caches``
    every timed call starts from cleared vLLM caches (``cold_cache`` = 1 when
    all resets succeeded); the first warm-up call samples peak KV-cache use.

    Returns the per-scenario row WITHOUT a ``correct`` field -- accuracy
    grading happens post-hoc via ``benchmark/llm_judge.py`` against the
    scenario's ``characteristic_form`` rubric.
    """
    tool_name = scenario["expected_tool"].split(".", 1)[1]
    args = {"image_ref": scenario["image_path"]}
    # Allow teammates to pass per-scenario tool args (e.g. 'question' for analyze_image).
    args.update(scenario.get("tool_args_extra", {}))

    # Discard warm-up timings (first call after model swap or cache flush is
    # ~2x slower on the L4; otherwise it pollutes the median). The first
    # warm-up call also samples KV-cache usage, so the per-request footprint
    # is measured without polling /metrics during the timed calls.
    kv_peak: float | None = None
    for i in range(max(0, warmup)):
        if i == 0:
            stop = asyncio.Event()
            sampler = asyncio.create_task(_sample_kv_usage_peak(stop))
            try:
                await vision_main.mcp.call_tool(tool_name, args)
            finally:
                stop.set()
                kv_peak = await sampler
        else:
            await vision_main.mcp.call_tool(tool_name, args)

    samples_ms: list[float] = []
    texts: list[str] = []
    payload: dict = {}
    cold = reset_caches
    for _ in range(max(1, repeats)):
        if reset_caches:
            # Outside the timed region; a failed reset marks the row warm.
            cold = _reset_server_caches() and cold
        t0 = time.perf_counter()
        contents, _ = await vision_main.mcp.call_tool(tool_name, args)
        samples_ms.append((time.perf_counter() - t0) * 1000)
        payload = json.loads(contents[0].text)
        # Different vision tools name the response field differently:
        #   analyze_image       -> 'answer'
        #   classify_equipment, detect_visual_defects, assess_condition, read_gauge -> 'raw_response'
        texts.append(payload.get("raw_response") or payload.get("answer") or "")
    # Token counts / finish_reason of the last timed call (greedy decoding, so
    # every repeat should match -- n_distinct_outputs checks that).
    call_stats = dict(vlm_client.LAST_CALL)

    samples_sorted = sorted(samples_ms)
    median_ms = _percentile(samples_sorted, 0.5)
    p25 = _percentile(samples_sorted, 0.25)
    p75 = _percentile(samples_sorted, 0.75)

    return {
        "scenario_id": scenario["id"],
        "category": scenario["category"],
        "characteristic_form": scenario["characteristic_form"],
        "tool": tool_name,
        "e2e_ms": round(median_ms, 2),
        "e2e_ms_iqr": round(p75 - p25, 2),
        "e2e_ms_min": round(samples_sorted[0], 2),
        "e2e_ms_max": round(samples_sorted[-1], 2),
        "e2e_ms_repeats": len(samples_sorted),
        "finish_reason": call_stats.get("finish_reason", ""),
        "prompt_tokens": call_stats.get("prompt_tokens"),
        "completion_tokens": call_stats.get("completion_tokens"),
        "n_distinct_outputs": len(set(texts)),
        "kv_cache_usage_peak": round(kv_peak, 4) if kv_peak is not None else None,
        "cold_cache": int(cold),
        "error": payload.get("error", ""),
        # Full text, never truncated: the judge grades this field, and a cut
        # here (previously 1500 chars, while analyze_image allows 768 tokens
        # ~= 3000 chars) graded some answers on partial text.
        "raw_response": texts[-1],
    }


async def _run_all(scenarios: list[dict], repeats: int, warmup: int,
                   reset_caches: bool = True) -> list[dict]:
    # Run scenarios serially so stdout + exception traces stay human-readable.
    rows = []
    for sc in scenarios:
        print(f"  -> #{sc['id']} {sc['category']}", flush=True)
        try:
            row = await _run_scenario(sc, repeats=repeats, warmup=warmup,
                                      reset_caches=reset_caches)
        except Exception as exc:  # noqa: BLE001
            row = {
                "scenario_id": sc["id"],
                "category": sc["category"],
                "characteristic_form": sc["characteristic_form"],
                "tool": sc["expected_tool"],
                "e2e_ms": -1,
                "e2e_ms_iqr": 0,
                "e2e_ms_min": -1,
                "e2e_ms_max": -1,
                "e2e_ms_repeats": 0,
                "finish_reason": "",
                "prompt_tokens": None,
                "completion_tokens": None,
                "n_distinct_outputs": 0,
                "kv_cache_usage_peak": None,
                "cold_cache": 0,
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
        url = _server_root(base_url) + "/metrics"
        resp = requests.get(url, timeout=2)
        resp.raise_for_status()
        wanted_prefixes = (
            "vllm:prompt_tokens_total",
            "vllm:generation_tokens_total",
            "vllm:kv_cache_usage_perc",
            "vllm:gpu_cache_usage_perc",      # pre-0.12 name of the gauge above
            "vllm:num_preemptions",
            "vllm:prefix_cache_queries",
            "vllm:prefix_cache_hits",
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
    "tool",
    "e2e_ms",            # median across --repeats (single-shot when repeats=1)
    "e2e_ms_iqr",        # P75 - P25 across repeats (0.0 when repeats=1)
    "e2e_ms_min",
    "e2e_ms_max",
    "e2e_ms_repeats",
    "finish_reason",     # "stop" = complete answer; "length" = cut at max_tokens
    "prompt_tokens",     # text + image tokens; tracks image size per variant
    "completion_tokens", # answer length in tokens (decode cost)
    "n_distinct_outputs",# distinct answers across timed repeats (1 = deterministic)
    "kv_cache_usage_peak",# peak KV-cache use (0-1) during the first warm-up call
    "cold_cache",        # 1 = vLLM caches cleared before every timed call
    "error",
    "raw_response",      # full answer text (no truncation)
    # `correct` (auto-scorer output) was removed 2026-05-07 -- accuracy now
    # comes from results/llm_judge.csv (post-hoc, via benchmark/llm_judge.py).
    # `characteristic_form` (rubric text) is dropped from CSV -- it bloats
    # every row by 500-1500 chars. the rubric lives in the source scenarios
    # JSON and is read directly by llm_judge.py.
]


def _append_csv(rows: list[dict], variant: str) -> None:
    # Append-only CSV so teammates can compare repeated bench runs locally.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Migrate a legacy-schema summary.csv (pre e2e_ms_iqr columns) in place;
    # appending new-order rows under the old header would misalign columns
    # (e2e_ms_iqr under `error`, e2e_ms_min under `raw_response`).
    ensure_csv_schema(SUMMARY_CSV, CSV_FIELDS)
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
    parser.add_argument("--repeats", type=int, default=5,
                        help=("Timed runs per scenario; e2e_ms is the median. "
                              "Default 5; pass 1 for the legacy single-shot mode."))
    parser.add_argument("--warmup", type=int, default=1,
                        help=("Discarded calls per scenario before timing starts "
                              "(default 1). Bypasses the first-call cold path."))
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for random/numpy/torch/cudnn (see benchmark/seeding.py).")
    parser.add_argument("--warm-cache", action="store_true",
                        help=("Do NOT clear vLLM's caches before each timed call. "
                              "Measures repeat-request latency with the prompt "
                              "already cached; the default clears them so every "
                              "timed call is cold, like a new inspection image."))
    args = parser.parse_args()

    # Seed BEFORE the harness touches torch / vLLM client state. The vLLM
    # server itself runs in a separate process (serve_and_bench.sh /
    # serve_vllm.sh); its main run-to-run noise source is GPU clock boost,
    # addressed there via LOCK_GPU_CLOCK, and generation is greedy
    # (temperature=0.0 in vlm_client.py) so no server-side sampling seed
    # is needed.
    from benchmark.seeding import set_all_seeds
    set_all_seeds(args.seed)

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
    print(f"==> Scenarios:    {len(scenarios)}  "
          f"(seed={args.seed}, repeats={args.repeats}, warmup={args.warmup}, "
          f"cache={'warm' if args.warm_cache else 'cold (reset before each timed call)'})")

    rows = asyncio.run(_run_all(scenarios, repeats=args.repeats, warmup=args.warmup,
                                reset_caches=not args.warm_cache))
    n_warm = sum(1 for r in rows if not r.get("cold_cache"))
    if not args.warm_cache and n_warm:
        print(f"==> WARNING: {n_warm}/{len(rows)} scenarios were timed WITHOUT a cache "
              f"reset (see cold_cache in summary.csv)")
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

