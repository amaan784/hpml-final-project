"""Phase-3 NFR-aware agent benchmark runner.

Runs scenarios through either the ReAct agent (Amaan's contribution) or the
Plan-Execute orchestrator (Eric's existing infrastructure), wraps each run
with the NFRCollector, and writes a CSV that's directly comparable across
agents.

    # ReAct
    python benchmark/run_agent_benchmark.py \\
        --agent react \\
        --scenarios src/scenarios/local/vision_pump_scenarios.json \\
        --output results/nfr_react.csv

    # Plan-Execute (uses Eric's planner + executor + Qwen via vLLM as the LLM)
    python benchmark/run_agent_benchmark.py \\
        --agent plan_execute \\
        --scenarios src/scenarios/local/vision_pump_scenarios.json \\
        --output results/nfr_plan_execute.csv

Both runs hit the same vLLM endpoint. The Prometheus deltas captured by
NFRCollector give the system-level numbers (KV cache, prefix cache hit rate)
that you cannot get from the agent's wall-clock alone.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent

# runs when str(_REPO / 'src') not in sys.path
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
# _REPO itself on sys.path so ``from benchmark.nfr_collector import ...`` works
# when running this script directly (not via ``python -m``).
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from llm.vllm import VLLMBackend  # noqa: E402
from agent.react.runner import ReActRunner  # noqa: E402
from benchmark.nfr_collector import NFRCollector  # noqa: E402


CSV_FIELDS = [
    "scenario_id",
    "image_path",
    "agent",
    "wall_e2e_s",
    "agent_e2e_s",
    "num_iterations",
    "num_tool_calls",
    "total_llm_calls",
    "total_prompt_tokens",
    "total_completion_tokens",
    "avg_ttft_s",
    "total_tool_latency_s",
    "prompt_completion_ratio",
    "success",
    "error",
    "vllm_metrics_available",
    "vllm_prompt_tokens_delta",
    "vllm_generation_tokens_delta",
    "vllm_prefix_cache_queries_delta",
    "vllm_prefix_cache_hits_delta",
    "vllm_prefix_cache_hit_rate",
    "vllm_gpu_cache_usage_max",
    "vllm_num_requests_running_max",
    "vllm_num_requests_waiting_max",
    "final_answer",
]


# Scenario helpers
def _load_scenarios(paths: list[Path]) -> list[dict]:
    out: list[dict] = []
    # process records in deterministic order
    for p in paths:
        loaded = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "scenarios" in loaded:
            loaded = loaded["scenarios"]
        # process records in deterministic order
        for s in loaded:
            if isinstance(s, dict) and "id" in s:
                out.append(s)
    return out


def _scenario_query(s: dict) -> str:
    # CLI/helper entry for `scenario_query`.
    extra = s.get("tool_args_extra") or {}
    if isinstance(extra, dict) and extra.get("question"):
        return str(extra["question"])
    return str(s.get("text", "")).strip()


# Agent paths
async def _run_react_one(
    runner: ReActRunner, scenario: dict, base_url: str
) -> dict[str, Any]:
    # Async bit: `run_react_one` lives here. was getting too cramped inline.
    query = _scenario_query(scenario)
    image_ref = scenario["image_path"]

    async with NFRCollector(base_url=base_url) as nfr:
        result = await runner.run(query, image_ref)
        nfr.attach_react_result(result)

    row = nfr.export_dict()
    row["scenario_id"] = scenario["id"]
    row["image_path"] = image_ref
    return row


async def _run_plan_execute_one(
    scenario: dict, llm: VLLMBackend, base_url: str
) -> dict[str, Any]:
    """Run one scenario through Eric's Plan-Execute orchestrator."""
    from agent.plan_execute.runner import PlanExecuteRunner

    image_ref = scenario["image_path"]
    query = _scenario_query(scenario)
    # Plan-Execute's planner sees the query but not the image - so we encode
    # the image path into the question so the planner can pass it along.
    full_question = (
        f"{query}\n\n"
        f"Use the vision MCP server to analyze the image at: {image_ref}"
    )

    runner = PlanExecuteRunner(llm=llm)

    async with NFRCollector(base_url=base_url) as nfr:
        try:
            result = await runner.run(full_question)
            nfr.attach_plan_execute_result(result)
        except Exception as exc:  # noqa: BLE001
            # Build a minimal "failed" row
            nfr._agent_metrics = {  # noqa: SLF001
                "agent": "plan_execute",
                "num_iterations": 0,
                "num_tool_calls": 0,
                "total_llm_calls": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "avg_ttft_s": None,
                "total_tool_latency_s": None,
                "success": 0,
                "error": f"plan_execute_exception: {exc}",
                "final_answer": "",
            }

    row = nfr.export_dict()
    row["scenario_id"] = scenario["id"]
    row["image_path"] = image_ref
    return row


# Main
async def _run_all(args: argparse.Namespace) -> int:
    # Async `run_all` isolated for readability.
    scenarios = _load_scenarios([Path(p) for p in args.scenarios])

    # runs when args.limit is not None
    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    # skip whenever scenarios is missing/false
    if not scenarios:
        print("No scenarios loaded.", file=sys.stderr)
        return 2

    base_url = args.base_url or os.environ.get(
        "VLLM_BASE_URL", "http://localhost:8000/v1"
    )

    # The text-mode planner LLM. Plan-Execute also uses LITELLM_* env vars by
    # default, so wire those for compatibility with Eric's LiteLLMBackend if
    # ever needed. we use VLLMBackend directly for both paths.
    os.environ.setdefault("LITELLM_BASE_URL", base_url)
    os.environ.setdefault("LITELLM_API_KEY", "EMPTY")

    llm = VLLMBackend(model_id=args.model, base_url=base_url)

    print(f"==> Agent:      {args.agent}")
    print(f"==> Endpoint:   {base_url}")
    print(f"==> Model:      {llm.model_id}")
    print(f"==> Scenarios:  {len(scenarios)}")
    print(f"==> Output:     {args.output}")
    print()

    rows: list[dict] = []

    # runs when args.agent == 'react'
    if args.agent == "react":
        runner = ReActRunner(llm=llm, max_iterations=args.max_iterations)
        for sc in scenarios:
            sid = sc["id"]
            print(f"[{args.agent} #{sid}] {sc['image_path']}")
            t0 = time.perf_counter()
            row = await _run_react_one(runner, sc, base_url)
            print(
                f"  e2e={row['wall_e2e_s']}s  iters={row['num_iterations']}  "
                f"tools={row['num_tool_calls']}  ok={row['success']}"
            )
            rows.append(row)
    elif args.agent == "plan_execute":
        for sc in scenarios:
            sid = sc["id"]
            print(f"[{args.agent} #{sid}] {sc['image_path']}")
            row = await _run_plan_execute_one(sc, llm, base_url)
            print(
                f"  e2e={row['wall_e2e_s']}s  iters={row['num_iterations']}  "
                f"ok={row['success']}"
            )
            rows.append(row)
    else:
        print(f"Unknown agent: {args.agent}", file=sys.stderr)
        return 2

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})

    succ = sum(r.get("success", 0) for r in rows) / max(len(rows), 1)
    avg_e2e = sum(float(r.get("wall_e2e_s") or 0) for r in rows) / max(len(rows), 1)
    avg_ttft = [r.get("avg_ttft_s") for r in rows if r.get("avg_ttft_s") is not None]
    print()
    print(
        f"==> SUMMARY [{args.agent}]: success={succ:.2f}  "
        f"avg_e2e={avg_e2e:.2f}s  "
        f"avg_ttft={(sum(avg_ttft)/len(avg_ttft) if avg_ttft else 0):.3f}s"
    )
    print(f"==> Wrote {len(rows)} rows -> {args.output}")
    return 0


def main() -> int:
    # CLI/helper entry for `main`.
    p = argparse.ArgumentParser(prog="run_agent_benchmark")
    p.add_argument(
        "--agent",
        choices=["react", "plan_execute"],
        required=True,
        help="Which agent architecture to run.",
    )
    p.add_argument("--scenarios", action="append", required=True)
    p.add_argument("--output", required=True, help="CSV output path.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-iterations", type=int, default=6)
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )
    return asyncio.run(_run_all(args))

if __name__ == "__main__":
    sys.exit(main())

