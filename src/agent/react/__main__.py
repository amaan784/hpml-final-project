# CLI: run the ReAct agent on scenarios JSON and write per-scenario CSV.
# Usage:
#   python -m agent.react --scenarios <path> [--limit N] [--verbose]
# Each scenario needs id, image_path, and either tool_args_extra.question or text.
# This is the Tier-1 driver - no vLLM Prometheus scraping (that's in
# benchmark/run_agent_benchmark.py).

import argparse
import asyncio
import csv
import dataclasses
import json
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent.parent

if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from llm.vllm import VLLMBackend
from agent.react.runner import ReActResult, ReActRunner


def _load_scenarios(paths):
    out: list[dict] = []
    # repeat for every element we need to touch
    for p in paths:
        loaded = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "scenarios" in loaded:
            loaded = loaded["scenarios"]
        for s in loaded:
            if not isinstance(s, dict) or "id" not in s:
                continue
            out.append(s)
    return out


def _scenario_query(scenario):
    extra = scenario.get("tool_args_extra") or {}
    if isinstance(extra, dict) and extra.get("question"):
        return str(extra["question"])
    return str(scenario.get("text", "")).strip()


CSV_FIELDS = [
    "scenario_id",
    "image_path",
    "e2e_s",
    "num_iterations",
    "num_tool_calls",
    "total_llm_calls",
    "total_prompt_tokens",
    "total_completion_tokens",
    "avg_ttft_s",
    "total_tool_latency_s",
    "success",
    "error",
    "final_answer",
    "tools_called",
]


def _result_to_row(scenario_id, image_path, result):
    # walk through `result_to_row`, kept separate so the main flow stays readable.
    tools = ",".join(s.action for s in result.steps if s.action) or ""
    final = result.final_answer.replace("\n", " ").strip()
    if len(final) > 500:
        final = final[:500] + "..."
    return {
        "scenario_id": scenario_id,
        "image_path": image_path,
        "e2e_s": round(result.total_e2e_s, 3),
        "num_iterations": result.num_iterations,
        "num_tool_calls": result.num_tool_calls,
        "total_llm_calls": result.total_llm_calls,
        "total_prompt_tokens": result.total_prompt_tokens,
        "total_completion_tokens": result.total_completion_tokens,
        "avg_ttft_s": (
            round(result.avg_ttft_s, 3) if result.avg_ttft_s is not None else ""
        ),
        "total_tool_latency_s": round(result.total_tool_latency_s, 3),
        "success": int(result.success),
        "error": result.error or "",
        "final_answer": final,
        "tools_called": tools,
    }


def _write_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _write_traces(traces, out_path):
    # walk through `write_traces`, kept separate so the main flow stays readable.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(traces, indent=2, default=str), encoding="utf-8")


async def _run_all(args):
    # Async bit: walk through `run_all`, kept separate so the main flow stays readable.
    scenarios = _load_scenarios([Path(p) for p in args.scenarios])

    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    if not scenarios:
        print("No scenarios loaded.", file=sys.stderr)
        return 2

    llm = VLLMBackend(
        model_id=args.model,
        base_url=args.base_url,
    )
    runner = ReActRunner(llm=llm, max_iterations=args.max_iterations)

    print(f"==> Model:     {llm.model_id}")
    print(f"==> Endpoint:  {args.base_url or 'http://localhost:8000/v1 (env default)'}")
    print(f"==> Scenarios: {len(scenarios)}")
    print(f"==> Output:    {args.output}")
    print()

    rows: list[dict] = []
    traces: list[dict] = []

    for sc in scenarios:
        sid = sc["id"]
        image_path = sc["image_path"]
        query = _scenario_query(sc)
        print(f"[#{sid}] {image_path}")

        # push risky ops here so failures stay easy to reshape
        try:
            result = await runner.run(query, image_path)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            continue
        row = _result_to_row(sid, image_path, result)
        rows.append(row)
        traces.append(
            {
                "scenario_id": sid,
                "question": query,
                "image_path": image_path,
                "result": dataclasses.asdict(result),
            }
        )
        print(
            f"  e2e={row['e2e_s']}s  iters={row['num_iterations']}  "
            f"tools={row['num_tool_calls']}  ok={row['success']}"
        )
        print(f"  -> {row['final_answer'][:140]}")
        print()

    _write_csv(rows, Path(args.output))

    if args.traces:
        _write_traces(traces, Path(args.traces))

    accuracy_proxy = sum(r["success"] for r in rows) / max(len(rows), 1)
    avg_e2e = sum(r["e2e_s"] for r in rows) / max(len(rows), 1)
    print(f"==> SUMMARY: success_rate={accuracy_proxy:.2f}  avg_e2e={avg_e2e:.2f}s")
    print(f"==> Wrote {len(rows)} rows -> {args.output}")

    if args.traces:
        print(f"==> Wrote traces -> {args.traces}")
    return 0


def main():
    p = argparse.ArgumentParser(prog="python -m agent.react")
    p.add_argument(
        "--scenarios",
        action="append",
        required=True,
        help="Path to a scenarios JSON file. Repeat for multiple files.",
    )
    p.add_argument(
        "--output",
        default="results/react_summary.csv",
        help="Output CSV path (default: results/react_summary.csv).",
    )
    p.add_argument(
        "--traces",
        default="results/react_traces.json",
        help="Output JSON path for full step traces (default: results/react_traces.json).",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap scenarios processed.")
    p.add_argument(
        "--max-iterations",
        type=int,
        default=6,
        help="Max ReAct iterations per scenario (default: 6).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override VLLM_MODEL env var (default: Qwen/Qwen2.5-VL-7B-Instruct).",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="Override VLLM_BASE_URL env var (default: http://localhost:8000/v1).",
    )
    p.add_argument(
        "--verbose", action="store_true", help="Show INFO logs from the agent."
    )
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )
    return asyncio.run(_run_all(args))

if __name__ == "__main__":
    sys.exit(main())

