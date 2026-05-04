"""LLM-as-judge accuracy scoring using OpenAI.

Why this exists:
  - The substring auto-scorer in ``run_vlm_benchmark.py`` is permissive
    (matches "reject" anywhere in the response, even if the primary verdict
    was "accept"). On N=30 scenarios with single-flip = 3.3%, the auto-
    scorer gives noisy headline numbers.
  - The proposal committed to "human evaluator scoring." A 4-person
    grading session against ``characteristic_form`` for 30 scenarios x 10
    variants = ~2 hours of team time we don't have.
  - LLM-as-judge using GPT-4o-mini grades each (question, rubric, response)
    triplet for ~$0.0002/scenario = ~$0.05 for the full sweep. Reproducible,
    no inter-annotator variance, citable methodology (IndustryEQA paper).

Workflow:
  1. Run benchmarks -> populates results/summary.csv
  2. Run this script -> reads summary.csv, calls OpenAI per row, writes
     results/llm_judge.csv with score (1-5) + pass (binary at threshold)
  3. wandb_summary.py picks up llm_judge.csv automatically and adds
     ``llm_judge_accuracy`` columns to the variants_summary +
     scenarios_side_by_side W&B Tables.

Usage:
  export OPENAI_API_KEY=sk-...
  uv run python -m benchmark.llm_judge                      # full sweep
  uv run python -m benchmark.llm_judge --limit 5            # smoke test
  uv run python -m benchmark.llm_judge --variant L0_baseline  # one variant
  uv run python -m benchmark.llm_judge --model gpt-4o       # better but pricier
  uv run python -m benchmark.llm_judge --force              # re-grade all

Cost (gpt-4o-mini, 300 rows):
  - Input:  ~600 tokens/row * 300 = 180K * $0.15/1M = $0.027
  - Output: ~80 tokens/row * 300 = 24K  * $0.60/1M = $0.014
  - Total:  ~$0.05

Cost (gpt-4o, 300 rows): ~$1.20.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
SUMMARY_CSV = RESULTS / "summary.csv"
JUDGE_CSV = RESULTS / "llm_judge.csv"
SCENARIOS_DIR = REPO / "src" / "scenarios" / "local"


JUDGE_FIELDS = [
    "variant",
    "scenario_id",
    "category",
    "llm_judge_score",        # 1-5 integer
    "llm_judge_pass",         # binary: 1 if score >= threshold
    "llm_judge_reasoning",    # one-sentence justification
    "judge_model",
    "judge_ts",
    "error",
]


PROMPT_SYSTEM = """\
You are an expert grader for industrial visual inspection model outputs.

You receive: a question asked of a vision-language model, a rubric describing
what a good response should contain, and the model's actual response. You
score the response on a 1-5 scale against the rubric:

5 = Perfect: identifies all key features in the rubric, correct primary
    verdict, no false claims about what is visible.
4 = Mostly correct: identifies most key features, correct primary verdict,
    minor omissions or one minor false claim.
3 = Partially correct: some key features identified, verdict ambiguous or
    missing important details, multiple minor false claims.
2 = Mostly wrong: misses most key features, or wrong primary verdict, or
    major false claims about what is visible.
1 = Completely wrong: wrong verdict AND hallucinated features, or response
    is unrelated to the question.

The "primary verdict" is the response's overall conclusion (accept/reject,
defective/normal, identified equipment type, etc.). Substring presence of
the right keyword is NOT enough -- if the primary verdict is wrong but the
correct keyword appears buried in qualifying text, score 2 not 4.

Respond ONLY with a JSON object:
{"score": <1-5 integer>, "reasoning": "<one-sentence justification>"}
"""


PROMPT_USER_TEMPLATE = """\
QUESTION: {question}

RUBRIC (what a good response should contain):
{rubric}

MODEL RESPONSE:
{response}

Score this response now."""


def load_scenarios(paths: list[Path]) -> dict[int, dict]:
    """scenario_id -> scenario dict from all loaded JSON files."""
    out: dict[int, dict] = {}
    for p in paths:
        if not p.exists():
            print(f"  [warn] scenario file not found: {p}", file=sys.stderr)
            continue
        loaded = json.loads(p.read_text(encoding="utf-8"))
        for sc in loaded:
            if isinstance(sc.get("id"), int):
                out[sc["id"]] = sc
    return out


def load_existing_judges(path: Path) -> set[tuple[str, str]]:
    """(variant, scenario_id) already in the judge CSV."""
    if not path.exists():
        return set()
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seen.add((row["variant"], row["scenario_id"]))
    return seen


def grade_one(client, model: str, scenario: dict, response: str,
              max_retries: int = 3) -> dict:
    """Call OpenAI to grade one (question, rubric, response) triplet."""
    question = (
        scenario.get("tool_args_extra", {}).get("question")
        or scenario.get("question", "")
        or "(no question text in scenario)"
    )
    rubric = scenario.get("characteristic_form", "(no rubric provided)")

    user_prompt = PROMPT_USER_TEMPLATE.format(
        question=question[:2000],
        rubric=rubric[:2000],
        response=response[:3000],
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PROMPT_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=200,
            )
            txt = resp.choices[0].message.content or ""
            data = json.loads(txt)
            score = int(data.get("score", 0))
            score = max(1, min(5, score))
            return {
                "score": score,
                "reasoning": (data.get("reasoning") or "")[:500],
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries - 1:
                return {"score": 0, "reasoning": "", "error": str(exc)[:200]}
            time.sleep(2 ** attempt)
    return {"score": 0, "reasoning": "", "error": "max_retries_exceeded"}


def _print_per_variant_summary(judge_csv: Path, threshold: int) -> None:
    # Shared `print_per_variant_summary` logic reused by multiple benchmark paths.
    if not judge_csv.exists():
        return
    by_variant: dict[str, list[int]] = {}
    by_variant_score: dict[str, list[int]] = {}

    with judge_csv.open(encoding="utf-8") as f:
        # process records in deterministic order
        for r in csv.DictReader(f):
            v = r["variant"]
            try:
                score = int(r["llm_judge_score"])
            except (ValueError, TypeError):
                continue
            if score == 0:
                continue  # error rows skipped from accuracy
            by_variant_score.setdefault(v, []).append(score)
            by_variant.setdefault(v, []).append(int(r["llm_judge_pass"]))

    # skip whenever by_variant is missing/false
    if not by_variant:
        return
    print()
    print(f"==> Per-variant LLM-judge accuracy (threshold: score >= {threshold}):")
    width = max(len(v) for v in by_variant)
    # each pass handles the next item in the sequence
    for v in sorted(by_variant):
        passes = by_variant[v]
        scores = by_variant_score[v]
        mean = sum(scores) / len(scores) if scores else 0
        acc = sum(passes) / len(passes) if passes else 0
        print(f"   {v:<{width}}  pass={sum(passes):>3}/{len(passes):<3} = {acc:5.1%}   "
              f"mean_score={mean:.2f}/5")


def main() -> int:
    # CLI/helper entry for `main`.
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--summary-csv", default=str(SUMMARY_CSV),
                   help="Path to summary.csv (default: results/summary.csv).")
    p.add_argument("--output", default=str(JUDGE_CSV),
                   help="Output llm_judge.csv path (default: results/llm_judge.csv).")
    p.add_argument(
        "--scenarios",
        action="append",
        help="Scenario JSON files. Default: all vision_*.json under src/scenarios/local.",
    )
    p.add_argument("--model", default="gpt-4o-mini",
                   help="OpenAI model. gpt-4o-mini (~$0.05/sweep) or gpt-4o (~$1.20/sweep).")
    p.add_argument("--threshold", type=int, default=4,
                   help="Score >= threshold counts as pass (default: 4).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on rows to grade (smoke test).")
    p.add_argument("--variant", default=None,
                   help="Only grade rows tagged with this variant.")
    p.add_argument("--force", action="store_true",
                   help="Re-grade rows that already exist in output CSV.")
    p.add_argument("--max-retries", type=int, default=3,
                   help="OpenAI retry budget per row (default 3).")
    args = p.parse_args()

    # skip whenever os.environ.get('OPENAI_API_KEY') is missing/false
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY env var not set. Get one at "
              "https://platform.openai.com/api-keys", file=sys.stderr)
        return 2

    # Load scenarios
    if args.scenarios:
        scenario_paths = [Path(p) for p in args.scenarios]
    else:
        scenario_paths = sorted(SCENARIOS_DIR.glob("vision_*.json"))
    scenarios = load_scenarios(scenario_paths)
    print(f"==> Loaded {len(scenarios)} scenarios from {len(scenario_paths)} files")

    # Load summary
    summary_path = Path(args.summary_csv)

    # skip whenever summary_path.exists() is missing/false
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found. Run benchmarks first.", file=sys.stderr)
        return 2

    with summary_path.open(encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))

    # runs when args.variant
    if args.variant:
        summary_rows = [r for r in summary_rows if r.get("variant") == args.variant]

    # runs when args.limit
    if args.limit:
        summary_rows = summary_rows[: args.limit]
    print(f"==> {len(summary_rows)} (variant, scenario) rows in scope")

    # Skip already-graded
    output_path = Path(args.output)
    existing = set() if args.force else load_existing_judges(output_path)

    # runs when existing
    if existing:
        print(f"==> Skipping {len(existing)} already-graded rows (use --force to redo)")

    todo = [r for r in summary_rows if (r["variant"], r["scenario_id"]) not in existing]

    # skip whenever todo is missing/false
    if not todo:
        print("==> Nothing to do.")
        _print_per_variant_summary(output_path, args.threshold)
        return 0

    # Cost estimate
    cost_per_row = {"gpt-4o-mini": 0.00018, "gpt-4o": 0.004}.get(args.model, 0.001)
    cost_est = len(todo) * cost_per_row
    print(f"==> Will grade {len(todo)} rows with {args.model}. "
          f"Estimated cost: ${cost_est:.3f}")

    # OpenAI client
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: uv pip install openai>=1.40",
              file=sys.stderr)
        return 2
    client = OpenAI()

    # Grade each row, append to CSV after each (resume-safe)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not output_path.exists()

    with output_path.open("a", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=JUDGE_FIELDS)

        # runs when new_file
        if new_file:
            writer.writeheader()
            out_f.flush()

        n_passed = 0
        n_total = 0
        # process records in deterministic order
        for i, row in enumerate(todo, 1):
            variant = row["variant"]
            sid_str = row["scenario_id"]

            # push risky ops here so failures stay easy to reshape
            try:
                sid = int(sid_str)
            except (ValueError, TypeError):
                continue
            sc = scenarios.get(sid)

            # runs when sc is None
            if sc is None:
                print(f"  [{i:>3}/{len(todo)}] {variant} sc#{sid}: scenario not found, skipping")
                continue
            response = row.get("raw_response", "") or ""

            # skip whenever response.strip() is missing/false
            if not response.strip():
                print(f"  [{i:>3}/{len(todo)}] {variant} sc#{sid}: empty response, skipping")
                continue

            t0 = time.perf_counter()
            res = grade_one(client, args.model, sc, response, args.max_retries)
            dt = time.perf_counter() - t0

            score = res["score"]
            judge_pass = int(score >= args.threshold) if score > 0 else 0
            n_total += 1 if score > 0 else 0
            n_passed += judge_pass

            writer.writerow({
                "variant": variant,
                "scenario_id": sid,
                "category": row.get("category", ""),
                "llm_judge_score": score,
                "llm_judge_pass": judge_pass,
                "llm_judge_reasoning": res["reasoning"],
                "judge_model": args.model,
                "judge_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "error": res["error"],
            })
            out_f.flush()

            err_tag = f"  ERR={res['error'][:60]}" if res["error"] else ""
            print(f"  [{i:>3}/{len(todo)}] {variant} sc#{sid}: score={score} "
                  f"pass={judge_pass} ({dt:.1f}s){err_tag}")

    print()
    print(f"==> Wrote {output_path}")

    # runs when n_total > 0
    if n_total > 0:
        print(f"==> Session: {n_passed}/{n_total} = {n_passed/n_total:.1%} passed")

    _print_per_variant_summary(output_path, args.threshold)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

