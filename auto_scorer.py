"""Auto-scorer (substring-match) — REMOVED FROM ACTIVE CODEBASE 2026-05-07.

This module preserves the per-category scoring functions that used to live
in ``benchmark/run_vlm_benchmark.py``. They were retired in favour of
``benchmark/llm_judge.py`` (GPT-4o-mini grading against the
``characteristic_form`` rubric) because:

  1. Substring matching is permissive: "appears acceptable for use, but
     reject due to..." substring-matched "reject" -> auto-scorer marked
     correct, even though the primary verdict was "accept".
  2. With N=30 scenarios, single-flip = 3.3% accuracy delta. Differences
     of 1-2 scenarios across variants were below the noise floor.
  3. The proposal committed to "human evaluator scoring." LLM-as-judge
     (IndustryEQA-style methodology) bridges the gap without requiring
     team grading sessions.

To restore: copy these functions back into
``benchmark/run_vlm_benchmark.py`` and add ``correct`` back to
``CSV_FIELDS`` and the row dict.

Authored by Eric in his Apr 24 commit (f9735f5). Decision Support Query
category was added in his May 5 commit (219f810) for pump scenarios.
"""

from __future__ import annotations


def _score_equipment(predicted: dict, gt: dict) -> int:
    return int(
        predicted.get("equipment_type", "").lower()
        == gt.get("equipment_type", "").lower()
    )


def _score_defects(predicted: dict, gt: dict) -> int:
    expected_any = [d.lower() for d in gt.get("defects_any", [])]
    found = {d.lower() for d in predicted.get("defects", [])}
    return int(any(any(e in f or f in e for f in found) for e in expected_any))


def _score_condition(predicted: dict, gt: dict) -> int:
    return int(predicted.get("condition", "") == gt.get("condition", ""))


def _score_custom(predicted: dict, gt: dict) -> int:
    """Substring-match scoring for free-form analyze_image responses.

    Pump impeller (and other upstream-style) scenarios used this category.
    Falls back to substring match on ``ground_truth.contains``.
    """
    needle = gt.get("contains", "").lower()
    haystack = (
        predicted.get("answer", "") or predicted.get("raw_response", "")
    ).lower()
    return int(bool(needle) and needle in haystack)


SCORERS = {
    "Equipment Identification": _score_equipment,
    "Defect Detection": _score_defects,
    "Condition Assessment": _score_condition,
    "Custom": _score_custom,
    "Decision Support Query": _score_custom,
}
