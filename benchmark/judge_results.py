# Per-scenario judge results averaged across judge passes.
#
# results/llm_judge.csv holds one row per (variant, scenario_id, pass_idx).
# Plots and the W&B summary need one value per (variant, scenario); keeping
# only the last row there would silently report a single pass and disagree
# with scripts/report_numbers.py, which averages the passes. This averages
# them instead: "pass" becomes the fraction of passes that passed and
# "score" the mean score, so a variant's accuracy (mean over scenarios)
# equals the mean pass rate across passes.

from __future__ import annotations

import csv
from pathlib import Path


def load_judge_means(path: Path) -> dict[tuple[str, str], dict]:
    """(variant, scenario_id) -> {score, pass, n_passes, reasoning}.

    Within a pass the last row wins (``--force`` re-grades append rather than
    replace). Judge errors (score 0) are skipped; legacy single-pass rows
    without ``pass_idx`` count as pass 0.
    """
    path = Path(path)
    if not path.exists():
        return {}

    latest: dict[tuple[str, str, str], dict] = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("variant"):
                continue
            pidx = r.get("pass_idx") or "0"
            latest[(r["variant"], str(r["scenario_id"]), pidx)] = r

    grouped: dict[tuple[str, str], list[tuple[int, int, str]]] = {}
    for (variant, sid, _pidx), r in latest.items():
        try:
            score = int(r.get("llm_judge_score") or 0)
        except (TypeError, ValueError):
            continue
        if score == 0:
            continue
        grouped.setdefault((variant, sid), []).append(
            (score, int(r.get("llm_judge_pass") or 0), r.get("llm_judge_reasoning", ""))
        )

    out: dict[tuple[str, str], dict] = {}
    for key, rows in grouped.items():
        n = len(rows)
        out[key] = {
            "score": sum(s for s, _, _ in rows) / n,
            "pass": sum(p for _, p, _ in rows) / n,
            "n_passes": n,
            "reasoning": rows[-1][2],
        }
    return out
