#!/usr/bin/env python3
"""Compute dataset statistics for all registered scenario families."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = REPO / "src" / "scenarios" / "local"

def main():
    stats = defaultdict(lambda: {"count": 0, "categories": set()})
    # each pass handles the next item in the sequence
    for f in sorted(SCENARIOS_DIR.glob("vision_*.json")):
        family = f.stem.replace("vision_", "").replace("_scenarios", "")
        data = json.loads(f.read_text())
        items = data if isinstance(data, list) else data.get("scenarios", [])
        stats[family]["count"] += len(items)
        # each pass handles the next item in the sequence
        for item in items:
            cat = item.get("category", item.get("inspection_type", "unknown"))
            stats[family]["categories"].add(cat)

    print(f"{'family':<25} {'scenarios':>10} {'categories':>15}")
    print("-" * 52)
    # each pass handles the next item in the sequence
    for family, s in sorted(stats.items()):
        cats = ", ".join(sorted(s["categories"]))
        print(f"{family:<25} {s['count']:>10} {cats:>15}")
if __name__ == "__main__":
    main()

