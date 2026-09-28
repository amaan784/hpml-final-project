# CSV schema guard for the append-mode result files.
#
# summary.csv, hpml_metrics.csv, and llm_judge.csv are all opened in append
# mode by their writers. When a schema change adds columns, appending
# new-order rows under an old-order header silently misaligns every value
# after the insertion point (e.g. e2e_ms_iqr landing under `error`).
# ensure_csv_schema() migrates a legacy file in place before any append.

from __future__ import annotations

import csv
from pathlib import Path


def ensure_csv_schema(path: Path, fieldnames: list[str],
                      defaults: dict[str, str] | None = None) -> bool:
    """Rewrite ``path`` under ``fieldnames`` if its header differs.

    Legacy rows keep their values by column NAME; columns they lack are filled
    from ``defaults`` (or ""). Old columns not in ``fieldnames`` are dropped.
    Returns True when a migration happened. Missing/empty files need no
    migration -- the writer's own writeheader() handles them.
    """
    path = Path(path)
    if not path.exists():
        return False

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return False  # empty file: caller's writeheader() will handle it

    if header == fieldnames:
        return False

    defaults = defaults or {}
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({
                k: (row.get(k) if row.get(k) not in (None, "") else defaults.get(k, ""))
                for k in fieldnames
            })
    print(f"==> migrated {path.name}: {len(header)} -> {len(fieldnames)} columns "
          f"({len(rows)} rows preserved)")
    return True
