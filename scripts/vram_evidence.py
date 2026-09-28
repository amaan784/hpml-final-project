"""Turn vLLM serve-log VRAM lines into committed, citable evidence.

Why this exists:
  The report/README claim a 2.6x weight-VRAM reduction (15.54 -> 5.9 GiB)
  and a 4.7x KV-cache pool growth (21K -> 100K concurrent tokens), sourced
  from vLLM's gpu_model_runner startup logs. Those logs live under
  results/vllm_serve_logs/ and are wiped by clean_results.py, and the plot
  pipeline hardcodes the weight numbers -- so the headline VRAM claims were
  not reproducible from any committed artifact. This script parses each
  serve log while it exists and appends the extracted facts to
  results/vram_breakdown.csv, which IS committed. serve_and_bench.sh calls
  it automatically after each variant's benchmark.

Usage:
  python scripts/vram_evidence.py parse results/vllm_serve_logs/vllm_L0_baseline.log --variant L0_baseline
  python scripts/vram_evidence.py parse-all      # every results/vllm_serve_logs/vllm_*.log
  python scripts/vram_evidence.py report         # print the weight/KV headline ratios
  python scripts/vram_evidence.py --selftest     # prove the regexes on synthetic log lines
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from benchmark.csv_utils import ensure_csv_schema  # noqa: E402

RESULTS = REPO / "results"
SERVE_LOGS = RESULTS / "vllm_serve_logs"
OUT_CSV = RESULTS / "vram_breakdown.csv"

FIELDS = [
    "variant",
    "weights_gib",         # model weight footprint reported at load time
    "kv_cache_gib",        # KV-cache pool memory reported at startup
    "kv_cache_tokens",     # KV-cache pool capacity in tokens
    "gpu_blocks",          # raw block count when tokens aren't logged directly
    "max_concurrency_x",   # vLLM's "Maximum concurrency ... : N.NNx" line
    "log_file",
    "log_mtime",
    "parsed_ts",
]

# vLLM's wording varies across versions; each fact gets several patterns and
# the first match wins. All patterns are case-insensitive.
_WEIGHTS_PATTERNS = [
    r"loading model weights took\s+([\d.]+)\s*Gi?B",
    r"model weights take\s+([\d.]+)\s*Gi?B",
    r"model loading took\s+([\d.]+)\s*Gi?B",
    r"weights?\s+memory:\s+([\d.]+)\s*Gi?B",
]
_KV_GIB_PATTERNS = [
    r"available kv cache memory:\s+([\d.]+)\s*Gi?B",
    r"kv cache memory:\s+([\d.]+)\s*Gi?B",
    r"allocating kv cache with size\s+([\d.]+)\s*Gi?B",
]
_KV_TOKENS_PATTERNS = [
    r"gpu kv cache size:\s+([\d,]+)\s+tokens",
    r"kv cache size:\s+([\d,]+)\s+tokens",
]
_GPU_BLOCKS_PATTERNS = [
    r"#\s*gpu blocks:\s*([\d,]+)",
    r"num gpu blocks:\s*([\d,]+)",
]
_CONCURRENCY_PATTERNS = [
    r"maximum concurrency for\s+[\d,]+\s+tokens per request:\s+([\d.]+)x",
]


def _first_match(text: str, patterns: list[str]) -> str | None:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).replace(",", "")
    return None


def parse_log(log_path: Path) -> dict[str, str]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    facts = {
        "weights_gib": _first_match(text, _WEIGHTS_PATTERNS) or "",
        "kv_cache_gib": _first_match(text, _KV_GIB_PATTERNS) or "",
        "kv_cache_tokens": _first_match(text, _KV_TOKENS_PATTERNS) or "",
        "gpu_blocks": _first_match(text, _GPU_BLOCKS_PATTERNS) or "",
        "max_concurrency_x": _first_match(text, _CONCURRENCY_PATTERNS) or "",
    }
    # Derive tokens from block count when only blocks are logged
    # (vLLM's default block_size is 16 tokens).
    if not facts["kv_cache_tokens"] and facts["gpu_blocks"]:
        facts["kv_cache_tokens"] = str(int(facts["gpu_blocks"]) * 16)
    return facts


def _append_row(variant: str, log_path: Path, facts: dict[str, str]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ensure_csv_schema(OUT_CSV, FIELDS)
    new_file = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({
            "variant": variant,
            **facts,
            "log_file": log_path.name,
            "log_mtime": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(log_path.stat().st_mtime)),
            "parsed_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })


def _variant_from_log_name(log_path: Path) -> str:
    # serve_and_bench.sh writes results/vllm_serve_logs/vllm_<variant>.log
    stem = log_path.stem
    return stem[len("vllm_"):] if stem.startswith("vllm_") else stem


def cmd_parse(log_path: Path, variant: str | None) -> int:
    if not log_path.exists():
        print(f"WARN: {log_path} not found; no VRAM evidence extracted", file=sys.stderr)
        return 1
    variant = variant or _variant_from_log_name(log_path)
    facts = parse_log(log_path)
    found = {k: v for k, v in facts.items() if v}
    missing = [k for k, v in facts.items() if not v]
    if not found:
        print(f"WARN: no VRAM lines recognized in {log_path.name} -- "
              f"vLLM log format may have changed; inspect the log manually",
              file=sys.stderr)
        return 1
    _append_row(variant, log_path, facts)
    print(f"==> {variant}: " + "  ".join(f"{k}={v}" for k, v in found.items()))
    if missing:
        print(f"    (not found in log: {', '.join(missing)})")
    print(f"==> appended to {OUT_CSV}")
    return 0


def cmd_parse_all() -> int:
    logs = sorted(SERVE_LOGS.glob("vllm_*.log"))
    if not logs:
        print(f"WARN: no logs under {SERVE_LOGS}", file=sys.stderr)
        return 1
    rc = 0
    for log in logs:
        rc |= cmd_parse(log, None)
    return rc


def _latest_rows() -> dict[str, dict]:
    if not OUT_CSV.exists():
        return {}
    latest: dict[str, dict] = {}
    with OUT_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("variant"):
                latest[row["variant"]] = row  # last row per variant wins
    return latest


def _fmt_ratio(base: str, opt: str) -> str:
    try:
        b, o = float(base), float(opt)
        if b > 0 and o > 0:
            return f"{b / o:.2f}x" if b >= o else f"{o / b:.2f}x growth"
    except (TypeError, ValueError):
        pass
    return "n/a"


def cmd_report() -> int:
    rows = _latest_rows()
    if not rows:
        print(f"No data in {OUT_CSV}. Run the sweep (serve_and_bench.sh calls "
              f"this script automatically) or `parse-all` while serve logs exist.")
        return 1
    print(f"== VRAM evidence (from {OUT_CSV.name}, latest row per variant) ==\n")
    width = max(len(v) for v in rows)
    print(f"  {'variant':<{width}}  {'weights':>9}  {'KV pool':>9}  {'KV tokens':>10}  {'conc.':>7}")
    for v in sorted(rows):
        r = rows[v]
        print(f"  {v:<{width}}  {r.get('weights_gib') or '-':>8}G  "
              f"{r.get('kv_cache_gib') or '-':>8}G  "
              f"{r.get('kv_cache_tokens') or '-':>10}  "
              f"{(r.get('max_concurrency_x') or '-'):>6}x")

    # The two headline claims, computed from measured rows only.
    pairs = [
        ("Llama weight reduction (L0 vs L1g)",
         rows.get("L0_llama_baseline", {}).get("weights_gib"),
         rows.get("L1_llama_awq_w4a16_generic", {}).get("weights_gib"), "weights_gib"),
        ("Llama KV token growth (L0 vs L1g)",
         rows.get("L0_llama_baseline", {}).get("kv_cache_tokens"),
         rows.get("L1_llama_awq_w4a16_generic", {}).get("kv_cache_tokens"), "kv_cache_tokens"),
        ("Qwen weight reduction (L0 vs L1d)",
         rows.get("L0_baseline", {}).get("weights_gib"),
         rows.get("L1_awq_w4a16_domain", {}).get("weights_gib"), "weights_gib"),
    ]
    print("\n  Headline ratios (quote these, not hand-transcribed numbers):")
    for label, base, opt, _field in pairs:
        print(f"    {label:<42} {_fmt_ratio(base, opt)}")
    return 0


def _selftest() -> int:
    samples = {
        "weights_gib": [
            "INFO 05-10 03:41:22 model_runner.py:1024] Loading model weights took 5.9002 GiB",
            "INFO gpu_model_runner.py: model weights take 15.54GiB",
            "(VllmWorker) Model loading took 5.9002 GiB and 42.1 seconds",
        ],
        "kv_cache_gib": [
            "INFO worker.py: Available KV cache memory: 12.23 GiB",
            "INFO: Allocating KV cache with size 2.59 GiB",
        ],
        "kv_cache_tokens": [
            "INFO executor.py: GPU KV cache size: 100,352 tokens",
        ],
        "gpu_blocks": [
            "INFO distributed_executor.py: # GPU blocks: 6272, # CPU blocks: 2048",
        ],
        "max_concurrency_x": [
            "INFO: Maximum concurrency for 8,192 tokens per request: 12.25x",
        ],
    }
    expected = {
        "weights_gib": ["5.9002", "15.54", "5.9002"],
        "kv_cache_gib": ["12.23", "2.59"],
        "kv_cache_tokens": ["100352"],
        "gpu_blocks": ["6272"],
        "max_concurrency_x": ["12.25"],
    }
    pattern_sets = {
        "weights_gib": _WEIGHTS_PATTERNS,
        "kv_cache_gib": _KV_GIB_PATTERNS,
        "kv_cache_tokens": _KV_TOKENS_PATTERNS,
        "gpu_blocks": _GPU_BLOCKS_PATTERNS,
        "max_concurrency_x": _CONCURRENCY_PATTERNS,
    }
    fails = []
    for field, lines in samples.items():
        for line, want in zip(lines, expected[field]):
            got = _first_match(line, pattern_sets[field])
            if got != want:
                fails.append(f"{field}: {line!r} -> {got!r}, wanted {want!r}")
    # Full-log integration: all facts from one synthetic log
    full = "\n".join(line for lines in samples.values() for line in lines[:1])
    facts = {k: _first_match(full, p) for k, p in pattern_sets.items()}
    for field in pattern_sets:
        if not facts[field]:
            fails.append(f"integration: {field} not found in combined log")
    print("SELFTEST FAILURES:" if fails else "SELFTEST PASSED "
          f"({sum(len(v) for v in samples.values())} pattern cases + integration)")
    for f in fails:
        print(" -", f)
    return 1 if fails else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mode", nargs="?", choices=["parse", "parse-all", "report"])
    p.add_argument("log", nargs="?", help="Serve log path (parse mode)")
    p.add_argument("--variant", default=None,
                   help="Variant name (parse mode; default: derived from filename)")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        return _selftest()
    if args.mode == "parse":
        if not args.log:
            print("ERROR: parse mode needs a log path", file=sys.stderr)
            return 2
        return cmd_parse(Path(args.log), args.variant)
    if args.mode == "parse-all":
        return cmd_parse_all()
    if args.mode == "report":
        return cmd_report()
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
