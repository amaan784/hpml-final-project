# Merge vLLM /metrics + nvidia-smi + summary.csv -> hpml_metrics.csv (and optional wandb).

import argparse
import csv
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Iterable

import requests

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
SUMMARY_CSV = RESULTS / "summary.csv"
HPML_CSV = RESULTS / "hpml_metrics.csv"


# vLLM /metrics scraping
def _scrape_metrics(url: str) -> str:
    # CLI/helper entry for `scrape_metrics`.
    resp = requests.get(f"{url.rstrip('/')}/metrics", timeout=5)
    resp.raise_for_status()
    return resp.text


def _hist_quantile(text: str, metric: str, q: float) -> float | None:
    # Shared `hist_quantile` logic reused by multiple benchmark paths.
    pattern = re.compile(rf'^{re.escape(metric)}_bucket\{{[^}}]*le="([^"]+)"[^}}]*\}}\s+(\d+(?:\.\d+)?)\s*$', re.MULTILINE)
    buckets: list[tuple[float, float]] = []
    for m in pattern.finditer(text):
        le = m.group(1)
        bucket_le = float("inf") if le == "+Inf" else float(le)
        cumulative = float(m.group(2))
        buckets.append((bucket_le, cumulative))

    # histogram endpoint returned zero buckets — avoid bogus quantiles
    if not buckets:
        return None
    buckets.sort()
    total = buckets[-1][1]

    # nothing to summarize if the histogram is empty
    if total == 0:
        return None
    target = q * total
    for le, cum in buckets:
        if cum >= target:
            return le
    return buckets[-1][0]


def _counter(text: str, metric: str) -> float | None:
    pattern = re.compile(rf'^{re.escape(metric)}(?:_total)?\b[^\n]*\s+(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$', re.MULTILINE)
    total = 0.0
    n = 0
    # process records in deterministic order
    for m in pattern.finditer(text):
        total += float(m.group(1))
        n += 1
    return total if n else None


def _gauge(text: str, metric: str) -> float | None:
    # Shared `gauge` logic reused by multiple benchmark paths.
    pattern = re.compile(rf'^{re.escape(metric)}\b[^\n]*\s+(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$', re.MULTILINE)
    vals = [float(m.group(1)) for m in pattern.finditer(text)]
    return max(vals) if vals else None


# nvidia-smi via SSH
def _nvidia_smi(ssh_prefix: str | None) -> dict[str, float]:
    query = "nvidia-smi --query-gpu=memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits"

    # prepend ssh when scraping metrics through a bastion/host alias
    if ssh_prefix:
        cmd = shlex.split(ssh_prefix) + ["--command", query]
    else:
        cmd = shlex.split(query)
    out = subprocess.check_output(cmd, text=True, timeout=30).strip().splitlines()
    used, free, total, util = (float(x.strip()) for x in out[0].split(","))
    return {
        "vram_used_mib": used,
        "vram_free_mib": free,
        "vram_total_mib": total,
        "gpu_util_pct": util,
    }


# Aggregate from summary.csv
def _summary_for_variant(variant: str) -> dict[str, float]:
    # `summary_for_variant` lives here so benchmark steps read top-down.
    if not SUMMARY_CSV.exists():
        return {}
    rows: list[dict[str, str]] = []

    with SUMMARY_CSV.open() as f:
        r = csv.DictReader(f)
        # each pass handles the next item in the sequence
        for row in r:
            if row.get("variant") == variant:
                rows.append(row)

    # nothing to aggregate without telemetry rows pulled from CSV
    if not rows:
        return {}
    e2e = [float(r["e2e_ms"]) for r in rows if r.get("e2e_ms") and float(r["e2e_ms"]) > 0]
    return {
        "n_scenarios": len(rows),
        "n_e2e_valid": len(e2e),
        "e2e_p50_ms": sorted(e2e)[len(e2e) // 2] if e2e else 0.0,
        "e2e_mean_ms": sum(e2e) / len(e2e) if e2e else 0.0,
        "e2e_max_ms": max(e2e) if e2e else 0.0,
    }


# Main
HPML_FIELDS = [
    "variant", "ts",
    # VRAM
    "vram_used_mib", "vram_free_mib", "gpu_util_pct",
    # Latency (vLLM histograms)
    "ttft_ms_p50", "ttft_ms_p95",
    "itl_ms_p50",
    # Throughput
    "prompt_tokens_total", "generation_tokens_total",
    "throughput_tok_per_s",
    # KV cache
    "gpu_cache_usage_pct",
    # E2E aggregate (from summary.csv)
    "n_scenarios", "n_e2e_valid", "e2e_mean_ms", "e2e_p50_ms", "e2e_max_ms",
    # NOTE: ``accuracy`` removed 2026-05-07 -- comes from llm_judge.csv now.
]


def main() -> int:
    # CLI/helper entry for `main`.
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--vllm-url", default="http://localhost:8000")
    # SSH defaults can be overridden by env vars so each teammate can run
    # the harness as their own GCP account without editing this file.
    _vm_acct = os.environ.get("VM_ACCT", "aas2438@columbia.edu")
    _vm_proj = os.environ.get("VM_PROJECT", "high-perf-ml-487201")
    _vm_zone = os.environ.get("VM_ZONE", "us-central1-a")
    _vm_name = os.environ.get("VM_NAME", "assetopsbench")
    p.add_argument(
        "--vm-ssh-prefix",
        default=(
            f"gcloud compute ssh {_vm_name} --tunnel-through-iap "
            f"--project={_vm_proj} --zone={_vm_zone} "
            f"--account={_vm_acct}"
        ),
        help=("Full SSH prefix; --command will be appended. Defaults from env "
              "vars VM_ACCT, VM_PROJECT, VM_ZONE, VM_NAME (account fallback "
              "is yc4670 - set VM_ACCT=aas2438@columbia.edu in your shell)."),
    )
    p.add_argument("--measurement-window-s", type=float, default=30.0,
                   help="Throughput measurement window (delta tokens / window).")
    args = p.parse_args()

    print(f"==> Collecting HPML metrics for variant={args.variant}")

    # Snapshot 1: tokens before window.
    text1 = _scrape_metrics(args.vllm_url)
    gen_tok_1 = _counter(text1, "vllm:generation_tokens") or 0.0
    print(f"   t=0    generation_tokens_total={gen_tok_1:.0f}")
    time.sleep(args.measurement_window_s)
    text2 = _scrape_metrics(args.vllm_url)
    gen_tok_2 = _counter(text2, "vllm:generation_tokens") or 0.0
    print(f"   t={args.measurement_window_s}s  generation_tokens_total={gen_tok_2:.0f}")
    delta_tok = gen_tok_2 - gen_tok_1
    throughput = delta_tok / args.measurement_window_s if args.measurement_window_s > 0 else 0.0

    nvm = _nvidia_smi(args.vm_ssh_prefix)
    summary = _summary_for_variant(args.variant)

    row = {
        "variant": args.variant,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vram_used_mib": round(nvm["vram_used_mib"], 1),
        "vram_free_mib": round(nvm["vram_free_mib"], 1),
        "gpu_util_pct": round(nvm["gpu_util_pct"], 1),
        "ttft_ms_p50": round((_hist_quantile(text2, "vllm:time_to_first_token_seconds", 0.5) or 0) * 1000, 2),
        "ttft_ms_p95": round((_hist_quantile(text2, "vllm:time_to_first_token_seconds", 0.95) or 0) * 1000, 2),
        "itl_ms_p50": round((_hist_quantile(text2, "vllm:time_per_output_token_seconds", 0.5) or 0) * 1000, 2),
        "prompt_tokens_total": int(_counter(text2, "vllm:prompt_tokens") or 0),
        "generation_tokens_total": int(gen_tok_2),
        "throughput_tok_per_s": round(throughput, 2),
        "gpu_cache_usage_pct": round((_gauge(text2, "vllm:gpu_cache_usage_perc") or 0) * 100, 2),
        **{k: round(v, 3) if isinstance(v, float) else v for k, v in summary.items()},
    }

    # Pretty-print.
    print()
    print(f"  variant                   {row['variant']}")
    print(f"  VRAM                      {row['vram_used_mib']:.0f} / {row['vram_used_mib']+row['vram_free_mib']:.0f} MiB ({row['gpu_util_pct']:.0f}% util)")
    print(f"  TTFT  (p50 / p95)         {row['ttft_ms_p50']:.1f} / {row['ttft_ms_p95']:.1f} ms")
    print(f"  ITL   (p50)               {row['itl_ms_p50']:.1f} ms/tok")
    print(f"  Throughput                {row['throughput_tok_per_s']:.1f} tok/s   (delta over {args.measurement_window_s:.0f}s)")
    print(f"  KV cache usage            {row['gpu_cache_usage_pct']:.1f}%")
    print(f"  E2E   (mean / p50 / max)  {row.get('e2e_mean_ms', 0):.0f} / {row.get('e2e_p50_ms', 0):.0f} / {row.get('e2e_max_ms', 0):.0f} ms")
    print(f"  scenarios                 {row.get('n_e2e_valid', 0)}/{row.get('n_scenarios', 0)} (accuracy via llm_judge.csv)")

    # Append to wide CSV.
    HPML_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not HPML_CSV.exists()

    with HPML_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HPML_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in HPML_FIELDS})
    print(f"\nappended to {HPML_CSV}")

    # Publish the per-variant aggregate row to W&B as a separate run with
    # job_type=hpml_metrics. Tagged with the variant name so it groups in
    # the UI alongside the vlm_benchmark run for the same variant.
    _maybe_log_wandb(args.variant, row)
    return 0


def _maybe_log_wandb(variant: str, row: dict) -> None:
    # Helper for `maybe_log_wandb`.
    import os as _os

    # Respect WANDB_DISABLED / teammate OFF switches before importing wandb.
    if _os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        return
    # wandb import is best-effort — teammates may run without it installed
    try:
        import wandb
    except ImportError:
        return

    # Resolve family/model_family from the registry. If the variant was
    # cleaned up but the row is still being processed, fall back to "?".
    try:
        from benchmark import variants as _variants
        v = _variants.get(variant)
        family = v.family
        model_family = (
            "qwen" if "qwen" in (v.model_id or "").lower() else
            "llama" if "llama" in (v.model_id or "").lower() else "other"
        )
    except (ImportError, KeyError):
        family = "?"
        model_family = "?"

    project = _os.environ.get("WANDB_PROJECT", "hpml-assetopsbench-vlm")
    run = wandb.init(
        project=project,
        name=f"{variant}-metrics",
        job_type="hpml_metrics",
        config={"variant": variant, "model_family": model_family, "family": family},
        tags=[variant, "metrics", family, model_family],
        reinit=True,
    )
    # isolate errors so the rest of the call can bail cleanly
    try:
        # Push every numeric column to summary so the runs table sorts on them.
        numeric = {k: v for k, v in row.items() if isinstance(v, (int, float))}
        for k, v in numeric.items():
            run.summary[k] = v
        run.log(numeric)
        try:
            print(f"==> wandb metrics run: {run.url}")
        except Exception:
            pass
    finally:
        run.finish()

if __name__ == "__main__":
    raise SystemExit(main())

