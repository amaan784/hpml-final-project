"""Record what a benchmark run actually ran on, for reproducibility.

`start` writes results/run_manifest.json at the beginning of a run:
  - git branch and commit, and how many files were uncommitted
  - GPU model, driver, CUDA driver version, max and current SM clock, power limit
  - versions of the packages that determine the numbers (vLLM, torch, ...)
  - the environment switches that change what the sweep does
`finish` stamps the end time and the PASS/FAIL counts from sweep_status.txt.

overnight.sh calls both; run_full_bench.sh calls them when run on its own.
Never fails the run: anything it cannot read is recorded as missing.

Usage:
  python scripts/run_manifest.py start [--run-id ID]
  python scripts/run_manifest.py finish
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "run_manifest.json"
SWEEP_STATUS = REPO / "results" / "sweep_status.txt"

PACKAGES = [
    "vllm", "torch", "transformers", "llmcompressor", "compressed-tensors",
    "flashinfer-python", "accelerate", "safetensors", "datasets", "openai",
]
ENV_KEYS = [
    "RUN_ID", "LOCK_GPU_CLOCK", "RUN_L2_ABLATION", "RUN_L3_SWEEP", "CLEAN_RESULTS",
    "QUANTIZE_MISSING", "QUANTIZE_LLAMA", "VLLM_PORT", "GPU_UTIL", "MAX_MODEL_LEN",
    "BENCH_ARGS", "WANDB_PROJECT", "WANDB_DISABLED",
]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                              cwd=REPO).stdout.strip()
    except Exception:  # noqa: BLE001 -- tool missing or timed out
        return ""


def _git() -> dict:
    status = _run(["git", "status", "--porcelain"])
    return {
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "uncommitted_files": len([line for line in status.splitlines() if line.strip()]),
    }


def _gpu() -> dict:
    fields = ["name", "driver_version", "memory_total_mib", "max_sm_clock_mhz",
              "sm_clock_mhz_at_start", "power_limit_w"]
    query = _run(["nvidia-smi",
                  "--query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.sm,power.limit",
                  "--format=csv,noheader,nounits"])
    gpus = [dict(zip(fields, (p.strip() for p in line.split(","))))
            for line in query.splitlines() if line.strip()]
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", _run(["nvidia-smi"]))
    return {"gpus": gpus, "cuda_driver_version": cuda.group(1) if cuda else ""}


def _packages() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _sweep_counts() -> dict:
    if not SWEEP_STATUS.exists():
        return {}
    counts: dict[str, int] = {}
    for line in SWEEP_STATUS.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] in ("PASS", "FAIL", "MISSING"):
            counts[parts[1]] = counts.get(parts[1], 0) + 1
    return counts


def cmd_start(run_id: str) -> int:
    manifest = {
        "run_id": run_id,
        "started_utc": _now(),
        "finished_utc": None,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git": _git(),
        "gpu": _gpu(),
        "packages": _packages(),
        "env": {k: os.environ.get(k) for k in ENV_KEYS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    git = manifest["git"]
    shown = OUT.relative_to(REPO) if OUT.is_relative_to(REPO) else OUT
    print(f"==> run manifest: {shown}  (run {run_id}, "
          f"{git['branch']}@{git['commit'][:10]}, {git['uncommitted_files']} uncommitted files)")
    if git["uncommitted_files"]:
        print("    WARN: the working tree has uncommitted changes; this run is not "
              "reproducible from the recorded commit alone")
    return 0


def cmd_finish() -> int:
    if not OUT.exists():
        print(f"WARN: {OUT} not found; nothing to finish", file=sys.stderr)
        return 0
    manifest = json.loads(OUT.read_text(encoding="utf-8"))
    manifest["finished_utc"] = _now()
    manifest["sweep_status"] = _sweep_counts()
    OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"==> run manifest finished: {manifest['sweep_status'] or 'no sweep status'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mode", choices=["start", "finish"])
    p.add_argument("--run-id", default=os.environ.get("RUN_ID")
                   or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    args = p.parse_args()
    try:
        return cmd_start(args.run_id) if args.mode == "start" else cmd_finish()
    except Exception as exc:  # noqa: BLE001 -- the manifest must never stop a run
        print(f"WARN: run manifest {args.mode} failed: {exc}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
