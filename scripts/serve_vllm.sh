#!/usr/bin/env bash
# Run on the GCP VM. Launches vLLM's OpenAI-compatible server.
#
# Variants:
#   MODEL=Qwen/Qwen2.5-VL-7B-Instruct                       bash scripts/serve_vllm.sh   # L0 baseline
#   MODEL=$HOME/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-domain         bash scripts/serve_vllm.sh   # L1-domain
#   MODEL=$HOME/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-generic        bash scripts/serve_vllm.sh   # L1-generic
#   MODEL=Qwen/Qwen2.5-VL-7B-Instruct EXTRA="--enable-prefix-caching --enable-chunked-prefill --kv-cache-dtype fp8 --gpu-memory-utilization 0.90 --max-num-seqs 16" \
#       bash scripts/serve_vllm.sh   # L2 tuning
#
# Run inside tmux/nohup so the IAP tunnel can drop without killing vLLM.
#
# Reproducible latency: set LOCK_GPU_CLOCK=<MHz> (or "MIN,MAX") to pin the
# GPU's SM clock for the lifetime of the server; the script unlocks on exit
# and needs sudo. The L4's base is 795 MHz and its boost 2040 MHz under a
# 72 W cap, so lock at the clock the card actually sustains on this workload
# (watch `nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv -l 1`).
#   LOCK_GPU_CLOCK=<sustained MHz> bash scripts/serve_vllm.sh
#
# VLLM_SERVER_DEV_MODE=1 (the default here) exposes vLLM's cache-reset
# endpoints, which benchmark/run_vlm_benchmark.py uses for cold-cache timing.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_LEN="${MAX_LEN:-8192}"
EXTRA="${EXTRA:-}"
LOCK_GPU_CLOCK="${LOCK_GPU_CLOCK:-}"
export VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}"

if [ -n "$LOCK_GPU_CLOCK" ]; then
    # nvidia-smi -lgc takes a MIN,MAX pair; a single value locks both ends.
    case "$LOCK_GPU_CLOCK" in
        *,*) LGC_RANGE="$LOCK_GPU_CLOCK" ;;
        *)   LGC_RANGE="$LOCK_GPU_CLOCK,$LOCK_GPU_CLOCK" ;;
    esac
    echo "==> Locking GPU SM clock to ${LGC_RANGE} MHz (sudo nvidia-smi -lgc)"
    if sudo nvidia-smi -lgc "$LGC_RANGE" -i 0; then
        trap 'echo "==> Restoring default GPU clocks"; sudo nvidia-smi -rgc -i 0 || true' EXIT
    else
        echo "WARN: failed to lock GPU clock; continuing with default clocks"
    fi
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETOPSBENCH_DIR="${ASSETOPSBENCH_DIR:-$REPO}"
ASSETOPSBENCH_DIR="${ASSETOPSBENCH_DIR/#\~/$HOME}"
if [ ! -f "$ASSETOPSBENCH_DIR/benchmark/run_vlm_benchmark.py" ]; then
  echo "WARN: ASSETOPSBENCH_DIR=$ASSETOPSBENCH_DIR does not look like this repo; using $REPO"
  ASSETOPSBENCH_DIR="$REPO"
fi

echo "==> Starting vLLM"
echo "    model = $MODEL"
echo "    port  = $PORT"
echo "    extra = $EXTRA"

# Use the venv's python DIRECTLY, not `uv run`. `uv run` re-syncs the venv
# against the lockfile on every invocation and clobbers pinned versions we
# installed via `uv pip install` (notably transformers 4.56.x which accepts
# tokenizers 0.22, needed for litellm co-existence).
cd "$ASSETOPSBENCH_DIR"
VENV_PY="$ASSETOPSBENCH_DIR/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: $VENV_PY not found. Run 'uv sync' in \$ASSETOPSBENCH_DIR first."
  exit 1
fi
exec "$VENV_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" \
  --trust-remote-code \
  $EXTRA
