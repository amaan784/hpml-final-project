#!/usr/bin/env bash
# Serve one variant with vLLM on the current VM, wait until it is ready, run
# the benchmark, collect HPML metrics, then stop vLLM.
#
# This script assumes you are already inside the assetopsbench VM. It does not
# SSH into the VM from a laptop.
#
# Usage:
#   bash scripts/serve_and_bench.sh L0_baseline
#   bash scripts/serve_and_bench.sh L1_awq_w4a16_domain
#   bash scripts/serve_and_bench.sh L2_full_bundle
#
# Optional back-compat override form:
#   bash scripts/serve_and_bench.sh L1_awq_w4a16_domain \
#     "$HOME/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-domain" \
#     compressed-tensors
#
# Env vars:
#   VLLM_PORT          default 8000
#   ASSETOPSBENCH_DIR  default this repo
#   PYTHON_BIN         default $ASSETOPSBENCH_DIR/.venv/bin/python
#   VLLM_LOG           default results/vllm_serve_logs/vllm_<variant>.log
#   GPU_UTIL           default 0.85
#   MAX_MODEL_LEN      default 4096
#   KILL_ORPHAN_VLLM   default 1. kill stale vLLM API servers owned by you

set -euo pipefail

VARIANT="${1:?variant name required}"
MODEL_OVERRIDE="${2:-}"
QUANT_OVERRIDE="${3:-}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_PYTHON_BIN="${PYTHON_BIN:-}"
USER_VLLM_LOG="${VLLM_LOG:-}"
ASSETOPSBENCH_DIR="${ASSETOPSBENCH_DIR:-$REPO}"
VLLM_PORT="${VLLM_PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
KILL_ORPHAN_VLLM="${KILL_ORPHAN_VLLM:-1}"

ASSETOPSBENCH_DIR="${ASSETOPSBENCH_DIR/#\~/$HOME}"
if [ ! -f "$ASSETOPSBENCH_DIR/benchmark/run_vlm_benchmark.py" ]; then
    echo "WARN: ASSETOPSBENCH_DIR=$ASSETOPSBENCH_DIR does not look like this repo; using $REPO"
    ASSETOPSBENCH_DIR="$REPO"
fi
if [ -n "$USER_PYTHON_BIN" ]; then
    PYTHON_BIN="${USER_PYTHON_BIN/#\~/$HOME}"
else
    PYTHON_BIN="$ASSETOPSBENCH_DIR/.venv/bin/python"
fi
if [ -n "$USER_VLLM_LOG" ]; then
    VLLM_LOG="${USER_VLLM_LOG/#\~/$HOME}"
else
    VLLM_LOG="$ASSETOPSBENCH_DIR/results/vllm_serve_logs/vllm_${VARIANT}.log"
fi

cd "$ASSETOPSBENCH_DIR"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: $PYTHON_BIN not found or not executable."
    echo "Run 'uv sync' in $ASSETOPSBENCH_DIR, or set PYTHON_BIN=/path/to/python."
    exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux is required on the VM."
    exit 1
fi

stop_vllm() {
    tmux kill-session -t vllm 2>/dev/null || true
    if [ "$KILL_ORPHAN_VLLM" = "1" ]; then
        pkill -u "$(id -u)" -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    fi
}

trap stop_vllm EXIT

# Resolve MODEL + EXTRA from the variants registry. The registry is the single
# source of truth, including compressed-tensors and L2 serving flags.
RESOLVED=$("$PYTHON_BIN" - "$VARIANT" <<'PY'
import sys
sys.path.insert(0, '.')
from benchmark import variants as V
v = V.get(sys.argv[1])
print(V.resolve_model_id(v.model_id))
print(' '.join(v.vllm_extra))
PY
)
MODEL_FROM_REGISTRY="$(printf '%s\n' "$RESOLVED" | sed -n '1p')"
EXTRA_FROM_REGISTRY="$(printf '%s\n' "$RESOLVED" | sed -n '2p')"

MODEL="${MODEL_OVERRIDE:-$MODEL_FROM_REGISTRY}"
EXTRA="$EXTRA_FROM_REGISTRY"
if [ -n "$QUANT_OVERRIDE" ]; then
    EXTRA="--quantization $QUANT_OVERRIDE $EXTRA"
fi

echo "==================================================================="
echo "  variant: $VARIANT"
echo "  model:   $MODEL"
echo "  extra:   ${EXTRA:-(none)}"
echo "  port:    $VLLM_PORT"
echo "  log:     $VLLM_LOG"
echo "==================================================================="

echo "==> Stopping any existing local vLLM tmux session"
stop_vllm
sleep 2
mkdir -p "$(dirname "$VLLM_LOG")"
: > "$VLLM_LOG"

SERVE_CMD=$(printf \
    'cd %q && %q -m vllm.entrypoints.openai.api_server --model %q --port %q --host 0.0.0.0 --gpu-memory-utilization %q --max-model-len %q --trust-remote-code %s 2>&1 | tee %q' \
    "$ASSETOPSBENCH_DIR" "$PYTHON_BIN" "$MODEL" "$VLLM_PORT" "$GPU_UTIL" "$MAX_MODEL_LEN" "$EXTRA" "$VLLM_LOG")

echo "==> Starting local vLLM in tmux session 'vllm'"
echo "    $SERVE_CMD"
tmux new-session -d -s vllm "$SERVE_CMD"

echo "==> Waiting for vLLM /v1/models on localhost:$VLLM_PORT (up to 8 min)..."
READY=0
for i in $(seq 1 96); do
    if curl -sf -m 5 "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
        echo "    READY (poll $i, ~$((i * 5))s)"
        READY=1
        break
    fi
    if ! tmux has-session -t vllm 2>/dev/null; then
        echo "    vLLM tmux session exited before readiness."
        echo "    --- vLLM log tail ---"
        tail -80 "$VLLM_LOG" 2>/dev/null || true
        exit 1
    fi
    sleep 5
done

if [ "$READY" != "1" ]; then
    echo "    vLLM did NOT become ready in 8 min."
    echo "    --- vLLM log tail ---"
    tail -30 "$VLLM_LOG" 2>/dev/null || true
    exit 1
fi

export VLM_BASE_URL="http://127.0.0.1:$VLLM_PORT/v1"
export VLM_API_KEY="EMPTY"
export VLM_MODEL="$MODEL"
mkdir -p results

STATUS=0

echo "==> Running benchmark for $VARIANT ..."
if ! "$PYTHON_BIN" benchmark/run_vlm_benchmark.py --variant "$VARIANT" \
    2>&1 | tee "results/bench_${VARIANT}.log"; then
    echo "    [error] benchmark failed for $VARIANT"
    STATUS=1
fi

echo "==> Collecting HPML metrics ..."
if ! "$PYTHON_BIN" benchmark/hpml_metrics.py --variant "$VARIANT" \
    --vllm-url "http://127.0.0.1:$VLLM_PORT" \
    --vm-ssh-prefix "" \
    --measurement-window-s 5; then
    echo "    [error] HPML metrics failed for $VARIANT"
    STATUS=1
fi

echo "==> Stopping vLLM"
stop_vllm
trap - EXIT

if [ "$STATUS" = "0" ]; then
    echo "==> $VARIANT done"
else
    echo "==> $VARIANT finished with errors"
fi
exit "$STATUS"
