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
#   LOCK_GPU_CLOCK     unset by default. An SM clock in MHz ("1500") or a
#                      "MIN,MAX" pair to pin for this variant's serve+bench,
#                      removing boost/thermal jitter from e2e_ms; unlocked on
#                      exit; needs sudo. The L4's base is 795 MHz and its
#                      boost 2040 MHz under a 72 W cap: lock at the clock the
#                      card actually sustains on this workload (watch
#                      `nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv -l 1`
#                      during the pre-flight). Locking higher than the power
#                      cap allows does not stop throttling.
#   BENCH_ARGS         extra flags for run_vlm_benchmark.py, e.g.
#                      BENCH_ARGS="--limit 1 --repeats 2" for a pre-flight run
#
# vLLM is started with VLLM_SERVER_DEV_MODE=1 so the harness can clear its
# prefix/encoder caches before each timed call (cold-cache timing).

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
LOCK_GPU_CLOCK="${LOCK_GPU_CLOCK:-}"
BENCH_ARGS="${BENCH_ARGS:-}"

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

GPU_CLOCK_LOCKED=0

stop_vllm() {
    tmux kill-session -t vllm 2>/dev/null || true
    if [ "$KILL_ORPHAN_VLLM" = "1" ]; then
        pkill -u "$(id -u)" -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    fi
    if [ "$GPU_CLOCK_LOCKED" = "1" ]; then
        echo "==> Restoring default GPU clocks"
        sudo nvidia-smi -rgc -i 0 >/dev/null 2>&1 || true
        GPU_CLOCK_LOCKED=0
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

# A quantized checkpoint must still match the SHA256 manifest recorded when it
# was built (overnight.sh writes it); otherwise results would come from an
# artifact other than the one documented.
if [ -d "$MODEL" ]; then
    if [ -f "$MODEL/SHA256SUMS" ]; then
        echo "==> Verifying checkpoint hashes: $MODEL"
        if ! "$PYTHON_BIN" scripts/hash_checkpoint.py verify "$MODEL" >/dev/null; then
            echo "ERROR: $MODEL does not match its SHA256SUMS manifest; not benchmarking it."
            exit 1
        fi
    else
        echo "    WARN: $MODEL has no SHA256SUMS manifest (scripts/hash_checkpoint.py write)"
    fi
fi

echo "==> Stopping any existing local vLLM tmux session"
stop_vllm
sleep 2

if [ -n "$LOCK_GPU_CLOCK" ]; then
    # nvidia-smi -lgc takes a MIN,MAX pair; a single value locks both ends.
    case "$LOCK_GPU_CLOCK" in
        *,*) LGC_RANGE="$LOCK_GPU_CLOCK" ;;
        *)   LGC_RANGE="$LOCK_GPU_CLOCK,$LOCK_GPU_CLOCK" ;;
    esac
    echo "==> Locking GPU SM clock to ${LGC_RANGE} MHz (sudo nvidia-smi -lgc)"
    if sudo nvidia-smi -lgc "$LGC_RANGE" -i 0 >/dev/null 2>&1; then
        GPU_CLOCK_LOCKED=1
    else
        echo "    WARN: failed to lock GPU clock; continuing with default clocks"
    fi
fi

mkdir -p "$(dirname "$VLLM_LOG")"
: > "$VLLM_LOG"

# Prepend the venv's bin directory to PATH so subprocesses spawned by vLLM
# (notably flashinfer JIT calling `ninja`) can find venv-installed binaries.
VENV_BIN_DIR="$(dirname "$PYTHON_BIN")"

SERVE_CMD=$(printf \
    'cd %q && VLLM_SERVER_DEV_MODE=1 PATH=%q:$PATH %q -m vllm.entrypoints.openai.api_server --model %q --port %q --host 0.0.0.0 --gpu-memory-utilization %q --max-model-len %q --trust-remote-code %s 2>&1 | tee %q' \
    "$ASSETOPSBENCH_DIR" "$VENV_BIN_DIR" "$PYTHON_BIN" "$MODEL" "$VLLM_PORT" "$GPU_UTIL" "$MAX_MODEL_LEN" "$EXTRA" "$VLLM_LOG")

echo "==> Starting local vLLM in tmux session 'vllm'"
echo "    $SERVE_CMD"
tmux new-session -d -s vllm "$SERVE_CMD"

echo "==> Waiting for vLLM /v1/models on localhost:$VLLM_PORT (up to 15 min)..."
READY=0
for i in $(seq 1 180); do
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
    echo "    vLLM did NOT become ready in 15 min."
    echo "    --- vLLM log tail ---"
    tail -30 "$VLLM_LOG" 2>/dev/null || true
    exit 1
fi

export VLM_BASE_URL="http://127.0.0.1:$VLLM_PORT/v1"
export VLM_API_KEY="EMPTY"
export VLM_MODEL="$MODEL"
mkdir -p results

STATUS=0

echo "==> Running benchmark for $VARIANT ${BENCH_ARGS:+(extra args: $BENCH_ARGS)}..."
# shellcheck disable=SC2086  # BENCH_ARGS is intentionally word-split
if ! "$PYTHON_BIN" benchmark/run_vlm_benchmark.py --variant "$VARIANT" $BENCH_ARGS \
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

# Harvest the weight/KV-cache VRAM breakdown from the serve log NOW, while it
# still exists (clean_results.py wipes vllm_serve_logs/). Appends committed
# evidence rows to results/vram_breakdown.csv. Best-effort: a parse failure
# does not fail the variant.
echo "==> Extracting VRAM evidence from serve log ..."
"$PYTHON_BIN" scripts/vram_evidence.py parse "$VLLM_LOG" --variant "$VARIANT" \
    || echo "    [warn] VRAM evidence extraction failed for $VARIANT (non-fatal)"

echo "==> Stopping vLLM"
stop_vllm
trap - EXIT

if [ "$STATUS" = "0" ]; then
    echo "==> $VARIANT done"
else
    echo "==> $VARIANT finished with errors"
fi
exit "$STATUS"
