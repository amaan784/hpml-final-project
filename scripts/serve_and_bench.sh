#!/usr/bin/env bash
# Serve a single variant via vLLM 0.19, wait ready, run benchmark, collect HPML
# metrics. Designed to be called per-variant by run_full_bench.sh.
#
# Usage:
#   bash scripts/serve_and_bench.sh L0_baseline llava-hf/llama3-llava-next-8b-hf
#   bash scripts/serve_and_bench.sh L1_llama_awq_w4a16_domain /opt/models/llama3-llava-next-8b-w4a16-domain compressed-tensors

set -euo pipefail

VARIANT="${1:?variant name required}"
MODEL="${2:?model id or path required}"
QUANT="${3:-}"

VM_PROJECT=high-perf-ml-487201
VM_ZONE=us-central1-a
VM_ACCT=yc4670@columbia.edu
VM_NAME=assetopsbench
VLLM_PORT=8000
GCP_SSH=(gcloud compute ssh "$VM_NAME" --tunnel-through-iap
         --project="$VM_PROJECT" --zone="$VM_ZONE" --account="$VM_ACCT")

echo "==================================================================="
echo "  variant: $VARIANT  |  model: $MODEL  |  quant: ${QUANT:-FP16}"
echo "==================================================================="

# 1. Stop any existing vLLM tmux on VM
"${GCP_SSH[@]}" --command="tmux kill-session -t vllm 2>/dev/null || true; sleep 2"

# 2. Start vLLM in tmux on VM. Using sh -c with single quotes to avoid escape issues.
EXTRA=""
if [ -n "$QUANT" ]; then
    EXTRA="--quantization $QUANT"
fi
SERVE_CMD="cd /opt/assetopsbench && /opt/assetopsbench/.venv/bin/python -m vllm.entrypoints.openai.api_server --model $MODEL --port $VLLM_PORT --host 0.0.0.0 --gpu-memory-utilization 0.85 --max-model-len 4096 --trust-remote-code $EXTRA 2>&1 | tee /tmp/vllm_${VARIANT}.log"
echo "==> Starting vLLM: $SERVE_CMD"
"${GCP_SSH[@]}" --command="tmux new -d -s vllm '$SERVE_CMD'"

# 3. Wait for vLLM ready (max 8 min. bigger model = longer load)
echo "==> Waiting for vLLM /v1/models on localhost:$VLLM_PORT (up to 8 min)..."
READY=0
for i in $(seq 1 96); do
    if curl -sf -m 5 "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
        echo "    READY (poll $i, ~$((i*5))s)"
        READY=1
        break
    fi
    sleep 5
done
if [ "$READY" != "1" ]; then
    echo "    vLLM did NOT become ready in 8 min."
    echo "    --- vLLM log tail ---"
    "${GCP_SSH[@]}" --command="tail -30 /tmp/vllm_${VARIANT}.log"
    "${GCP_SSH[@]}" --command="tmux kill-session -t vllm 2>/dev/null || true"
    exit 1
fi

# 4. Run benchmark from Mac
echo "==> Running benchmark for $VARIANT ..."
cd "$(dirname "$0")/.."
export VLM_BASE_URL="http://localhost:$VLLM_PORT/v1"
export VLM_API_KEY="EMPTY"
export VLM_MODEL="$MODEL"
export WANDB_DISABLED=true
mkdir -p results
uv run python benchmark/run_vlm_benchmark.py --variant "$VARIANT" \
    --scenarios src/scenarios/local/vision_pump_scenarios.json \
    --scenarios src/scenarios/local/vision_transformer_scenarios.json \
    2>&1 | tee "results/bench_${VARIANT}.log" || true

# 5. Collect HPML metrics
echo "==> Collecting HPML metrics ..."
uv run python benchmark/hpml_metrics.py --variant "$VARIANT" --vllm-url "http://localhost:$VLLM_PORT" --measurement-window-s 5 || true

# 6. Stop vLLM
echo "==> Stopping vLLM"
"${GCP_SSH[@]}" --command="tmux kill-session -t vllm 2>/dev/null || true"

echo "==> $VARIANT done"
