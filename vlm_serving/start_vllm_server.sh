#!/usr/bin/env bash
# Launch Qwen2.5-VL-7B-Instruct behind vLLM's OpenAI-compatible server.
# This is the L0 (FP16) baseline for the AssetOpsBench inference-optimization study.
#
# Usage:
#   bash vlm_serving/start_vllm_server.sh
#
# Env overrides (with defaults):
#   MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct
#   PORT=8000
#   MAX_MODEL_LEN=8192          # trim if VRAM is tight
#   GPU_MEMORY_UTILIZATION=0.90 # vLLM KV-cache budget
#
# Stop with Ctrl-C or `pkill -f "vllm.entrypoints.openai.api_server"`.
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

echo "Starting vLLM server"
echo "  model : ${MODEL_ID}"
echo "  port  : ${PORT}"
echo "  max-model-len : ${MAX_MODEL_LEN}"
echo "  gpu-mem-util  : ${GPU_MEMORY_UTILIZATION}"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --limit-mm-per-prompt image=1 \
    --trust-remote-code
