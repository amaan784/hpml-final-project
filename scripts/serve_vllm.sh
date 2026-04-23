#!/usr/bin/env bash
# Run on the GCP VM. Launches vLLM's OpenAI-compatible server.
# Usage: MODEL=Qwen/Qwen2.5-VL-7B-Instruct bash scripts/serve_vllm.sh
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${PORT:-8000}"

echo "==> Starting vLLM model=$MODEL port=$PORT"

cd /opt/assetopsbench
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --trust-remote-code
