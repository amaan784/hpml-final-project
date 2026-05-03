#!/usr/bin/env bash
# Run full HPML benchmark sweep: L0 FP16 → L1d AWQ-domain → L1g AWQ-generic.
# Output:
#   results/summary.csv          (per-scenario rows)
#   results/hpml_metrics.csv     (per-variant VRAM/TTFT/ITL/throughput)

set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/serve_and_bench.sh L0_baseline                "llava-hf/llama3-llava-next-8b-hf"
bash scripts/serve_and_bench.sh L1_llama_awq_w4a16_domain  "/opt/models/llama3-llava-next-8b-w4a16-domain"  compressed-tensors
bash scripts/serve_and_bench.sh L1_llama_awq_w4a16_generic "/opt/models/llama3-llava-next-8b-w4a16-generic" compressed-tensors

echo
echo "==================================================================="
echo "  FULL BENCH SWEEP COMPLETE"
echo "==================================================================="
echo
echo "Summary CSV:    results/summary.csv"
echo "HPML metrics:   results/hpml_metrics.csv"
echo
column -ts, results/hpml_metrics.csv 2>/dev/null || cat results/hpml_metrics.csv
