#!/usr/bin/env bash
# Full HPML benchmark sweep, run from inside the assetopsbench VM.
#
# Each variant is fully self-described in benchmark/variants/. The local
# serve_and_bench.sh wrapper resolves the model path and vLLM flags from that
# registry, starts a local tmux session named "vllm", benchmarks it, and stops
# the server before moving to the next variant.
#
# Output:
#   results/summary.csv          per-scenario rows, all variants combined
#   results/hpml_metrics.csv     per-variant VRAM / TTFT / ITL / throughput
#   results/bench_<variant>.log  raw benchmark stdout per variant
#   results/sweep_status.txt     per-variant PASS/FAIL summary
#
# Per-variant time on L4: roughly 3 min for L0/L2/L3 and 3-4 min for L1.
# Total wall-clock for all 10 active variants is usually 30-40 min once all
# checkpoints are already built.
#
# Failure tolerance: one failed variant does not abort the sweep. The status
# file at the end shows which variants need to be rerun.
#
# Useful env vars:
#   ASSETOPSBENCH_DIR  default: this repo
#   PYTHON_BIN         default: $ASSETOPSBENCH_DIR/.venv/bin/python
#   VLLM_PORT          default: 8000
#
# After this sweep finishes:
#   1. python -m benchmark.llm_judge       # grade accuracy if OPENAI_API_KEY is set
#   2. python -m benchmark.wandb_summary   # plots + cross-variant W&B dashboard

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

declare -A STATUS=()

run_one() {
    local v="$1"
    echo
    echo ">>> [$(date '+%H:%M:%S')] starting $v"
    if bash scripts/serve_and_bench.sh "$v"; then
        STATUS["$v"]=PASS
        echo ">>> [$(date '+%H:%M:%S')] $v PASS"
    else
        STATUS["$v"]=FAIL
        echo ">>> [$(date '+%H:%M:%S')] $v FAIL (continuing sweep)"
    fi
}

# Qwen track
run_one L0_baseline
run_one L1_awq_w4a16_domain
run_one L1_awq_w4a16_generic
run_one L2_full_bundle
run_one L3_image_512

# Llama track
run_one L0_llama_baseline
run_one L1_llama_awq_w4a16_domain
run_one L1_llama_awq_w4a16_generic
run_one L2_llama_full_bundle
run_one L3_llama_image_512

# Status report
mkdir -p results
{
    echo "Sweep finished: $(date)"
    echo
    printf '%-32s  %s\n' "VARIANT" "STATUS"
    printf '%-32s  %s\n' "-------" "------"
    for v in \
        L0_baseline L1_awq_w4a16_domain L1_awq_w4a16_generic L2_full_bundle L3_image_512 \
        L0_llama_baseline L1_llama_awq_w4a16_domain L1_llama_awq_w4a16_generic L2_llama_full_bundle L3_llama_image_512
    do
        printf '%-32s  %s\n' "$v" "${STATUS[$v]:-MISSING}"
    done
} | tee results/sweep_status.txt

echo
echo "==================================================================="
echo "  FULL BENCH SWEEP COMPLETE"
echo "==================================================================="
echo "Summary CSV:    results/summary.csv"
echo "HPML metrics:   results/hpml_metrics.csv"
echo "Status:         results/sweep_status.txt"
echo
column -ts, results/hpml_metrics.csv 2>/dev/null || cat results/hpml_metrics.csv 2>/dev/null || true
echo
echo "Next steps:"
echo "  1. Run LLM-as-judge to grade accuracy:"
echo "       export OPENAI_API_KEY=sk-..."
echo "       python -m benchmark.llm_judge"
echo "  2. Generate plots + cross-variant W&B dashboard:"
echo "       python -m benchmark.wandb_summary"
