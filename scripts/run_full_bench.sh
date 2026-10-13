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
# Variant groups (run in this order, so the core results land first):
#   core          10 variants, 5 per family (the original sweep)
#   L2 ablation    5 variants isolating the Qwen L2 FP8-KV collapse
#   L3 sweep       6 variants, 256/384/768 px on both families
#
# Per-variant time on L4 with the default --repeats 5 --warmup 1: roughly
# 15-25 min (collapsed L2 variants finish faster). Core ~3-4 h, L2 ablation
# ~1 h, L3 sweep ~2 h, once all checkpoints are built.
#
# Failure tolerance: one failed variant does not abort the sweep. The status
# file at the end shows which variants need to be rerun.
#
# Useful env vars:
#   ASSETOPSBENCH_DIR  default: this repo
#   PYTHON_BIN         default: $ASSETOPSBENCH_DIR/.venv/bin/python
#   VLLM_PORT          default: 8000
#   RUN_L2_ABLATION    default 1; set 0 to skip the L2 FP8 ablation group
#   RUN_L3_SWEEP       default 1; set 0 to skip the L3 resolution sweep group
#   LOCK_GPU_CLOCK     consumed by scripts/serve_and_bench.sh (this sweep's
#                      serving path); pins the SM clock (MHz, or "MIN,MAX")
#                      for each variant's vLLM process so e2e_ms is comparable
#                      run-to-run. Use the clock the L4 sustains under load
#                      (base 795, boost 2040 MHz, 72 W cap); see that script.
#                      Requires sudo on the VM. Omit to use default clocks.
#
# After this sweep finishes:
#   1. python -m benchmark.llm_judge       # grade accuracy if OPENAI_API_KEY is set
#   2. python -m benchmark.wandb_summary   # plots + cross-variant W&B dashboard

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Record the run manifest (git commit, GPU, package versions) unless
# overnight.sh is driving this sweep and already did.
MANIFEST_PY="${PYTHON_BIN:-$PWD/.venv/bin/python}"
if [ "${OVERNIGHT_RUN:-0}" != "1" ]; then
    "$MANIFEST_PY" scripts/run_manifest.py start || true
fi

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

CORE_VARIANTS=(
    # Qwen track
    L0_baseline L1_awq_w4a16_domain L1_awq_w4a16_generic L2_full_bundle L3_image_512
    # Llama track
    L0_llama_baseline L1_llama_awq_w4a16_domain L1_llama_awq_w4a16_generic
    L2_llama_full_bundle L3_llama_image_512
)
# Isolates the Qwen L2 collapse: FP8 KV alone, with dynamic scales, without
# prefix caching, the bundle with the fix, and a Llama FP8 control.
L2_ABLATION_VARIANTS=(
    L2_fp8_kv L2_fp8_kv_dynamic_scales L2_fp8_kv_no_prefix_cache
    L2_full_bundle_dynamic_scales L2_llama_fp8_kv
)
# With L0 (1024 px cap) and L3_image_512 this gives a 5-point resolution curve.
L3_SWEEP_VARIANTS=(
    L3_image_256 L3_image_384 L3_image_768
    L3_llama_image_256 L3_llama_image_384 L3_llama_image_768
)

VARIANTS=("${CORE_VARIANTS[@]}")
if [ "${RUN_L2_ABLATION:-1}" = "1" ]; then
    VARIANTS+=("${L2_ABLATION_VARIANTS[@]}")
else
    echo ">>> RUN_L2_ABLATION=0: skipping the L2 ablation group"
fi
if [ "${RUN_L3_SWEEP:-1}" = "1" ]; then
    VARIANTS+=("${L3_SWEEP_VARIANTS[@]}")
else
    echo ">>> RUN_L3_SWEEP=0: skipping the L3 resolution sweep group"
fi
echo ">>> Sweep plan: ${#VARIANTS[@]} variants"

for v in "${VARIANTS[@]}"; do
    run_one "$v"
done

# Status report
mkdir -p results
{
    echo "Sweep finished: $(date)"
    echo
    printf '%-32s  %s\n' "VARIANT" "STATUS"
    printf '%-32s  %s\n' "-------" "------"
    for v in "${VARIANTS[@]}"; do
        printf '%-32s  %s\n' "$v" "${STATUS[$v]:-MISSING}"
    done
} | tee results/sweep_status.txt

if [ "${OVERNIGHT_RUN:-0}" != "1" ]; then
    "$MANIFEST_PY" scripts/run_manifest.py finish || true
fi

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
