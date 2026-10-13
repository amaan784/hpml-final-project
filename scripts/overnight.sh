#!/usr/bin/env bash
# Overnight wrapper for the assetopsbench VM: build missing AWQ checkpoints,
# run the full sweep, then run optional LLM judge and W&B summary/plots.
#
# This assumes you are already inside the VM:
#   tmux new -s overnight
#   cd ~/hpml-final-project
#   bash scripts/overnight.sh
#   # detach: Ctrl-b d
#   # resume later: tmux attach -t overnight
#
# The run logs itself to results/logs/overnight_<RUN_ID>.log (no tee needed)
# and at the end archives every log of the run -- this one, each variant's
# vLLM server log and benchmark log, sweep_status.txt, run_manifest.json --
# into results/run_logs_<RUN_ID>.tar.gz, the file to commit as evidence.
#
# Useful env vars:
#   QUANTIZE_MISSING=1  default. build missing AWQ checkpoints before sweep
#   QUANTIZE_LLAMA=1    default. set 0 to skip missing Llama AWQ builds
#   OPENAI_API_KEY      optional. else LLM judge is skipped
#   wandb login         optional. else W&B upload is skipped
#   RUN_ID              optional. defaults to the UTC start time
#
# Optional fresh-start cleanup:
#   CLEAN_RESULTS=1 bash scripts/overnight.sh

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_PYTHON_BIN="${PYTHON_BIN:-}"
ASSETOPSBENCH_DIR="${ASSETOPSBENCH_DIR:-$REPO}"
ASSETOPSBENCH_DIR="${ASSETOPSBENCH_DIR/#\~/$HOME}"
if [ ! -f "$ASSETOPSBENCH_DIR/scripts/run_full_bench.sh" ]; then
    echo "WARN: ASSETOPSBENCH_DIR=$ASSETOPSBENCH_DIR does not look like this repo; using $REPO"
    ASSETOPSBENCH_DIR="$REPO"
fi
if [ -n "$USER_PYTHON_BIN" ]; then
    PYTHON_BIN="${USER_PYTHON_BIN/#\~/$HOME}"
else
    PYTHON_BIN="$ASSETOPSBENCH_DIR/.venv/bin/python"
fi
export ASSETOPSBENCH_DIR PYTHON_BIN

cd "$ASSETOPSBENCH_DIR"
TMPDIR="${TMPDIR:-$HOME/tmp}"
TMPDIR="${TMPDIR/#\~/$HOME}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR PYTORCH_ALLOC_CONF
mkdir -p results models "$TMPDIR"

# Every run logs itself, whether or not the caller pipes through tee.
RUN_ID="${RUN_ID:-$(date -u '+%Y%m%dT%H%M%SZ')}"
OVERNIGHT_LOG="results/logs/overnight_${RUN_ID}.log"
mkdir -p results/logs
exec > >(tee -a "$OVERNIGHT_LOG") 2>&1
# OVERNIGHT_RUN tells run_full_bench.sh this script owns the run manifest.
export RUN_ID OVERNIGHT_RUN=1

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: $PYTHON_BIN not found or not executable."
    echo "Run 'uv sync' in $ASSETOPSBENCH_DIR, or set PYTHON_BIN=/path/to/python."
    exit 1
fi

START_TS=$(date '+%Y-%m-%dT%H:%M:%S')
echo "==================================================================="
echo "  OVERNIGHT START: $START_TS   (run $RUN_ID)"
echo "  log: $OVERNIGHT_LOG"
echo "==================================================================="
"$PYTHON_BIN" scripts/run_manifest.py start --run-id "$RUN_ID" || true

echo
echo ">>> Preflight: required Python packages"
"$PYTHON_BIN" - <<'PY' || exit 1
import importlib
import sys

required = [
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "vllm",
    "llmcompressor",
    "compressed_tensors",
    "qwen_vl_utils",
    "safetensors",
    "requests",
    "wandb",
    "openai",
    "matplotlib",
]

missing = []
print(f"    python: {sys.executable}")
for name in required:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "installed")
        print(f"    [ok] {name} {version}")
    except Exception as exc:
        missing.append((name, exc))
        print(f"    [missing] {name}: {exc}")

if missing:
    print("\nERROR: required Python packages are missing from the repo venv.")
    print("Install with uv pip before running overnight.")
    raise SystemExit(1)
PY

echo
echo ">>> Preflight: patch transformers offloaded save bug"
"$PYTHON_BIN" - <<'PY' || exit 1
from pathlib import Path
import sysconfig

site_paths = [sysconfig.get_paths().get(k) for k in ("purelib", "platlib")]
for base in filter(None, site_paths):
    path = Path(base) / "transformers" / "modeling_utils.py"
    if path.exists():
        break
else:
    print("ERROR: could not find transformers/modeling_utils.py")
    raise SystemExit(1)

text = path.read_text()
if "if module_map and False:" in text:
    print(f"    [ok] already patched: {path}")
elif "if module_map:" in text:
    path.write_text(text.replace("if module_map:", "if module_map and False:", 1))
    print(f"    [ok] patched: {path}")
else:
    print(f"ERROR: expected save_pretrained module_map guard not found in {path}")
    raise SystemExit(1)
PY

build_checkpoint_if_missing() {
    local name="$1"
    local out_dir="$2"
    shift 2

    checkpoint_ready() {
        local d="$1"
        [ -f "$d/config.json" ] || return 1
        [ -f "$d/tokenizer_config.json" ] || return 1
        find "$d" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.bin" \) -size +100M | grep -q .
    }

    checkpoint_status() {
        local d="$1"
        [ -f "$d/config.json" ] && echo "      config.json: OK" || echo "      config.json: MISSING"
        [ -f "$d/tokenizer_config.json" ] && echo "      tokenizer_config.json: OK" || echo "      tokenizer_config.json: MISSING"
        local weights
        weights="$(find "$d" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.bin" \) -size +100M -printf "%f " 2>/dev/null || true)"
        if [ -n "$weights" ]; then
            echo "      weights: OK ($weights)"
        else
            echo "      weights: MISSING"
        fi
    }

    if checkpoint_ready "$out_dir"; then
        echo "    [ok] $name already exists: $out_dir"
        return 0
    fi

    if [ "${QUANTIZE_MISSING:-1}" != "1" ]; then
        echo "    [missing/incomplete] $name: $out_dir"
        checkpoint_status "$out_dir"
        echo "    [skip] QUANTIZE_MISSING is not 1; benchmark variants needing this checkpoint will fail."
        return 0
    fi

    echo "    [build] $name -> $out_dir"
    rm -rf "$out_dir"
    mkdir -p "$(dirname "$out_dir")"
    TMPDIR="$TMPDIR" PYTORCH_ALLOC_CONF="$PYTORCH_ALLOC_CONF" "$@"
    if ! checkpoint_ready "$out_dir"; then
        echo "ERROR: $name build finished but checkpoint is incomplete:"
        checkpoint_status "$out_dir"
        return 1
    fi
    echo "    [ok] $name built"
}

# Record (first build) or verify (reuse) a checkpoint's SHA256 manifest and
# copy it into results/checkpoint_manifests/, so the hashes are committed --
# models/ itself is gitignored.
pin_checkpoint() {
    local d="$1"
    [ -f "$d/config.json" ] || return 0   # not built (e.g. QUANTIZE_MISSING=0)
    if [ -f "$d/SHA256SUMS" ]; then
        if ! "$PYTHON_BIN" scripts/hash_checkpoint.py verify "$d" >/dev/null; then
            echo "ERROR: $d no longer matches its SHA256SUMS manifest; refusing to"
            echo "       benchmark a modified checkpoint. Delete the directory to rebuild it."
            return 1
        fi
        echo "    [ok] hashes verified: $(basename "$d")"
    else
        "$PYTHON_BIN" scripts/hash_checkpoint.py write "$d" >/dev/null || return 1
        echo "    [ok] hashes recorded: $(basename "$d")"
    fi
    mkdir -p results/checkpoint_manifests
    cp "$d/SHA256SUMS" "results/checkpoint_manifests/$(basename "$d").SHA256SUMS"
}

echo
echo ">>> Stage 1/4: build missing AWQ checkpoints"
build_checkpoint_if_missing \
    "Qwen AWQ domain" \
    "$ASSETOPSBENCH_DIR/models/qwen2.5-vl-7b-awq-domain" \
    "$PYTHON_BIN" scripts/quantize_qwen_v010.py \
        --mode w4a16_domain \
        --pipeline sequential \
        --max-seq-len 512 \
        --num-samples 64 \
        --out-dir "$ASSETOPSBENCH_DIR/models/qwen2.5-vl-7b-awq-domain" \
    || exit 1

build_checkpoint_if_missing \
    "Qwen AWQ generic" \
    "$ASSETOPSBENCH_DIR/models/qwen2.5-vl-7b-awq-generic" \
    "$PYTHON_BIN" scripts/quantize_qwen_v010.py \
        --mode w4a16_generic \
        --pipeline sequential \
        --max-seq-len 512 \
        --num-samples 64 \
        --out-dir "$ASSETOPSBENCH_DIR/models/qwen2.5-vl-7b-awq-generic" \
    || exit 1

if [ "${QUANTIZE_LLAMA:-1}" = "1" ]; then
    build_checkpoint_if_missing \
        "Llama AWQ domain" \
        "$ASSETOPSBENCH_DIR/models/llama3-llava-next-8b-awq-domain-real" \
        "$PYTHON_BIN" scripts/quantize_llmcompressor_v010.py \
            --mode domain \
            --out-dir "$ASSETOPSBENCH_DIR/models/llama3-llava-next-8b-awq-domain-real" \
        || exit 1

    build_checkpoint_if_missing \
        "Llama AWQ generic" \
        "$ASSETOPSBENCH_DIR/models/llama3-llava-next-8b-awq-generic-real" \
        "$PYTHON_BIN" scripts/quantize_llmcompressor_v010.py \
            --mode generic \
            --out-dir "$ASSETOPSBENCH_DIR/models/llama3-llava-next-8b-awq-generic-real" \
        || exit 1
else
    echo "    [skip] QUANTIZE_LLAMA=0; missing Llama AWQ variants will fail if their checkpoints do not exist."
fi

echo
echo ">>> Stage 1b: pin checkpoints by SHA256 (manifests -> results/checkpoint_manifests/)"
for ckpt in qwen2.5-vl-7b-awq-domain qwen2.5-vl-7b-awq-generic \
            llama3-llava-next-8b-awq-domain-real llama3-llava-next-8b-awq-generic-real; do
    pin_checkpoint "$ASSETOPSBENCH_DIR/models/$ckpt" || exit 1
done

if [ "${CLEAN_RESULTS:-0}" = "1" ]; then
    echo
    echo ">>> Fresh-start cleanup"
    "$PYTHON_BIN" scripts/clean_results.py --yes --plots --tensorboard \
        || echo "    [warn] clean_results.py errored; continuing"
    rm -f results/llm_judge.csv results/REPORT_TEMPLATE.md
fi

echo
echo ">>> Stage 2/4: full benchmark sweep"
echo "    New defaults: --repeats 5 --warmup 1 in benchmark/run_vlm_benchmark.py"
echo "    Per-variant time: ~15-25 min (5 timed + 1 warmup vLLM call per scenario)"
echo "    Groups: core (10 variants, ~3-4 h) + L2 FP8 ablation (5, ~1 h)"
echo "            + L3 resolution sweep (6, ~2 h); total ~6-7 h after checkpoints exist"
echo "    Skip a group with RUN_L2_ABLATION=0 or RUN_L3_SWEEP=0."
echo "    Timing is cold-cache: vLLM's caches are cleared before every timed call."
echo "    For the legacy single-shot mode (~30-40 min total), set"
echo "    BENCH_ARGS=\"--repeats 1 --warmup 0\"."
bash scripts/run_full_bench.sh || echo "    [warn] sweep had failures; see results/sweep_status.txt"

echo
echo ">>> Stage 3/4: LLM-as-judge grading"
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "    [skip] OPENAI_API_KEY not set; skipping llm_judge."
    echo "           To grade later: export OPENAI_API_KEY=sk-... && python -m benchmark.llm_judge"
else
    "$PYTHON_BIN" -m benchmark.llm_judge \
        || echo "    [warn] llm_judge errored; partial results/llm_judge.csv may exist"
fi

echo
echo ">>> Stage 4/4: W&B summary + plots"
"$PYTHON_BIN" -m benchmark.wandb_summary \
    || echo "    [warn] wandb_summary errored; plots may still be in results/plots/"

"$PYTHON_BIN" scripts/run_manifest.py finish || true

END_TS=$(date '+%Y-%m-%dT%H:%M:%S')
echo
echo "==================================================================="
echo "  OVERNIGHT DONE: $END_TS  (started $START_TS, run $RUN_ID)"
echo "==================================================================="

# Bundle every log of this run into one committed archive; the raw logs stay
# on the VM (gitignored). This log is still open, so the archive holds it up
# to this point.
LOG_ARCHIVE="results/run_logs_${RUN_ID}.tar.gz"
ARCHIVE_FILES=()
shopt -s nullglob
for f in "$OVERNIGHT_LOG" results/vllm_serve_logs/*.log results/bench_*.log \
         results/sweep_status.txt results/run_manifest.json; do
    [ -e "$f" ] && ARCHIVE_FILES+=("$f")
done
shopt -u nullglob
if [ "${#ARCHIVE_FILES[@]}" -gt 0 ] && tar czf "$LOG_ARCHIVE" "${ARCHIVE_FILES[@]}"; then
    echo "  log archive: $LOG_ARCHIVE (${#ARCHIVE_FILES[@]} files)"
else
    echo "  [warn] could not create $LOG_ARCHIVE"
fi

echo
echo "Artifacts (commit everything under results/ except the raw logs):"
echo "  results/sweep_status.txt          per-variant PASS/FAIL"
echo "  results/summary.csv               per-scenario rows (full answers, tokens, cache state)"
echo "  results/hpml_metrics.csv          per-variant aggregate"
echo "  results/vram_breakdown.csv        weight/KV memory + engine config per variant"
echo "  results/llm_judge.csv             LLM-judge rows, every pass, if OPENAI_API_KEY was set"
echo "  results/checkpoint_manifests/     SHA256 of every quantized checkpoint"
echo "  results/run_manifest.json         git commit, GPU, driver, package versions"
echo "  $LOG_ARCHIVE   all logs of this run"
echo "  results/plots/                    generated plots"
echo "  results/REPORT_TEMPLATE.md        W&B report template"
