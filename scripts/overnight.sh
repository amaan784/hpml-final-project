#!/usr/bin/env bash
# Overnight wrapper for the assetopsbench VM: build missing AWQ checkpoints,
# run the full sweep, then run optional LLM judge and W&B summary/plots.
#
# This assumes you are already inside the VM:
#   tmux new -s overnight
#   cd ~/HPML-AssetOpsBench
#   bash scripts/overnight.sh 2>&1 | tee results/overnight.log
#   # detach: Ctrl-b d
#   # resume later: tmux attach -t overnight
#
# Useful env vars:
#   QUANTIZE_MISSING=1  default. build missing AWQ checkpoints before sweep
#   QUANTIZE_LLAMA=1    default. set 0 to skip missing Llama AWQ builds
#   OPENAI_API_KEY      optional. else LLM judge is skipped
#   wandb login         optional. else W&B upload is skipped
#
# Optional fresh-start cleanup:
#   CLEAN_RESULTS=1 bash scripts/overnight.sh 2>&1 | tee results/overnight.log

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

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: $PYTHON_BIN not found or not executable."
    echo "Run 'uv sync' in $ASSETOPSBENCH_DIR, or set PYTHON_BIN=/path/to/python."
    exit 1
fi

START_TS=$(date '+%Y-%m-%dT%H:%M:%S')
echo "==================================================================="
echo "  OVERNIGHT START: $START_TS"
echo "==================================================================="

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

if [ "${CLEAN_RESULTS:-0}" = "1" ]; then
    echo
    echo ">>> Fresh-start cleanup"
    "$PYTHON_BIN" scripts/clean_results.py --yes --plots --tensorboard \
        || echo "    [warn] clean_results.py errored; continuing"
    rm -f results/llm_judge.csv results/REPORT_TEMPLATE.md
fi

echo
echo ">>> Stage 2/4: full benchmark sweep (~30-40 min on L4 after checkpoints exist)"
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

END_TS=$(date '+%Y-%m-%dT%H:%M:%S')
echo
echo "==================================================================="
echo "  OVERNIGHT DONE: $END_TS  (started $START_TS)"
echo "==================================================================="
echo
echo "Artifacts:"
echo "  results/sweep_status.txt    per-variant PASS/FAIL"
echo "  results/summary.csv         per-scenario rows"
echo "  results/hpml_metrics.csv    per-variant aggregate"
echo "  results/llm_judge.csv       LLM-judge accuracy, if OPENAI_API_KEY was set"
echo "  results/plots/              generated plots"
echo "  results/REPORT_TEMPLATE.md  W&B report template"
echo "  results/overnight.log       full script output, if you used tee"
