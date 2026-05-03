# amaan_run.md — End-to-End Runbook for the Visual Inspection Agent

Operational guide for HPML Team 23. Covers the **finalized 10-variant sweep**
(5 Qwen + 5 Llama) on a single GCP L4 VM, with rich W&B integration.

> **Submission deadline: 2026-05-09.**
> **Estimated VM time: ~3h 25min (~$2.40 @ $0.71/hr).**
>
> AssetOpsBench upstream's 141 text scenarios use WatsonX as the orchestrator.
> **We don't need WatsonX.** Both the planner LLM and the vision tool hit the
> same locally-served vLLM endpoint on the GCP VM. No remote API keys required.

---

## 0. What to run, in order (the tl;dr)

```bash
# ─── On Windows: pre-flight ────────────────────────────────────────────────
python -m pytest benchmark/tests/ -v               # 17 tests should pass
python -m benchmark.variants list                  # confirm 10 variants

# ─── Bring up VM ──────────────────────────────────────────────────────────
gcloud compute instances start assetopsbench --zone=us-central1-a
gcloud compute ssh assetopsbench --tunnel-through-iap `
    --project=high-perf-ml-487201 --zone=us-central1-a `
    --ssh-flag="-L 8000:localhost:8000"

# ─── On VM (after env setup, see §2): quantize × 4 (~2h 20min) ────────────
tmux new -s quantize
cd ~/HPML-AssetOpsBench
mkdir -p ~/HPML-AssetOpsBench/models ~/tmp
TMPDIR=~/tmp PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/quantize_qwen_v010.py --mode w4a16_domain  --pipeline sequential --max-seq-len 512 --num-samples 64 --out-dir ~/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-domain
TMPDIR=~/tmp PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/quantize_qwen_v010.py --mode w4a16_generic --pipeline sequential --max-seq-len 512 --num-samples 64 --out-dir ~/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-generic
PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/quantize_llmcompressor_v010.py --mode domain  --out-dir ~/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-domain-real
PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/quantize_llmcompressor_v010.py --mode generic --out-dir ~/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-generic-real

# ─── On VM: bench × 10 (~1h) ──────────────────────────────────────────────
tmux new -s bench
bash scripts/serve_and_bench.sh L0_baseline             "Qwen/Qwen2.5-VL-7B-Instruct"
bash scripts/serve_and_bench.sh L1_awq_w4a16_domain     "$HOME/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-domain"  compressed-tensors
bash scripts/serve_and_bench.sh L1_awq_w4a16_generic    "$HOME/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-generic" compressed-tensors
bash scripts/serve_and_bench.sh L2_full_bundle          "Qwen/Qwen2.5-VL-7B-Instruct"
bash scripts/serve_and_bench.sh L3_image_512            "Qwen/Qwen2.5-VL-7B-Instruct"
bash scripts/serve_and_bench.sh L0_llama_baseline           "llava-hf/llama3-llava-next-8b-hf"
bash scripts/serve_and_bench.sh L1_llama_awq_w4a16_domain   "$HOME/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-domain-real"  compressed-tensors
bash scripts/serve_and_bench.sh L1_llama_awq_w4a16_generic  "$HOME/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-generic-real" compressed-tensors
bash scripts/serve_and_bench.sh L2_llama_full_bundle        "llava-hf/llama3-llava-next-8b-hf"
bash scripts/serve_and_bench.sh L3_llama_image_512          "llava-hf/llama3-llava-next-8b-hf"

# ─── Plots + W&B dashboard (~1 min) ───────────────────────────────────────
uv run python -m benchmark.wandb_summary --name "may7-final"
# Open the URL printed at the end → headline artifact for the report.
```

### Current one-command VM flow

The manual quantize and benchmark commands above are still valid for debugging
or re-running a single stage. For the final VM run, prefer the overnight wrapper:

```bash
cd ~/HPML-AssetOpsBench
unset ASSETOPSBENCH_DIR PYTHON_BIN VLLM_LOG
export VLLM_PORT=8001
CLEAN_RESULTS=1 bash scripts/overnight.sh 2>&1 | tee results/overnight.log
```

`scripts/overnight.sh` now does the full pipeline:

1. Preflight required Python packages in `.venv`.
2. Build any missing AWQ checkpoints under `models/`.
3. Clean local CSVs, plots, logs, and TensorBoard traces if `CLEAN_RESULTS=1`.
4. Run the 10-variant benchmark sweep.
5. Run LLM-as-judge if `OPENAI_API_KEY` is set.
6. Generate plots and W&B summary artifacts.

It is resumable at the checkpoint level. If a checkpoint already has
`config.json`, the quantization stage skips it. If no checkpoints exist, budget
about **2.5-4 hours** total. Once checkpoints exist, future clean benchmark
runs should be closer to **40-60 minutes**.

Useful toggles:

```bash
QUANTIZE_MISSING=0 CLEAN_RESULTS=1 bash scripts/overnight.sh  # skip checkpoint builds
QUANTIZE_LLAMA=0 CLEAN_RESULTS=1 bash scripts/overnight.sh    # skip missing Llama builds
```

Before leaving it overnight, verify the Qwen save fix is present:

```bash
grep -n "output_dir=_tmp\|tempfile" scripts/quantize_qwen_v010.py || echo "OK: no internal oneshot save"
```

Expected:

```text
OK: no internal oneshot save
```

---

## 1. The 10 active variants

| # | Variant | Family | Model | Optimization | Output dir |
|---|---|---|---|---|---|
| 1 | `L0_baseline`                  | L0 | Qwen2.5-VL-7B  | FP16 baseline | (HF cache) |
| 2 | `L1_awq_w4a16_domain`          | L1 | Qwen2.5-VL-7B  | INT4 W4A16, substation calibration | `~/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-domain` |
| 3 | `L1_awq_w4a16_generic`         | L1 | Qwen2.5-VL-7B  | INT4 W4A16, ultrachat calibration  | `~/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-generic` |
| 4 | `L2_full_bundle`               | L2 | Qwen2.5-VL-7B  | FP16 + prefix cache + chunked prefill + FP8 KV + GPU 90% + max-num-seqs 16 | (HF cache) |
| 5 | `L3_image_512`                 | L3 | Qwen2.5-VL-7B  | FP16, images downscaled to 512 px | (HF cache) |
| 6 | `L0_llama_baseline`            | L0 | Llama-3-LLaVA-NeXT-8B | FP16 baseline | (HF cache) |
| 7 | `L1_llama_awq_w4a16_domain`    | L1 | Llama-3-LLaVA-NeXT-8B | INT4 W4A16, substation calibration | `~/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-domain-real` |
| 8 | `L1_llama_awq_w4a16_generic`   | L1 | Llama-3-LLaVA-NeXT-8B | INT4 W4A16, ultrachat calibration  | `~/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-generic-real` |
| 9 | `L2_llama_full_bundle`         | L2 | Llama-3-LLaVA-NeXT-8B | Same L2 stack as #4 on Llama | (HF cache) |
| 10 | `L3_llama_image_512`          | L3 | Llama-3-LLaVA-NeXT-8B | Llama at 512 px | (HF cache) |

**Cut variants** (parked in `_unused/variants/`, restorable if needed):
W8A8 (both families), L2 individual ablations (prefix_cache / chunked_prefill / fp8_kv on both
families), L3 image_768 / image_1024 / image_1536. See "Why we cut these" below.

---

## 2. VM environment setup (one-time, ~15 min)

After SSH'ing into the VM with `--ssh-flag="-L 8000:localhost:8000"`:

```bash
# Pull latest fork
cd ~ && rm -rf HPML-AssetOpsBench
git clone https://github.com/amaan784/AssetOpsBench.git HPML-AssetOpsBench
cd HPML-AssetOpsBench

# Base deps (uv-managed)
uv sync --group vision

# GPU stack (matches Eric's pin set)
uv pip install --python ./.venv/bin/python --torch-backend=cu129 \
    "vllm==0.19.0" "llmcompressor==0.10.0.2" "compressed-tensors==0.14.0.1" \
    "transformers==4.57.6" "accelerate>=1.0" "openai>=1.40" "pillow>=10.0" \
    "matplotlib>=3.8" "wandb>=0.17"

# Verify
./.venv/bin/python -c "
import vllm, transformers, llmcompressor, torch
print(f'vllm={vllm.__version__}  transformers={transformers.__version__}  llmcompressor={llmcompressor.__version__}')
print(f'cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0)}')
"

# Auth
huggingface-cli login
wandb login

# Pre-pull both model weights (~10 min, ~31 GB)
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
huggingface-cli download llava-hf/llama3-llava-next-8b-hf

# Pin W&B project name
export WANDB_PROJECT=hpml-assetopsbench-vlm
echo "export WANDB_PROJECT=hpml-assetopsbench-vlm" >> ~/.bashrc

# Verify ≥100 GB free disk
# Keep generated checkpoints in the repo-local models folder and temp files in home
mkdir -p ~/HPML-AssetOpsBench/models ~/tmp
export AWQ_DOMAIN="$HOME/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-domain"
export AWQ_GENERIC="$HOME/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-generic"
export AWQ_LLAMA_DOMAIN="$HOME/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-domain-real"
export AWQ_LLAMA_GENERIC="$HOME/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-generic-real"
cat >> ~/.bashrc <<'EOF'
export AWQ_DOMAIN="$HOME/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-domain"
export AWQ_GENERIC="$HOME/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-generic"
export AWQ_LLAMA_DOMAIN="$HOME/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-domain-real"
export AWQ_LLAMA_GENERIC="$HOME/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-generic-real"
EOF

# Verify home filesystem has enough free disk
df -h /
df -h ~
```

---

## 3. Quantize the 4 INT4 checkpoints (~2h 20min in tmux)

```bash
tmux new -s quantize
cd ~/HPML-AssetOpsBench
mkdir -p ~/HPML-AssetOpsBench/models ~/tmp

# Qwen W4A16 domain on L4 (~25-60 min)
TMPDIR=~/tmp PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/quantize_qwen_v010.py \
    --mode w4a16_domain \
    --pipeline sequential \
    --max-seq-len 512 \
    --num-samples 64 \
    --out-dir ~/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-domain

# Qwen W4A16 generic on L4 (~25-60 min)
TMPDIR=~/tmp PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/quantize_qwen_v010.py \
    --mode w4a16_generic \
    --pipeline sequential \
    --max-seq-len 512 \
    --num-samples 64 \
    --out-dir ~/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-generic

# Llama W4A16 domain (~35 min)
PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/quantize_llmcompressor_v010.py --mode domain \
    --out-dir ~/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-domain-real

# Llama W4A16 generic (~35 min)
PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/quantize_llmcompressor_v010.py --mode generic \
    --out-dir ~/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-generic-real

# Detach: Ctrl-b d
# Reattach: tmux attach -t quantize
```

Both quantize scripts:
- Use the modern llmcompressor 0.10 API (`from llmcompressor import oneshot`)
- Set `PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce CUDA fragmentation
- Save final checkpoints under the repo-local `~/HPML-AssetOpsBench/models`, not `/opt/models`; `/opt` is not writable for the normal VM user.
- Qwen uses `--pipeline sequential --max-seq-len 512 --num-samples 64` for L4 headroom. The script also patches the Qwen2.5-VL attention path so llmcompressor's FX tracer does not fail on multimodal RoPE.
- Save weights via `dispatch_model` + `remove_hook_from_module` + `save_pretrained(save_compressed=True)` (the offload-dance fix from Eric's bug catalog)
- Log a metadata-only W&B Artifact at the end (config.json + size manifest, NOT the 5-9 GB safetensors)

Verify all 4 checkpoints written:
```bash
for d in "$HOME/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-domain" \
         "$HOME/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-generic" \
         "$HOME/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-domain-real" \
         "$HOME/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-generic-real"; do
    test -f "$d/config.json" && echo "OK   $d" || echo "FAIL $d"
done
```

If a Qwen quantize OOMs (defaults should prevent it):
```bash
TMPDIR=~/tmp PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/quantize_qwen_v010.py \
    --mode w4a16_domain \
    --pipeline sequential \
    --max-seq-len 512 \
    --num-samples 64 \
    --out-dir ~/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-domain
```

---

## 4. Run the 10 benchmarks (~1h)

`serve_and_bench.sh` per variant: kill old vLLM → start new → wait ready →
run 30 scenarios → scrape `hpml_metrics` → kill vLLM. Each variant takes
~5-7 min. Auto-publishes to W&B on completion (URL printed at end).

```bash
tmux new -s bench
cd ~/HPML-AssetOpsBench

# ── Qwen track (5 variants) ─────────────────────────────────────────────
bash scripts/serve_and_bench.sh L0_baseline             "Qwen/Qwen2.5-VL-7B-Instruct"
bash scripts/serve_and_bench.sh L1_awq_w4a16_domain     "$HOME/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-domain"  compressed-tensors
bash scripts/serve_and_bench.sh L1_awq_w4a16_generic    "$HOME/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-generic" compressed-tensors
bash scripts/serve_and_bench.sh L2_full_bundle          "Qwen/Qwen2.5-VL-7B-Instruct"
bash scripts/serve_and_bench.sh L3_image_512            "Qwen/Qwen2.5-VL-7B-Instruct"

# ── Llama track (5 variants) ────────────────────────────────────────────
bash scripts/serve_and_bench.sh L0_llama_baseline           "llava-hf/llama3-llava-next-8b-hf"
bash scripts/serve_and_bench.sh L1_llama_awq_w4a16_domain   "$HOME/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-domain-real"  compressed-tensors
bash scripts/serve_and_bench.sh L1_llama_awq_w4a16_generic  "$HOME/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-generic-real" compressed-tensors
bash scripts/serve_and_bench.sh L2_llama_full_bundle        "llava-hf/llama3-llava-next-8b-hf"
bash scripts/serve_and_bench.sh L3_llama_image_512          "llava-hf/llama3-llava-next-8b-hf"
```

Each variant produces:
- Per-scenario rows appended to `results/summary.csv`
- Per-variant aggregate row appended to `results/hpml_metrics.csv`
- Bench log at `results/bench_<variant>.log`
- **Two W&B runs**: one `vlm_benchmark` run (per-scenario table + scalars) and one `hpml_metrics` run (TTFT/ITL/VRAM aggregate)

---

## 4a. LLM-as-judge accuracy scoring (~5 min, ~$0.05, REQUIRED for accuracy plots)

**The auto-scorer was retired 2026-05-07** (parked in `_unused/scoring/auto_scorer.py`).
LLM-as-judge is now the **only** accuracy source. `summary.csv` no longer has
a `correct` column; `results/llm_judge.csv` carries the per-scenario
1-5 score + binary pass after you run this step.

GPT-4o-mini reads each `(question, characteristic_form rubric,
model_response)` triplet and grades 1-5 against the rubric. Reproducible,
citable methodology (IndustryEQA paper does this), no team-scheduling
overhead.

**Cost: ~$0.05 for the full sweep.** No team time required. Anywhere with
network + an OpenAI key works (run on the VM or your laptop).

**This step is required to populate the accuracy-dependent plots
(`pareto`, `accuracy_per_category`, `calibration_compare`, `family_compare`,
`accuracy_compare`).** Without it those plots gracefully skip; only the
latency / VRAM / TTFT plots render.

```bash
# One-time: get an API key from https://platform.openai.com/api-keys
export OPENAI_API_KEY=sk-...

# Smoke test on 3 rows first (~$0.001)
uv run python -m benchmark.llm_judge --limit 3
# Expected: prints score=N pass=0/1 per row, then per-variant accuracy summary

# Full sweep (~5 min, ~$0.05)
uv run python -m benchmark.llm_judge

# Output: results/llm_judge.csv (columns: variant, scenario_id, llm_judge_score,
# llm_judge_pass, llm_judge_reasoning, judge_model, judge_ts, error)
```

The script is **resumable** — re-running skips rows already in the output CSV
unless you pass `--force`. Useful if the network drops mid-sweep or if you
re-run only one variant.

```bash
# Re-grade just one variant (e.g. after re-running its benchmark)
uv run python -m benchmark.llm_judge --variant L1_awq_w4a16_domain --force

# Use the more expensive but stronger judge (~$1.20 instead of ~$0.05)
uv run python -m benchmark.llm_judge --model gpt-4o
```

After this step, `wandb_summary` automatically picks up `llm_judge.csv` and
populates the `accuracy` + `mean_score` columns in the `variants_summary`
Table, plus the per-variant `__llm_score` + `__llm_pass` columns in
`scenarios_side_by_side`. The 5 accuracy-dependent plots also start
rendering.

---

## 5. Generate the dashboard (~1 min)

After all 10 benchmarks land (and optionally after `llm_judge`):

```bash
uv run python -m benchmark.wandb_summary --name "may7-final"
```

This single command does five things:

1. **Regenerates all 8 plots** in `results/plots/` (PDF + 300 DPI PNG, IEEE single-column)
2. **Builds `variants_summary` Table** — one row per variant (auto-scorer accuracy, LLM-judge accuracy, latency, VRAM, throughput)
3. **Builds `scenarios_side_by_side` Table** — one row per scenario, columns per variant (auto correct + LLM score + LLM pass + e2e_ms + response). **This is the table teammates use for rubric grading.**
4. **Uploads everything to W&B** as a single `cross_variant_summary` run with the two Tables + 8 plots pinned
5. **Writes `results/REPORT_TEMPLATE.md`** with auto-filled headline numbers (Qwen INT4 speedup, Llama INT4 speedup) and section structure for the W&B Report

The W&B URL printed at the end is your headline artifact — link it in the IEEE paper as a footnote.

---

## 6. ReAct vs Plan-Execute (NFR comparison, ~20 min, optional)

```bash
# vLLM should still be serving the L0_baseline from §4. If not, restart:
tmux new -s vllm
MODEL=Qwen/Qwen2.5-VL-7B-Instruct bash scripts/serve_vllm.sh   # Ctrl-b d

cd ~/HPML-AssetOpsBench

uv run python -m benchmark.run_agent_benchmark \
    --agent react \
    --scenarios src/scenarios/local/vision_pump_scenarios.json \
    --scenarios src/scenarios/local/vision_transformer_scenarios.json \
    --scenarios src/scenarios/local/vision_turbine_scenarios.json \
    --output results/nfr_react.csv

uv run python -m benchmark.run_agent_benchmark \
    --agent plan_execute \
    --scenarios src/scenarios/local/vision_pump_scenarios.json \
    --scenarios src/scenarios/local/vision_transformer_scenarios.json \
    --scenarios src/scenarios/local/vision_turbine_scenarios.json \
    --output results/nfr_plan_execute.csv

uv run python -m benchmark.compare_agents \
    results/nfr_react.csv results/nfr_plan_execute.csv \
    --output results/comparison_table.md \
    --plot-dir results/plots/agents
```

---

## 7. Profilers (~15 min, one-time, optional)

```bash
# PyTorch Profiler — vision tower in isolation
uv run python benchmark/profile_vision_encoder.py
# Output: ./tb/  (TensorBoard trace; viewable with `tensorboard --logdir tb`)

# Nsight Systems — single E2E request with NVTX ranges
sudo apt install -y nvidia-nsight-systems-cli || true
nsys profile --trace=cuda,nvtx,osrt --output=results/nsys_l0 \
    uv run python benchmark/profile_single.py \
    --image hf://transformer/train/0 \
    --prompt "What equipment is shown in this image?"
# Output: results/nsys_l0.qdrep — open in Nsight Systems UI on Windows
```

---

## 8. Re-running variants without polluting CSVs

`run_vlm_benchmark.py` and `hpml_metrics.py` open their CSVs in **append
mode**. Re-running a variant doubles its rows; plots and W&B aggregates
get contaminated.

Use [scripts/clean_results.py](scripts/clean_results.py):

```bash
# Dry-run (default — shows what WOULD be deleted)
python scripts/clean_results.py

# Wipe everything (preserves results/eric/, results/madhav/, etc.)
python scripts/clean_results.py --yes

# Wipe rows for ONE variant (preserves other variants' rows)
python scripts/clean_results.py --yes --variants L1_awq_w4a16_domain

# Also clear regenerated outputs
python scripts/clean_results.py --yes --plots --tensorboard
```

Typical re-run flow for one variant:

```bash
python scripts/clean_results.py --yes --variants L1_awq_w4a16_domain
bash scripts/serve_and_bench.sh L1_awq_w4a16_domain "$HOME/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-domain" compressed-tensors
uv run python -m benchmark.wandb_summary
```

W&B server runs are NOT touched. Delete from the W&B UI (project page → bulk-delete runs) for a fully clean dashboard.

---

## 9. End-of-day cleanup

```bash
# On VM
tmux kill-server
exit

# On Windows
gcloud compute instances stop assetopsbench --zone=us-central1-a

# Resume later:
gcloud compute instances start assetopsbench --zone=us-central1-a
```

**Don't `terraform destroy`** unless you're truly done — that nukes
the VM disk, including `~/HPML-AssetOpsBench/models/...` quantized checkpoints and installed packages.

---

## 10. W&B integration overview

The harness logs **three job types** of W&B runs per variant:

| Job type | Source | What |
|---|---|---|
| `vlm_benchmark` | `run_vlm_benchmark.py` → `wandb_logger.log_variant_run()` | One per variant. Per-scenario table + per-step scalars + summary metrics (accuracy, e2e_mean/p50/p95/p99) + vLLM Prometheus deltas + GPU system metrics auto-collected at 2s rate. Tags: `family`, `model_family`, `variant_short`, `variant`. |
| `hpml_metrics` | `hpml_metrics.py` → `_maybe_log_wandb()` | One per variant. Aggregate row: TTFT p50/p95, ITL p50, throughput, VRAM, KV cache. |
| `quantize` | `quantize_*.py` → `wandb_logger.log_checkpoint_artifact()` | One per quantization. Logs the checkpoint as a metadata-only W&B Artifact (config.json + size manifest, NOT the 5-9 GB safetensors). |
| `cross_variant_summary` | `wandb_summary.py` | One per dashboard build. The `variants_summary` Table + `scenarios_side_by_side` Table + 7 plots + Report template. |

To skip W&B for one run: `WANDB_DISABLED=true uv run python ...`

To use a different W&B project: `export WANDB_PROJECT=my-other-project`

---

## 11. Repo layout (after cleanup, 2026-05-07)

```
HPML-AssetOpsBench-eric/
├── _unused/                   # Everything cut from the active sweep, preserved
│   ├── variants/              # 11 cut variants (W8A8, L2 ablations, extra L3)
│   ├── scripts/               # 11 deprecated quantize + stage scripts
│   ├── benchmark/             # cods_track1, cods_track2, Dockerfile, requirements.txt
│   └── repo/                  # aobench, aaaiwebsite, notebook
├── benchmark/
│   ├── variants/              # 10 active variants + base.py + __init__/main
│   ├── tests/test_variants.py # 17 tests
│   ├── calibration.py         # Substation domain corpus + ultrachat loader
│   ├── plots.py               # 7 IEEE-quality plots from CSVs
│   ├── wandb_logger.py        # Centralised W&B helpers (used by harness + summary)
│   ├── wandb_summary.py       # Cross-variant dashboard + Report template
│   ├── run_vlm_benchmark.py   # Main harness (per-scenario, calls wandb_logger)
│   ├── hpml_metrics.py        # Prometheus aggregator (logs to W&B)
│   ├── nfr_collector.py       # NFR context manager (used by run_agent_benchmark)
│   ├── compare_agents.py      # ReAct vs Plan-Execute markdown table
│   ├── concurrent_load.py     # Tail-latency stress test (optional)
│   ├── profile_single.py      # Nsight Systems wrapper
│   ├── profile_vision_encoder.py  # PyTorch Profiler (vision tower)
│   └── run_agent_benchmark.py # ReAct / Plan-Execute NFR runner
├── scripts/
│   ├── quantize_qwen_v010.py            # Qwen W4A16 (modern llmcompressor 0.10 API)
│   ├── quantize_llmcompressor_v010.py   # Llama W4A16 (Eric's, supports w8a8_domain too)
│   ├── clean_results.py                 # CSV row scrubber + dry-run mode
│   ├── verify_checkpoint.py             # Post-quantize format check
│   ├── serve_vllm.sh                    # vLLM launcher
│   ├── serve_and_bench.sh               # Per-variant: serve + bench + metrics + kill
│   ├── run_full_bench.sh                # Full sweep wrapper (Eric's, Llama-only)
│   └── iap_tunnel.sh                    # GCP IAP SSH tunnel
├── data/
│   ├── pumps/{defective,fine}/*.jpeg    # 5 images (Amaan)
│   ├── transformer/*.jpg                # 20 images (Eric)
│   └── turbine/{defective,normal}/*.jpg # 10 images (Eric)
├── src/
│   ├── agent/                # ReAct + Plan-Execute orchestrators (our re-impl)
│   ├── llm/                  # LLMBackend abstractions (LiteLLM, vLLM)
│   ├── servers/vision/       # 5-tool vision MCP server
│   ├── servers/{iot,fmsr,...}/  # Upstream MCP servers (kept for PR; unused by us)
│   ├── scenarios/local/      # vision_{pump,transformer,turbine}_scenarios.json
│   └── tmp/agent_hive/       # Upstream AgentHive (kept as reference, unused)
├── results/
│   ├── eric/                 # Eric's Apr 24 Llama runs (preserved, immutable)
│   ├── summary.csv           # Per-scenario rows, all variants (will populate after sweep)
│   ├── hpml_metrics.csv      # Per-variant aggregates
│   ├── plots/                # 7 PNGs + 7 PDFs (regenerated by plots.py)
│   └── REPORT_TEMPLATE.md    # Auto-generated W&B Report skeleton (after wandb_summary)
├── terraform/                # GCP L4 VM provisioning
├── report/                   # main.tex + references.bib (paper draft)
├── presentation/OUTLINE.md   # Slides outline
├── proposal_edits.md         # What proposal said vs what we did (drives report)
├── benchmark_explained.md    # Eval methodology framing (auto-scorer + rubric)
└── amaan_run.md              # ← this file
```

---

## 12. Why we cut variants (for the report)

- **W8A8 (both families)** — INT8 vs INT4 difference is below auto-scorer noise on N=30. Highest OOM risk at quantize time. **Cost > benefit.** Scoped to future work.
- **L2 individual ablations (prefix_cache / chunked_prefill / fp8_kv on both)** — All three are concurrency-targeted; on N=30 sequential single-batch traffic, they show ~no measurable difference from L0. Keep `L2_full_bundle` as the production-stack Pareto point.
- **L3 image_768 / image_1024 / image_1536** — `L3_image_1024` is functionally identical to L0_baseline (same FP16, same default resolution). Keeping just `L3_image_512` gives the directional answer (half-resolution → faster + accuracy tradeoff). 768/1536 add Pareto curvature but eat VM budget.

All cut variants live in `_unused/variants/` — restore by `mv` back to `benchmark/variants/`.

---

## 13. Time + cost summary

| Phase | L4 time | Other cost |
|---|---:|---:|
| VM bring-up + env setup (one-time) | ~15 min | — |
| Quantize × 4 (Qwen w4a16-d/g + Llama w4a16-d/g) | ~2h 20min | — |
| Bench × 10 (5 Qwen + 5 Llama) | ~1h 00min | — |
| LLM-judge accuracy scoring (optional) | ~5 min (network, not GPU) | ~$0.05 OpenAI |
| Plots + W&B summary | ~5 min | — |
| **L4 GPU total** | **~3h 25min** | |
| **L4 GPU cost @ $0.71/hr** | **~$2.40** | |

If a quantize OOMs and you need to debug: budget +30 min slack.

---

## 14. Common errors and fixes

### `ModuleNotFoundError: No module named 'benchmark'`

Cause: `python benchmark/script.py` doesn't put repo root on `sys.path`.
Fix: use `python -m benchmark.script` (no `.py`, dotted path).

### Qwen W4A16 quantize OOMs with 1.43 GiB Hessian error

Use the lower-memory L4 command. `--pipeline sequential` reduces calibration state, and `512/64` cuts activation/Hessian pressure:
```bash
TMPDIR=~/tmp PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/quantize_qwen_v010.py \
  --mode w4a16_domain \
  --pipeline sequential \
  --max-seq-len 512 \
  --num-samples 64 \
  --out-dir ~/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-domain
```

### Qwen W4A16 quantize fails with `Proxy object cannot be iterated` in `apply_multimodal_rotary_pos_emb`

This usually means the VM is running an old `scripts/quantize_qwen_v010.py`. Pull latest on `main`, then confirm the CLI has `--pipeline`:

```bash
cd ~/HPML-AssetOpsBench
git checkout main
git pull --ff-only
python scripts/quantize_qwen_v010.py --help | grep pipeline
python -c "import llmcompressor, vllm, transformers; print(llmcompressor.__version__, vllm.__version__, transformers.__version__)"
```

Expected versions: `0.10.0.2 0.19.0 4.57.6`. Then rerun the L4-safe sequential command from the OOM section.

### `PermissionError: [Errno 13] Permission denied: '/opt/models/...'`

The normal VM user cannot write to `/opt/models`. Save checkpoints in the repo-local models folder:

```bash
mkdir -p ~/HPML-AssetOpsBench/models ~/tmp
TMPDIR=~/tmp PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/quantize_qwen_v010.py \
  --mode w4a16_domain \
  --pipeline sequential \
  --max-seq-len 512 \
  --num-samples 64 \
  --out-dir ~/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-domain
```

### Qwen quantize save fails with `visual.patch_embed.proj.weight`

This happens after quantization completes, during checkpoint save. It means
Transformers 4.57's offloaded-save branch is still active and crashes on a
parameter-only vision attribute. Current scripts patch
`transformers/modeling_utils.py` before importing Transformers, replacing
`if module_map:` with `if module_map and False:`. Qwen also clears stale
`hf_device_map` after `dispatch_model` + `remove_hook_from_module`, then calls
`save_pretrained(save_compressed=True)`.

Verify the fix:

```bash
grep -n "if module_map and False" .venv/lib/python3.12/site-packages/transformers/modeling_utils.py
grep -n "hf_device_map" scripts/quantize_qwen_v010.py
grep -n "output_dir=_tmp\|tempfile" scripts/quantize_qwen_v010.py || echo "OK: no internal oneshot save"
```

Expected:

```text
... if module_map and False ...
... hf_device_map ...
OK: no internal oneshot save
```

### `ModuleNotFoundError: No module named 'vllm'` or `qwen_vl_utils`

Install packages with `uv pip`, not plain `pip`, because this VM venv may not
include `pip`:

```bash
cd ~/HPML-AssetOpsBench
uv pip install --python .venv/bin/python \
  "vllm==0.19.0" \
  "llmcompressor==0.10.0.2" \
  "compressed-tensors==0.14.0.1" \
  qwen-vl-utils
```

### `ModuleNotFoundError: No module named 'matplotlib'`

The benchmark and LLM judge already finished; only Stage 4 plots/W&B summary
failed. Install plotting support, then rerun just the summary step:

```bash
cd ~/HPML-AssetOpsBench
uv pip install --python .venv/bin/python "matplotlib>=3.8"
.venv/bin/python -m benchmark.wandb_summary
```

### Llama AWQ vLLM fails with `qkv_proj.weight`

The Llama AWQ checkpoint saved, but vLLM cannot load it if
`quantization_config.ignore` was expanded into individual vision-tower keys.
Repair the existing checkpoint configs in place; no re-quantization needed:

```bash
python - <<'PY'
import json
from pathlib import Path

ignore = ["lm_head", "re:.*vision_tower.*", "re:.*multi_modal_projector.*"]
for d in [
    "models/llama3-llava-next-8b-awq-domain-real",
    "models/llama3-llava-next-8b-awq-generic-real",
]:
    p = Path(d) / "config.json"
    cfg = json.loads(p.read_text())
    for qc in [cfg.get("quantization_config"), cfg.get("text_config", {}).get("quantization_config")]:
        if isinstance(qc, dict):
            qc["ignore"] = ignore
    p.write_text(json.dumps(cfg, indent=2) + "\n")
    print(d, "patched")
PY
```

### All variants finish in seconds

That means vLLM failed before readiness and the sweep skipped through failures.
Check the per-variant serve logs:

```bash
cat results/sweep_status.txt
grep -n "vLLM tmux session exited\|did NOT become ready\|ERROR\|Traceback" results/overnight.log
find results -maxdepth 3 -path '*vllm*' -type f -print -exec tail -80 {} \;
```

Current scripts write vLLM logs under:

```text
results/vllm_serve_logs/vllm_<variant>.log
```

### vLLM log under `/tmp` is stale or owned by another user

Old scripts wrote `/tmp/vllm_<variant>.log`, which can be owned by another VM
user. Current `serve_and_bench.sh` writes to `results/vllm_serve_logs/`.

Verify:

```bash
grep -n "vllm_serve_logs" scripts/serve_and_bench.sh
```

### `ASSETOPSBENCH_DIR=/opt/assetopsbench`

Unset stale overrides before running from the VM repo:

```bash
unset ASSETOPSBENCH_DIR PYTHON_BIN VLLM_LOG
export VLLM_PORT=8001
```

### Old vLLM still hogging GPU on next run

```bash
nvidia-smi                                # find PID
tmux kill-session -t vllm 2>/dev/null     # gentle
pkill -9 -f "vllm.entrypoints.openai" 2>/dev/null  # hard kill
```

### `vllm: KeyError: 'qkv_proj.weight'` or `image_newline` on save

Toolchain bug Eric documented. Handled in `quantize_*.py` via the `dispatch_model` + `remove_hook_from_module` dance. If you see it, you're on an old script — pull latest.

### TTFT/ITL plot shows flat 1000ms / 2500ms bars

`hpml_metrics.py` measurement window had no traffic — scraped empty histograms whose default bucket boundaries are 1000 ms / 2500 ms. Re-run `serve_and_bench.sh` (it keeps vLLM busy long enough), or extend `--measurement-window-s` in `hpml_metrics.py`. The plot annotates this case with a red warning box.

### `uv: command not found` (Windows)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### W&B not logging despite `wandb login` success

Check `WANDB_DISABLED` env var:
```bash
echo $WANDB_DISABLED   # should be empty
unset WANDB_DISABLED
```

---

## 15. Accuracy methodology

We use **LLM-as-judge** (GPT-4o-mini grading against `characteristic_form`
rubric) as the canonical accuracy source — implemented in
`benchmark/llm_judge.py`. The substring auto-scorer that used to live in
`run_vlm_benchmark.py` was retired 2026-05-07 because it was too permissive
("if response contains 'reject' anywhere, mark correct" → false positives
when the primary verdict was "accept"). Auto-scorer code preserved at
`_unused/scoring/auto_scorer.py` for reference; restorable if needed.

The research signal lives across three axes (not just accuracy):

1. **Accuracy** — LLM-judge pass rate (score ≥ 4) and mean 1-5 score
   per variant. ~$0.05 for the full sweep, reproducible, no team
   grading session needed.
2. **Latency** (Pareto + calibration_compare plots). The ~2× speedup with
   AWQ INT4 and the calibration finding (runaway-generation under generic).
3. **VRAM** (vram_breakdown plot). 2.6× weight reduction → 4.7× more KV
   cache headroom on the same L4.

For an even stronger accuracy claim, schedule a team grading session
against the `scenarios_side_by_side` Table that `wandb_summary` publishes
— it shows N=30 scenarios × 10 variants in one view, with LLM-judge
score+pass per cell, ideal for sanity-checking the LLM-judge verdicts
or grading scenarios where the LLM-judge disagrees with itself across
variants.

---

## 16. Key file pointers

| What | Where |
|---|---|
| Variant registry | `benchmark/variants/` (10 active files + base.py) |
| Calibration corpus | `benchmark/calibration.py` |
| Qwen quantize | `scripts/quantize_qwen_v010.py` |
| Llama quantize | `scripts/quantize_llmcompressor_v010.py` |
| Per-scenario harness | `benchmark/run_vlm_benchmark.py` |
| Per-variant aggregate | `benchmark/hpml_metrics.py` |
| LLM-as-judge accuracy scorer | `benchmark/llm_judge.py` |
| W&B logger (centralised) | `benchmark/wandb_logger.py` |
| Cross-variant dashboard | `benchmark/wandb_summary.py` |
| Plots (8 IEEE figures) | `benchmark/plots.py` |
| Result cleanup | `scripts/clean_results.py` |
| NFR collector | `benchmark/nfr_collector.py` |
| Vision MCP server | `src/servers/vision/main.py` |
| Image loader / dataset registry | `src/servers/vision/image_loader.py` |
| Scenarios | `src/scenarios/local/vision_*.json` |
| Eric's prior results | `results/eric/HPML_REPORT.md` |
| Proposal deviations (drives report) | `proposal_edits.md` |
| Eval methodology | `benchmark_explained.md` |
| GCP infra | `terraform/` |

---

## 17. Pre-launch checklist

Before kicking off the VM sweep, walk through this list:

- [ ] `git status` is clean on Windows + VM
- [ ] `python -m pytest benchmark/tests/ -v` → 17/17 pass on Windows
- [ ] `python -m benchmark.variants list` → 10 variants
- [ ] VM started, SSH session open with `-L 8000:localhost:8000`
- [ ] `nvidia-smi` shows GPU clear (<500 MiB used by other processes)
- [ ] `df -h ~` shows enough free space for checkpoints
- [ ] `huggingface-cli login` and `wandb login` both done on VM
- [ ] Both base models cached: `ls ~/.cache/huggingface/hub/` shows Qwen + Llama
- [ ] tmux session created for quantize OR you accept babysitting the SSH connection
- [ ] `WANDB_PROJECT=hpml-assetopsbench-vlm` exported in shell

You're ready. Open the W&B project URL in a browser tab, kick off the
quantize loop, and watch runs land in real time.
