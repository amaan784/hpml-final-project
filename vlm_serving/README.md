# Step 3 - Qwen2.5-VL on vLLM + benchmark

This directory stands up **Qwen2.5-VL-7B-Instruct** behind vLLM's OpenAI-compatible HTTP endpoint and benchmarks it per optimization layer (L0 FP16 / L1 AWQ INT4 / L2 serving-tuned) of the study laid out in the project proposal.

> **Run this in Colab or on a GCP L4 instance.** It is not executed from the repo. You need a CUDA GPU with at least ~16 GB VRAM for FP16.

## Files

| File | Purpose |
|---|---|
| `requirements.txt` | Install vLLM + the OpenAI Python client. |
| `start_vllm_server.sh` | Launch the vLLM OpenAI-compatible server. |
| `test_client.py` | Smoke test - send one impeller image + one question, non-streaming. |
| `run_scenarios.py` | Run every scenario through the endpoint once and save text responses for human scoring (`qwen_vllm_results.json`). |
| `benchmark.py` | Per-scenario perf benchmark with warmup, N streaming iterations, TTFT / E2E / decode-tps measurement, and a `/metrics` snapshot. Writes CSV. |
| `summarize_runs.py` | Aggregate and diff benchmark CSVs across layers (prints mean / p50 / p95 + %-delta vs baseline). |

## Quickstart (Colab cell ordering)

```bash
# 1. Install deps
pip install -r vlm_serving/requirements.txt

# 2. Start the server in the background (log in server.log)
bash vlm_serving/start_vllm_server.sh > server.log 2>&1 &

# 3. Wait for the server to finish loading
tail -f server.log   # Ctrl-C once you see "Uvicorn running on 0.0.0.0:8000"

# 4. Smoke test
python vlm_serving/test_client.py

# 5. Functional pass for human scoring
python vlm_serving/run_scenarios.py \
    --scenarios scenarios/vlm_impeller_scenarios.json \
    --out qwen_vllm_results.json

# 6. Perf benchmark for this layer
mkdir -p benchmarks
python vlm_serving/benchmark.py \
    --scenarios scenarios/vlm_impeller_scenarios.json \
    --label L0_fp16 \
    --iters 3 --warmup 1 \
    --out benchmarks/L0_fp16.csv
```

Step 6 writes:
- `benchmarks/L0_fp16.csv` - one row per (scenario, iteration) with `ttft_s`, `e2e_s`, `decode_s`, `prompt_tokens`, `completion_tokens`, `decode_tps`.
- `benchmarks/L0_fp16.prom.txt` - raw vLLM Prometheus text (KV cache %, prefix cache hit rate, queue depth).
- `benchmarks/L0_fp16.metrics.json` - parsed numeric version of the above.

## Layer sweep (L0 -> L1 -> L2)

Kill and relaunch the server between layers, then re-run `benchmark.py` with a new `--label`. Client code does not change.

**L0 - FP16 baseline** (`start_vllm_server.sh` as-is):
```bash
python vlm_serving/benchmark.py ... --label L0_fp16 --out benchmarks/L0_fp16.csv
```

**L1 - AWQ INT4** (relaunch server against the AWQ checkpoint):
```bash
MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
VLLM_EXTRA_ARGS="--quantization awq_marlin" \
bash vlm_serving/start_vllm_server.sh > server.log 2>&1 &
# ... wait for ready ...
VLLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
python vlm_serving/benchmark.py ... --label L1_awq_int4 --out benchmarks/L1_awq_int4.csv
```

> Note: `start_vllm_server.sh` does not currently read `VLLM_EXTRA_ARGS` - either edit the script to append it, or launch vLLM directly with the extra flag for L1. Kept explicit on purpose so each layer is obvious in the run history.

**L2 - serving-tuned** (AWQ + vLLM feature flags):
Append `--enable-prefix-caching --enable-chunked-prefill --kv-cache-dtype fp8` to the launch command, then run the benchmark with `--label L2_tuned`.

## Compare layers

```bash
python vlm_serving/summarize_runs.py \
    benchmarks/L0_fp16.csv \
    benchmarks/L1_awq_int4.csv \
    benchmarks/L2_tuned.csv
```

Example output (synthetic, shows the shape):
```
Metric             L0_fp16  L1_awq_int4  vs L0_fp16
---------------------------------------------------
TTFT (s) mean      0.823    0.558        -32.2%
E2E (s) mean       3.550    2.205        -37.9%
Decode tok/s mean  23.500   38.933       +65.7%
```

Add `--markdown` to drop it straight into the report.

## What this does and doesn't measure

**Measures** (end-to-end via the OpenAI streaming API, so no vLLM internals required):
- TTFT (time from request send -> first content chunk)
- E2E latency per request
- Decode time (`e2e - ttft`) and output throughput (`completion_tokens / decode_s`)
- Prompt / completion token counts
- vLLM-level aggregates from `/metrics`: GPU cache usage %, prefix cache hit rate, queue depth

**Does not** (yet) measure:
- Per-component timing (ViT encode vs. LLM prefill vs. LLM decode) - needs PyTorch Profiler or vLLM request-level traces.
- Peak VRAM as a function of concurrency - add a load generator with `asyncio.gather` once single-request numbers are stable.
- Accuracy - human scoring lives in the notebook + `scenarios/vlm_impeller_scenarios.json`'s `expected_answer` field.

These are scoped for Phase 2 of the proposal. Single-request per-layer numbers are enough to draw the accuracy-latency Pareto curves the proposal commits to.

## Next step after this directory works

Wrap `run_scenarios.py`'s `analyze_pump_impeller(image_path, question)` function in an MCP server so MetaAgent can call it as a regular AssetOpsBench agent. That's step 4 in the roadmap, not step 3.
