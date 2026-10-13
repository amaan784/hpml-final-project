from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L2_full_bundle",
    short="L2",
    family="L2",
    description="All vLLM tuning flags combined (the headline L2 variant).",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    vllm_extra=(
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--kv-cache-dtype", "fp8",
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", "16",
    ),
    image_max_side=1024,
    requires=("Same as L2_fp8_kv (Ada-Lovelace+ for FP8).",),
    how_it_works=(
        "Stacks prefix caching + chunked prefill + FP8 KV cache + 90% GPU "
        "memory + max 16 concurrent sequences. Under vLLM 0.19's V1 engine "
        "prefix caching and chunked prefill are already on by default (L0 "
        "runs with both), so FP8 KV is the only correctness-relevant change. "
        "Ablations: L2_fp8_kv, L2_fp8_kv_dynamic_scales, "
        "L2_fp8_kv_no_prefix_cache, L2_full_bundle_dynamic_scales, and the "
        "Llama control L2_llama_fp8_kv."
    ),
    howto_test=(
        "MODEL=Qwen/Qwen2.5-VL-7B-Instruct EXTRA='--enable-prefix-caching "
            "--enable-chunked-prefill --kv-cache-dtype fp8 "
            "--gpu-memory-utilization 0.90 --max-num-seqs 16' "
            "bash scripts/serve_vllm.sh\n"
        "python benchmark/run_vlm_benchmark.py --variant L2_full_bundle"
    ),
))
