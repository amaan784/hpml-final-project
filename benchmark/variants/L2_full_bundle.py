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
        "memory + max 16 concurrent sequences. The per-flag ablation variants "
        "(L2_prefix_cache, L2_chunked_prefill, L2_fp8_kv) measure each "
        "contribution in isolation; this bundle is the production L2 "
        "configuration reported in the headline results table."
    ),
    howto_test=(
        "MODEL=Qwen/Qwen2.5-VL-7B-Instruct EXTRA='--enable-prefix-caching "
            "--enable-chunked-prefill --kv-cache-dtype fp8 "
            "--gpu-memory-utilization 0.90 --max-num-seqs 16' "
            "bash scripts/serve_vllm.sh\n"
        "python benchmark/run_vlm_benchmark.py --variant L2_full_bundle"
    ),
))
