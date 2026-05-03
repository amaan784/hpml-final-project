from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L2_fp8_kv",
    short="L2-kv",
    family="L2",
    description="Add FP8 KV cache only.",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    vllm_extra=("--kv-cache-dtype", "fp8"),
    image_max_side=1024,
    requires=(
        "vLLM 0.6.4 + L4/L40S (FP8 needs Hopper or Ada-Lovelace+; L4 is Ada).",
    ),
    how_it_works=(
        "Each token cached during decode normally takes 2 bytes per element "
        "(FP16). Switching the KV cache to FP8 halves that to 1 byte, "
        "doubling the number of tokens that fit in the same VRAM. For our "
        "small-batch workload this mostly buys headroom for higher "
        "--max-num-seqs and longer max_model_len without OOM, indirectly "
        "raising throughput."
    ),
    howto_test=(
        "MODEL=Qwen/Qwen2.5-VL-7B-Instruct "
            "EXTRA='--kv-cache-dtype fp8' bash scripts/serve_vllm.sh\n"
        "python benchmark/run_vlm_benchmark.py --variant L2_fp8_kv\n"
        "# Sanity: curl http://localhost:8000/metrics | grep gpu_cache_usage"
    ),
))
