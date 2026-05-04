from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L2_llama_fp8_kv",
    short="L2-kv-llama",
    family="L2",
    description="Llama FP16 + FP8 KV cache only.",
    model_id="llava-hf/llama3-llava-next-8b-hf",
    vllm_extra=("--kv-cache-dtype", "fp8"),
    image_max_side=1024,
    requires=(
        "vLLM 0.19+ + L4/L40S (FP8 needs Hopper or Ada-Lovelace+; L4 is Ada).",
    ),
    how_it_works=(
        "FP8 KV only; Llama track counterpart to L2_fp8_kv.",
    ),
    howto_test=(
        "MODEL=llava-hf/llama3-llava-next-8b-hf "
            "EXTRA='--kv-cache-dtype fp8' bash scripts/serve_vllm.sh\n"
        "uv run python -m benchmark.run_vlm_benchmark --variant L2_llama_fp8_kv\n"
        "# Sanity: curl http://localhost:8000/metrics | grep gpu_cache_usage"
    ),
))
