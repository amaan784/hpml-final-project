from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L2_llama_full_bundle",
    short="L2-llama",
    family="L2",
    description="Llama FP16 + all vLLM tuning flags combined (the headline L2 variant).",
    model_id="llava-hf/llama3-llava-next-8b-hf",
    vllm_extra=(
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--kv-cache-dtype", "fp8",
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", "16",
    ),
    image_max_side=1024,
    requires=("Same as L2_llama_fp8_kv (Ada-Lovelace+ for FP8).",),
    how_it_works=(
        "Llama track: same flag stack as L2_full_bundle on Qwen.",
    ),
    howto_test=(
        "MODEL=llava-hf/llama3-llava-next-8b-hf EXTRA='--enable-prefix-caching "
            "--enable-chunked-prefill --kv-cache-dtype fp8 "
            "--gpu-memory-utilization 0.90 --max-num-seqs 16' "
            "bash scripts/serve_vllm.sh\n"
        "uv run python -m benchmark.run_vlm_benchmark --variant L2_llama_full_bundle"
    ),
))
