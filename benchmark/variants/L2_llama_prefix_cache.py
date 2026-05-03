from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L2_llama_prefix_cache",
    short="L2-pc-llama",
    family="L2",
    description="Llama FP16 + vLLM prefix caching only (no other tuning flags).",
    model_id="llava-hf/llama3-llava-next-8b-hf",
    vllm_extra=("--enable-prefix-caching",),
    image_max_side=1024,
    requires=("vLLM 0.19+ on the L4 VM.",),
    how_it_works=(
        "Prefix cache only; Llama track counterpart to L2_prefix_cache.",
    ),
    howto_test=(
        "MODEL=llava-hf/llama3-llava-next-8b-hf "
            "EXTRA='--enable-prefix-caching' bash scripts/serve_vllm.sh\n"
        "uv run python -m benchmark.run_vlm_benchmark --variant L2_llama_prefix_cache\n"
        "# Inspect: curl http://localhost:8000/metrics | grep prefix_cache"
    ),
))
