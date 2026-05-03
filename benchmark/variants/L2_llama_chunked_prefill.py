from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L2_llama_chunked_prefill",
    short="L2-cp-llama",
    family="L2",
    description="Llama FP16 + vLLM chunked prefill only.",
    model_id="llava-hf/llama3-llava-next-8b-hf",
    vllm_extra=("--enable-chunked-prefill",),
    image_max_side=1024,
    requires=("vLLM 0.19+ on the L4 VM.",),
    how_it_works=(
        "Same as L2_chunked_prefill but on the Llama vision model.",
    ),
    howto_test=(
        "MODEL=llava-hf/llama3-llava-next-8b-hf "
            "EXTRA='--enable-chunked-prefill' bash scripts/serve_vllm.sh\n"
        "uv run python -m benchmark.run_vlm_benchmark --variant L2_llama_chunked_prefill\n"
        "# Tail-latency view: uv run python benchmark/concurrent_load.py --levels 1 2 4 8"
    ),
))
