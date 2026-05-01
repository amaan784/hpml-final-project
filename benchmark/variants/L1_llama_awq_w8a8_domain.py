from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L1_llama_awq_w8a8_domain",
    short="L1-W8A8-llama",
    family="L1",
    description="llama3-llava-next-8b SmoothQuant + GPTQ W8A8, substation-domain calibration.",
    model_id="$AWQ_LLAMA_W8A8",  # /opt/models/llama3-llava-next-8b-w8a8
    vllm_extra=(),  # vLLM auto-detects compressed-tensors weights
    image_max_side=1024,
    requires=(
        "quantize_llmcompressor_v010 --mode w8a8_domain done.",
        "Checkpoint dir exists.",
    ),
    how_it_works=(
        "W8A8 weights+acts; vision stack FP16; middle point between FP16 and W4.",
    ),
    howto_test=(
        "python scripts/quantize_llmcompressor_v010.py --mode w8a8_domain "
            "--out-dir /opt/models/llama3-llava-next-8b-w8a8\n"
        "bash scripts/serve_and_bench.sh L1_llama_awq_w8a8_domain "
            "/opt/models/llama3-llava-next-8b-w8a8 compressed-tensors\n"
        "MODEL=/opt/models/llama3-llava-next-8b-w8a8 bash scripts/serve_vllm.sh\n"
        "uv run python -m benchmark.run_vlm_benchmark --variant L1_llama_awq_w8a8_domain"
    ),
))
