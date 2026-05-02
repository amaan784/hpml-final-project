from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L1_awq_w8a8_domain",
    short="L1-W8A8",
    family="L1",
    description="Qwen2.5-VL-7B SmoothQuant + GPTQ W8A8, domain-calibrated.",
    model_id="$AWQ_W8A8",  # /opt/models/qwen2.5-vl-7b-w8a8-domain
    vllm_extra=("--quantization", "compressed-tensors"),
    image_max_side=1024,
    requires=(
        "scripts/quantize_qwen_v010.py --mode w8a8_domain has been run.",
        "Output dir /opt/models/qwen2.5-vl-7b-w8a8-domain (~7-8 GB) exists.",
    ),
    how_it_works=(
        "Half-as-aggressive precision point on the precision axis: weights AND "
        "activations are INT8 (8 bits each). Recipe: SmoothQuantModifier "
        "(smoothing_strength=0.7) first dampens activation outliers, then "
        "GPTQModifier with scheme=W8A8 quantizes weights and activations. "
        "Vision tower stays FP16 (same ignore pattern as L1d). Provides the "
        "third Pareto point so the report shows the full FP16 -> INT8 -> INT4 "
        "progression rather than just two endpoints, and isolates the "
        "weight-precision axis from the activation-precision axis."
    ),
    howto_test=(
        "python scripts/quantize_qwen_v010.py --mode w8a8_domain "
            "--out-dir /opt/models/qwen2.5-vl-7b-w8a8-domain\n"
        "bash scripts/serve_and_bench.sh L1_awq_w8a8_domain "
            "/opt/models/qwen2.5-vl-7b-w8a8-domain compressed-tensors\n"
        "# Or manually:\n"
        "MODEL=/opt/models/qwen2.5-vl-7b-w8a8-domain "
            "EXTRA='--quantization compressed-tensors' bash scripts/serve_vllm.sh\n"
        "uv run python -m benchmark.run_vlm_benchmark --variant L1_awq_w8a8_domain"
    ),
))
