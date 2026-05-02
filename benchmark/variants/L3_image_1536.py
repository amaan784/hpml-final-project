from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L3_image_1536",
    short="L3-1536",
    family="L3",
    description="Upscale to 1536px max side (more vision tokens, more detail).",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    vllm_extra=("--max-model-len", "16384"),  # accommodate longer vision-token prefix
    image_max_side=1536,
    requires=(
        "Serve with --max-model-len 16384 (variant passes it).",
        "Need big enough source images if you want real upscaling.",
    ),
    how_it_works=(
        "1536px = more vision tokens; good for tiny defects, costs prefill time.",
    ),
    howto_test=(
        "MODEL=Qwen/Qwen2.5-VL-7B-Instruct "
            "EXTRA='--max-model-len 16384' bash scripts/serve_vllm.sh\n"
        "python benchmark/run_vlm_benchmark.py --variant L3_image_1536"
    ),
))
