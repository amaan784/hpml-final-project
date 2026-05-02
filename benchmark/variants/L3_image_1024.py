from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L3_image_1024",
    short="L3-1024",
    family="L3",
    description="Default resolution (current vlm_client default).",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    vllm_extra=(),
    image_max_side=1024,
    requires=("Nothing extra.",),
    how_it_works=(
        "Same 1024 cap as the L0/L1/L2 runs; anchor for the L3 chart.",
    ),
    howto_test=(
        "python benchmark/run_vlm_benchmark.py --variant L3_image_1024"
    ),
))
