from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L3_image_768",
    short="L3-768",
    family="L3",
    description="Downscale images to 768px max side.",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    vllm_extra=(),
    image_max_side=768,
    requires=("Nothing extra.",),
    how_it_works=(
        "768px midpoint on the resolution sweep.",
    ),
    howto_test=(
        "python benchmark/run_vlm_benchmark.py --variant L3_image_768"
    ),
))
