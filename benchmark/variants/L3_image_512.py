from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L3_image_512",
    short="L3-512",
    family="L3",
    description="Downscale images to 512px max side before sending to the VLM.",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    vllm_extra=(),
    image_max_side=512,
    requires=("Nothing extra; vlm_client honors VLM_IMAGE_MAX_SIDE at request time.",),
    how_it_works=(
        "Changes only the 11 scenario images larger than 512 px (transformers "
        "and turbines); for those, Qwen's dynamic resolution cuts visual "
        "tokens ~4x vs the 1024 px cap. Pump (512 px) and motor (320 px) "
        "requests are byte-identical to L0."
    ),
    howto_test=(
        "MODEL=Qwen/Qwen2.5-VL-7B-Instruct bash scripts/serve_vllm.sh\n"
        "python benchmark/run_vlm_benchmark.py --variant L3_image_512"
    ),
))
