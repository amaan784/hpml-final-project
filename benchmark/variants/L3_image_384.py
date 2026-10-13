from .base import Variant, register

# Resolution sweep point. Scenario image sizes (longest side): motors 320 px,
# pumps 512, transformers 1042, turbines 5456 -- so a 384 px cap changes 16
# of 22 images (the motor images are already smaller).


register(Variant(
    name="L3_image_384",
    short="L3-384",
    family="L3",
    description="Downscale images to 384px max side before sending to the VLM.",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    vllm_extra=(),
    image_max_side=384,
    requires=("Nothing extra; vlm_client honors VLM_IMAGE_MAX_SIDE at request time.",),
    how_it_works=(
        "Changes 16 of 22 scenario images (pumps, transformers, turbines). "
        "Qwen2.5-VL uses native dynamic resolution, so image tokens fall with "
        "pixel area."
    ),
    howto_test="bash scripts/serve_and_bench.sh L3_image_384",
))
