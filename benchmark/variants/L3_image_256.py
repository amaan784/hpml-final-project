from .base import Variant, register

# Resolution sweep point. Scenario image sizes (longest side): motors 320 px,
# pumps 512, transformers 1042, turbines 5456 -- so a 256 px cap changes all
# 22 scenario images.


register(Variant(
    name="L3_image_256",
    short="L3-256",
    family="L3",
    description="Downscale images to 256px max side before sending to the VLM.",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    vllm_extra=(),
    image_max_side=256,
    requires=("Nothing extra; vlm_client honors VLM_IMAGE_MAX_SIDE at request time.",),
    how_it_works=(
        "Changes all 22 scenario images. Qwen2.5-VL uses native dynamic "
        "resolution (about one visual token per 28x28 px), so image tokens -- "
        "logged as prompt_tokens -- fall with pixel area."
    ),
    howto_test="bash scripts/serve_and_bench.sh L3_image_256",
))
