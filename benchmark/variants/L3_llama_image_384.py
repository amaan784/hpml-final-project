from .base import Variant, register

# Resolution sweep point (Llama). A 384 px cap changes 16 of 22 scenario images.


register(Variant(
    name="L3_llama_image_384",
    short="L3-384-llama",
    family="L3",
    description="Llama FP16 with images downscaled to 384px max side.",
    model_id="llava-hf/llama3-llava-next-8b-hf",
    vllm_extra=(),
    image_max_side=384,
    requires=("Nothing extra; vlm_client honors VLM_IMAGE_MAX_SIDE at request time.",),
    how_it_works=(
        "Changes 16 of 22 scenario images. LLaVA-NeXT picks a grid of 336 px "
        "crops, so image tokens change in steps; prompt_tokens records the count."
    ),
    howto_test="bash scripts/serve_and_bench.sh L3_llama_image_384",
))
