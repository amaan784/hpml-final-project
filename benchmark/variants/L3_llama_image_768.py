from .base import Variant, register

# Resolution sweep point (Llama). A 768 px cap changes 11 of 22 scenario
# images (transformers and turbines).


register(Variant(
    name="L3_llama_image_768",
    short="L3-768-llama",
    family="L3",
    description="Llama FP16 with images downscaled to 768px max side.",
    model_id="llava-hf/llama3-llava-next-8b-hf",
    vllm_extra=(),
    image_max_side=768,
    requires=("Nothing extra; vlm_client honors VLM_IMAGE_MAX_SIDE at request time.",),
    how_it_works=(
        "Changes 11 of 22 scenario images (transformers and turbines). "
        "LLaVA-NeXT picks a grid of 336 px crops, so image tokens change in "
        "steps; prompt_tokens records the count."
    ),
    howto_test="bash scripts/serve_and_bench.sh L3_llama_image_768",
))
