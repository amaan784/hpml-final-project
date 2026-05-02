from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L3_llama_image_512",
    short="L3-512-llama",
    family="L3",
    description="Llama FP16 with images downscaled to 512px max side.",
    model_id="llava-hf/llama3-llava-next-8b-hf",
    vllm_extra=(),
    image_max_side=512,
    requires=("Nothing extra; vlm_client honors VLM_IMAGE_MAX_SIDE at request time.",),
    how_it_works=(
        "512px on Llama vision; same idea as L3_image_512.",
    ),
    howto_test=(
        "MODEL=llava-hf/llama3-llava-next-8b-hf bash scripts/serve_vllm.sh\n"
        "uv run python -m benchmark.run_vlm_benchmark --variant L3_llama_image_512"
    ),
))
