from .base import Variant, register

# L2 ablation: FP8 KV cache with prefix caching explicitly OFF. Tests the
# alternative explanation that the collapse comes from the image-input /
# prefix-cache path rather than FP8 numerics. Prefix caching is the only other
# bundle feature that can be disabled under vLLM's V1 engine (chunked prefill
# is always on there).


register(Variant(
    name="L2_fp8_kv_no_prefix_cache",
    short="L2-fp8-nopc",
    family="L2",
    description="L2 ablation: FP8 KV cache with prefix caching disabled.",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    vllm_extra=("--kv-cache-dtype", "fp8", "--no-enable-prefix-caching"),
    image_max_side=1024,
    requires=("Ada-Lovelace+ GPU for FP8 KV cache (the L4 is sm_89).",),
    how_it_works=(
        "Same as L2_fp8_kv but with prefix caching off. If this still collapses, "
        "the prefix-cache / image-hashing path is ruled out as the trigger."
    ),
    howto_test="bash scripts/serve_and_bench.sh L2_fp8_kv_no_prefix_cache",
))
