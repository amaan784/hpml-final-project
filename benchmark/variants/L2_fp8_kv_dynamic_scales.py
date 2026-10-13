from .base import Variant, register

# L2 ablation: FP8 KV cache with scales computed at runtime instead of 1.0.
# If this restores accuracy where L2_fp8_kv collapses, uncalibrated FP8
# scaling is the mechanism behind the L2 failure.


register(Variant(
    name="L2_fp8_kv_dynamic_scales",
    short="L2-fp8-dyn",
    family="L2",
    description="L2 ablation: FP8 KV cache with dynamically computed K/V scales.",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    vllm_extra=("--kv-cache-dtype", "fp8", "--calculate-kv-scales"),
    image_max_side=1024,
    requires=("Ada-Lovelace+ GPU for FP8 KV cache (the L4 is sm_89).",),
    how_it_works=(
        "--calculate-kv-scales derives K/V scales from the first real input "
        "that reaches each attention layer (vLLM PR #11906, which reported "
        "Qwen2-7B perplexity improving from 34.84 to 22.62 versus scale 1.0). "
        "This is the documented workaround for vLLM issue #41343."
    ),
    howto_test="bash scripts/serve_and_bench.sh L2_fp8_kv_dynamic_scales",
))
