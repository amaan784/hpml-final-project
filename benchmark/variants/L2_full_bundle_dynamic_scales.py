from .base import Variant, register

# The production L2 bundle with the FP8 scaling fix applied. Answers the
# practical question: does the bundle keep its latency win once FP8 KV
# scales are computed instead of defaulting to 1.0?


register(Variant(
    name="L2_full_bundle_dynamic_scales",
    short="L2-dyn",
    family="L2",
    description="The full L2 bundle plus dynamically computed FP8 K/V scales.",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    vllm_extra=(
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--kv-cache-dtype", "fp8",
        "--calculate-kv-scales",
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", "16",
    ),
    image_max_side=1024,
    requires=("Ada-Lovelace+ GPU for FP8 KV cache (the L4 is sm_89).",),
    how_it_works=(
        "Identical to L2_full_bundle except for --calculate-kv-scales. Compare "
        "both latency and accuracy against L2_full_bundle and L0."
    ),
    howto_test="bash scripts/serve_and_bench.sh L2_full_bundle_dynamic_scales",
))
