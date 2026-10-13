from .base import Variant, register

# L2 ablation: FP8 KV cache alone. Under vLLM 0.19's V1 engine, prefix caching
# and chunked prefill are on by default -- L0 already runs with both -- so FP8 KV
# is the one correctness-relevant change in the L2 bundle. This variant tests
# whether it alone reproduces the Qwen collapse.


register(Variant(
    name="L2_fp8_kv",
    short="L2-fp8",
    family="L2",
    description="L2 ablation: FP8 KV cache alone, with vLLM's default KV scales.",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    vllm_extra=("--kv-cache-dtype", "fp8"),
    image_max_side=1024,
    requires=("Ada-Lovelace+ GPU for FP8 KV cache (the L4 is sm_89).",),
    how_it_works=(
        "Stores K/V in FP8 (fp8 = e4m3 on CUDA). The checkpoint ships no KV "
        "scales, so vLLM uses a scale of 1.0; Qwen2.5 keys have large outlier "
        "channels that a single per-tensor e4m3 scale cannot represent. vLLM "
        "issue #41343 documents FP8-KV output corruption on Qwen2-VL/Qwen2.5-VL "
        "with default scaling. Everything else matches L0."
    ),
    howto_test="bash scripts/serve_and_bench.sh L2_fp8_kv",
))
