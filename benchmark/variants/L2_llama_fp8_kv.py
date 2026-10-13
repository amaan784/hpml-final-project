from .base import Variant, register

# Cross-family control for the L2 ablation: FP8 KV cache alone on Llama,
# whose full L2 bundle did not collapse. Together with L2_fp8_kv this shows
# whether the FP8 failure is specific to the Qwen family.


register(Variant(
    name="L2_llama_fp8_kv",
    short="L2-fp8-llama",
    family="L2",
    description="L2 ablation control: Llama FP16 with FP8 KV cache alone.",
    model_id="llava-hf/llama3-llava-next-8b-hf",
    vllm_extra=("--kv-cache-dtype", "fp8"),
    image_max_side=1024,
    requires=("Ada-Lovelace+ GPU for FP8 KV cache (the L4 is sm_89).",),
    how_it_works=(
        "Same FP8 KV setting as L2_fp8_kv, on Llama-3-LLaVA-NeXT-8B. Llama-3 "
        "has no QKV bias terms (Qwen2.5 does), and its full L2 bundle stayed "
        "coherent, so FP8 KV is expected to hold up here."
    ),
    howto_test="bash scripts/serve_and_bench.sh L2_llama_fp8_kv",
))
