from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L2_prefix_cache",
    short="L2-pc",
    family="L2",
    description="Add vLLM prefix caching only (FP16 model, no other tuning flags).",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    vllm_extra=("--enable-prefix-caching",),
    image_max_side=1024,
    requires=("vLLM 0.6.4 (already pinned).",),
    how_it_works=(
        "vLLM hashes the token prefix of every request and reuses the KV "
        "cache when a later request shares it. For our benchmark each "
        "specialized tool (classify_equipment / detect_visual_defects / ...) "
        "uses a fixed system-prompt-style preamble per dataset, so prefix-cache "
        "should hit on the second-and-subsequent calls per (tool x dataset) "
        "pair, eliminating prefill cost on those tokens."
    ),
    howto_test=(
        "MODEL=Qwen/Qwen2.5-VL-7B-Instruct "
            "EXTRA='--enable-prefix-caching' bash scripts/serve_vllm.sh\n"
        "python benchmark/run_vlm_benchmark.py --variant L2_prefix_cache\n"
        "# Inspect: curl http://localhost:8000/metrics | grep prefix_cache"
    ),
))
