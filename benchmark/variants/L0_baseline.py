from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L0_baseline",
    short="L0",
    family="L0",
    description="FP16 baseline served by vLLM with default flags.",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    vllm_extra=(),
    image_max_side=1024,
    requires=(
        "Qwen/Qwen2.5-VL-7B-Instruct fits in 24GB VRAM at FP16 (~15GB weights).",
    ),
    how_it_works=(
        "No optimization. The full FP16 weights are loaded by vLLM with its "
        "default scheduler (continuous batching, paged attention, no prefix "
        "cache, no FP8 KV). This is the reference point every other variant "
        "is measured against."
    ),
    howto_test=(
        "# On the VM:\n"
        "MODEL=Qwen/Qwen2.5-VL-7B-Instruct bash scripts/serve_vllm.sh\n"
        "# Then on the Mac (with iap_tunnel.sh open):\n"
        "python benchmark/run_vlm_benchmark.py --variant L0_baseline"
    ),
))
