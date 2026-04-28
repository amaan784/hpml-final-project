from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L0_llama_baseline",
    short="L0-llama",
    family="L0",
    description="FP16 llama3-llava-next-8b baseline served by vLLM (Eric's Llama track).",
    model_id="llava-hf/llama3-llava-next-8b-hf",
    vllm_extra=(),
    image_max_side=1024,
    requires=(
        "~16GB FP16 weights; fine on L4.",
    ),
    how_it_works=(
        "Llama+VLM 8B FP16 (picked over 11B for VRAM).",
    ),
    howto_test=(
        "# On the VM:\n"
        "MODEL=llava-hf/llama3-llava-next-8b-hf bash scripts/serve_vllm.sh\n"
        "# Then on the Mac (with iap_tunnel.sh open):\n"
        "python benchmark/run_vlm_benchmark.py --variant L0_llama_baseline"
    ),
))
