from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L1_llama_awq_w4a16_generic",
    short="L1g-llama",
    family="L1",
    description="llama3-llava-next-8b AWQ W4A16 INT4, GENERIC (text-only) calibration.",
    model_id="$AWQ_LLAMA_GENERIC",
    vllm_extra=("--quantization", "compressed-tensors"),
    image_max_side=1024,
    requires=(
        "scripts/quantize_llmcompressor_v010.py --mode generic has been run on the VM.",
        "Output dir ~/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-generic-real (~5 GB) exists.",
    ),
    how_it_works=(
        "Same recipe as L1_llama_awq_w4a16_domain but calibrated on 128 "
        "instruction samples from HuggingFaceH4/ultrachat_200k instead of the "
        "substation-domain text corpus. Acts as the *baseline* against which "
        "the domain-calibrated checkpoint is compared in the report. Eric's "
        "Apr 24 run surfaced a runaway-generation outlier on 1/25 scenarios "
        "(~122 s, max_tokens cap) that domain calibration never triggered - "
        "this is the project's headline reliability finding."
    ),
    howto_test=(
        "# One-time on the VM (~25 min) - Eric's modern llmcompressor 0.10 API:\n"
        "python scripts/quantize_llmcompressor_v010.py --mode generic "
            "--out-dir ~/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-generic-real\n"
        "# Serve + bench together (recommended):\n"
        "bash scripts/serve_and_bench.sh L1_llama_awq_w4a16_generic "
            "$HOME/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-generic-real compressed-tensors\n"
        "# Or manually:\n"
        "MODEL=$HOME/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-generic-real "
            "EXTRA='--quantization compressed-tensors' bash scripts/serve_vllm.sh\n"
        "uv run python -m benchmark.run_vlm_benchmark --variant L1_llama_awq_w4a16_generic"
    ),
))
