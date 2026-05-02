from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L1_awq_w4a16_domain",
    short="L1d",
    family="L1",
    description="Qwen2.5-VL-7B AWQ W4A16 INT4, calibrated on substation/pump/turbine domain texts.",
    model_id="$AWQ_DOMAIN",
    vllm_extra=("--quantization", "compressed-tensors"),
    image_max_side=1024,
    requires=(
        "scripts/quantize_qwen_v010.py --mode w4a16_domain has been run on the VM.",
        "Output dir ~/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-domain (~5 GB) exists.",
        "vLLM 0.19+, llmcompressor 0.10.0.2, compressed-tensors 0.14.0.1.",
    ),
    how_it_works=(
        "GPTQModifier with scheme=W4A16: weights INT4, activations FP16. The "
        "visual tower (re:.*visual.*) and lm_head are kept in FP16 - Qwen2.5-VL's "
        "vision encoder collapses at INT4. Calibration uses the canonical "
        "substation-text corpus from benchmark/calibration.py (the same one "
        "Eric used for the Llama track) so the Qwen and Llama numbers come from "
        "the same calibration distribution. Pass --multi-domain to spread "
        "calibration across transformer/pumps/turbine in proportion to scenario "
        "counts. The compressed-tensors pack-quantized format is auto-detected "
        "by vLLM from the saved quantization_config; we still pass the flag "
        "explicitly so the variant is self-documenting."
    ),
    howto_test=(
        "# One-time on the VM (~25 min):\n"
        "TMPDIR=~/tmp PYTORCH_ALLOC_CONF=expandable_segments:True "
            "python scripts/quantize_qwen_v010.py --mode w4a16_domain "
            "--pipeline sequential --max-seq-len 512 --num-samples 64 "
            "--out-dir ~/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-domain\n"
        "# Serve + bench together (recommended):\n"
        "bash scripts/serve_and_bench.sh L1_awq_w4a16_domain "
            "$HOME/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-domain compressed-tensors\n"
        "# Or manually:\n"
        "MODEL=$HOME/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-domain "
            "EXTRA='--quantization compressed-tensors' bash scripts/serve_vllm.sh\n"
        "uv run python -m benchmark.run_vlm_benchmark --variant L1_awq_w4a16_domain"
    ),
))
