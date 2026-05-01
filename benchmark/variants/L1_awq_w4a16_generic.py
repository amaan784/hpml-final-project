from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L1_awq_w4a16_generic",
    short="L1g",
    family="L1",
    description="Qwen2.5-VL-7B AWQ W4A16 INT4, calibrated on ultrachat_200k (generic).",
    model_id="$AWQ_GENERIC",
    vllm_extra=("--quantization", "compressed-tensors"),
    image_max_side=1024,
    requires=(
        "quantize_qwen_v010 --mode w4a16_generic done; checkpoint on disk.",
    ),
    how_it_works=(
        "Same W4A16 setup as domain L1 but ultrachat calib instead of domain text.",
    ),
    howto_test=(
        "TMPDIR=~/tmp PYTORCH_ALLOC_CONF=expandable_segments:True "
            "python scripts/quantize_qwen_v010.py --mode w4a16_generic "
            "--pipeline sequential --max-seq-len 512 --num-samples 64 "
            "--out-dir ~/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-generic\n"
        "bash scripts/serve_and_bench.sh L1_awq_w4a16_generic "
            "$HOME/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-generic compressed-tensors\n"
        "# Or manually:\n"
        "MODEL=$HOME/HPML-AssetOpsBench/models/qwen2.5-vl-7b-awq-generic "
            "EXTRA='--quantization compressed-tensors' bash scripts/serve_vllm.sh\n"
        "uv run python -m benchmark.run_vlm_benchmark --variant L1_awq_w4a16_generic"
    ),
))
