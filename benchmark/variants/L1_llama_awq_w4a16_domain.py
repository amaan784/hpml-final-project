from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L1_llama_awq_w4a16_domain",
    short="L1d-llama",
    family="L1",
    description="llama3-llava-next-8b AWQ W4A16 INT4, calibrated on substation images.",
    model_id="$AWQ_LLAMA_DOMAIN",
    vllm_extra=("--quantization", "compressed-tensors"),
    image_max_side=1024,
    requires=(
        "quantize_llmcompressor_v010 --mode domain done; checkpoint on disk.",
        "Single L4 24GB is enough for quant + serve.",
    ),
    how_it_works=(
        "W4A16 on decoder linears; vision + projector stay FP16; domain image calib.",
    ),
    howto_test=(
        "python scripts/quantize_llmcompressor_v010.py --mode domain "
            "--out-dir ~/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-domain-real\n"
        "bash scripts/serve_and_bench.sh L1_llama_awq_w4a16_domain "
            "$HOME/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-domain-real compressed-tensors\n"
        "MODEL=$HOME/HPML-AssetOpsBench/models/llama3-llava-next-8b-awq-domain-real "
            "EXTRA='--quantization compressed-tensors' bash scripts/serve_vllm.sh\n"
        "uv run python -m benchmark.run_vlm_benchmark --variant L1_llama_awq_w4a16_domain"
    ),
))
