from .base import Variant, register

# Tiny module: just registers one Variant row for the sweep driver.


register(Variant(
    name="L2_chunked_prefill",
    short="L2-cp",
    family="L2",
    description="Add vLLM chunked prefill only.",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    vllm_extra=("--enable-chunked-prefill",),
    image_max_side=1024,
    requires=("vLLM 0.6.4.",),
    how_it_works=(
        "Without chunked prefill, a long-prompt request blocks decode tokens "
        "from running on the SM during its prefill phase. Chunked prefill "
        "splits a long prompt into smaller chunks that interleave with the "
        "decode steps of other in-flight requests, reducing tail latency "
        "(p95) under concurrency. Image-heavy VLM prompts are exactly the "
        "case this targets - the vision tower output is a long token sequence."
    ),
    howto_test=(
        "MODEL=Qwen/Qwen2.5-VL-7B-Instruct "
            "EXTRA='--enable-chunked-prefill' bash scripts/serve_vllm.sh\n"
        "python benchmark/run_vlm_benchmark.py --variant L2_chunked_prefill\n"
        "# Tail-latency view: python benchmark/concurrent_load.py --levels 1 2 4 8"
    ),
))
