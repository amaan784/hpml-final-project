# Generic-calibration variant of quantize_awq_llama_vision.py.
# 128 instruction-tuning samples from garage-bAInd/Open-Platypus (text-only).
# Baseline for comparison vs the domain-calibrated checkpoint.
# Output: /opt/models/llama3-llava-next-8b-awq-generic.
# Pinned: vllm==0.7.3, llmcompressor==0.3.0.

import os
from pathlib import Path

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

OUT_DIR = Path(os.environ.get(
    "AWQ_OUT_DIR", "/opt/models/llama3-llava-next-8b-awq-generic"))
NUM_SAMPLES = int(os.environ.get("AWQ_NUM_SAMPLES", "128"))
MAX_SEQ_LEN = int(os.environ.get("AWQ_MAX_SEQ_LEN", "2048"))
MODEL_ID = os.environ.get(
    "AWQ_MODEL_ID", "llava-hf/llama3-llava-next-8b-hf")
CAL_REPO = os.environ.get("AWQ_CAL_REPO", "garage-bAInd/Open-Platypus")
CAL_SPLIT = os.environ.get("AWQ_CAL_SPLIT", "train")


def main():
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer, LlavaNextForConditionalGeneration
    from llmcompressor.transformers import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    print(f"==> Loading {MODEL_ID} (FP16) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype="float16",
        device_map="auto",
        # SDPA crashes inside llmcompressor's calibration for LLaVA-NeXT. eager works
        attn_implementation="eager",
    )

    print(f"==> Loading calibration data: {CAL_REPO}[{CAL_SPLIT}], {NUM_SAMPLES} samples")
    ds = load_dataset(CAL_REPO, split=CAL_SPLIT).select(range(NUM_SAMPLES))

    def to_chat(example):
        return [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example.get("output", "")},
        ]

    print("==> Tokenizing calibration samples ...")
    calib_texts = [
        tokenizer.apply_chat_template(to_chat(ex), tokenize=False)
        for ex in ds
    ]
    # wrap as HF Dataset. otherwise llmcompressor tries to interpret a list as a path
    calib_ds = Dataset.from_dict({"text": calib_texts})

    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        sequential_targets=["LlamaDecoderLayer"],
        ignore=[
            "re:.*lm_head",
            "re:.*vision_tower.*",
            "re:.*multi_modal_projector.*",
        ],
    )

    # use_cache off everywhere. calibration disables cache
    model.config.use_cache = False
    model.config.text_config.use_cache = False

    if hasattr(model, "language_model"):
        model.language_model.config.use_cache = False

    # see quantize_awq_llama_vision.py for the inner-only quantization rationale
    inner = model.language_model
    inner.config.use_cache = False

    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"==> Running oneshot quantization (language_model only) -> {OUT_DIR}")
    # transformers 4.45's Trainer requires a real output_dir path
    import tempfile as _tempfile
    _temp_outdir = _tempfile.mkdtemp(prefix="llmc_inner_")
    oneshot(
        model=inner,
        tokenizer=tokenizer,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=NUM_SAMPLES,
        output_dir=_temp_outdir,
    )
    # processor needed for save_pretrained on the wrapper
    from transformers import AutoProcessor as _AP
    processor = _AP.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"==> Saving full LlavaNext wrapper -> {OUT_DIR}")
    model.save_pretrained(str(OUT_DIR))
    processor.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print("==> Done.")

if __name__ == "__main__":
    main()

