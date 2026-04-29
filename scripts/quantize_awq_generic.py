# DEPRECATED. Use scripts/quantize_qwen_v010.py --mode w4a16_generic.
# Old llmcompressor 0.3.0 API. The replacement also switches calibration
# source from Open-Platypus to ultrachat_200k.

import os
from pathlib import Path

OUT_DIR = Path(os.environ.get("AWQ_OUT_DIR", "/opt/models/qwen2.5-vl-7b-awq-generic"))
NUM_SAMPLES = int(os.environ.get("AWQ_NUM_SAMPLES", "128"))
MAX_SEQ_LEN = int(os.environ.get("AWQ_MAX_SEQ_LEN", "2048"))
MODEL_ID = os.environ.get("AWQ_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
CAL_REPO = os.environ.get("AWQ_CAL_REPO", "garage-bAInd/Open-Platypus")
CAL_SPLIT = os.environ.get("AWQ_CAL_SPLIT", "train")


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForVision2Seq
    from llmcompressor.transformers import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    print(f"==> Loading {MODEL_ID} (FP16) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype="float16",
        device_map="auto",
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

    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head", "re:visual.*"],
    )

    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"==> Running oneshot quantization -> {OUT_DIR}")
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=calib_texts,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=NUM_SAMPLES,
        output_dir=str(OUT_DIR),
    )
    print("==> Done.")

if __name__ == "__main__":
    main()

