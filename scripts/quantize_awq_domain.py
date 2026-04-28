# DEPRECATED. Use scripts/quantize_qwen_v010.py --mode w4a16_domain instead.
# Targets the old pin stack (vllm==0.7.3, llmcompressor==0.3.0). kept for
# git history. The L1_awq_w4a16_domain variant now points at the new path,
# so checkpoints from this script require a manual env override.

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from servers.vision import image_loader


def _build_calib_texts(datasets, per_dataset, processor):
    # build chat-templated calibration text from N image datasets
    from datasets import load_dataset

    all_texts = []
    # repeat for every element we need to touch
    for alias in datasets:
        spec = image_loader.get_spec(alias)
        ds = load_dataset(spec.repo, split=spec.default_split)
        n = min(per_dataset, len(ds))
        ds = ds.select(range(n))
        prompt = (
            f"Describe the {spec.domain_hint} shown in this image."
            if spec.domain_hint else "Describe the equipment in this image."
        )
        # repeat for every element we need to touch
        for ex in ds:
            img = ex.get(spec.image_col)
            if img is None:
                continue
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }]
            all_texts.append(processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            ))
        print(f"   + {alias}: {n} samples ({spec.repo})")
    return all_texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["transformer"],
                   help="Dataset aliases (see src/servers/vision/image_loader.py).")
    p.add_argument("--per-dataset", type=int, default=128,
                   help="Calibration samples per dataset.")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--model-id", default=os.environ.get(
        "AWQ_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct"))
    p.add_argument("--out-dir", default=os.environ.get(
        "AWQ_OUT_DIR", "/opt/models/qwen2.5-vl-7b-awq-domain"))
    args = p.parse_args()

    from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq
    from llmcompressor.transformers import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    print(f"==> Loading {args.model_id} (FP16) ...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id, trust_remote_code=True, torch_dtype="float16", device_map="auto"
    )

    print(f"==> Building domain calibration set: datasets={args.datasets}, "
          f"per_dataset={args.per_dataset}")
    calib_texts = _build_calib_texts(args.datasets, args.per_dataset, processor)
    print(f"==> Total calibration samples: {len(calib_texts)}")

    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head", "re:visual.*"],
    )

    out = Path(args.out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"==> oneshot quantization -> {out}")
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=calib_texts,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=len(calib_texts),
        output_dir=str(out),
    )
    print("==> Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

