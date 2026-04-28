# DEPRECATED. Use scripts/quantize_qwen_v010.py --mode w8a8_domain instead.
# Old llmcompressor 0.3.0 API + old output path qwen2.5-vl-7b-w8a8.

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from servers.vision import image_loader


def _build_calib_texts(datasets, per_dataset, processor):
    # does `build_calib_texts`, split out so we can reuse it from a few call sites.
    from datasets import load_dataset

    all_texts = []
    # each pass handles the next item in the sequence
    for alias in datasets:
        spec = image_loader.get_spec(alias)
        ds = load_dataset(spec.repo, split=spec.default_split)
        n = min(per_dataset, len(ds))
        ds = ds.select(range(n))
        prompt = (
            f"Describe the {spec.domain_hint} shown in this image."
            if spec.domain_hint else "Describe the equipment in this image."
        )
        # step through the batch one entry at a time
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
    p.add_argument("--datasets", nargs="+", default=["transformer"])
    p.add_argument("--per-dataset", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--model-id", default=os.environ.get(
        "AWQ_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct"))
    p.add_argument("--out-dir", default=os.environ.get(
        "AWQ_W8A8_OUT_DIR", "/opt/models/qwen2.5-vl-7b-w8a8"))
    args = p.parse_args()

    from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq
    from llmcompressor.transformers import oneshot
    # SmoothQuant dampens activation outliers before INT8 round-to-nearest.
    # GPTQ then handles weights.
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

    print(f"==> Loading {args.model_id} (FP16) ...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id, trust_remote_code=True, torch_dtype="float16", device_map="auto"
    )

    print(f"==> Building W8A8 calibration set: datasets={args.datasets}, "
          f"per_dataset={args.per_dataset}")
    calib_texts = _build_calib_texts(args.datasets, args.per_dataset, processor)
    print(f"==> Total calibration samples: {len(calib_texts)}")

    recipe = [
        SmoothQuantModifier(smoothing_strength=0.7),
        GPTQModifier(
            targets="Linear",
            scheme="W8A8",
            ignore=["lm_head", "re:visual.*"],
        ),
    ]

    out = Path(args.out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"==> oneshot W8A8 quantization -> {out}")
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

