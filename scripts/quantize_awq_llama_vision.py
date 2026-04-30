# AWQ INT4 quantization of llama3-llava-next-8b with DOMAIN calibration.
# llava-hf/llama3-llava-next-8b-hf (Llama-3-8B + LLaVA-NeXT, ~16 GB FP16)
# fits on a single L4 24GB. Llama-3.2-11B-Vision does not.
# Notes vs the Qwen variant:
#  - model class is LlavaNextForConditionalGeneration
#  - kept FP16 modules are vision_tower.* and multi_modal_projector.*
#    (Qwen's are visual.*)
#  - sequential_targets is LlamaDecoderLayer
#  - repo is public (no HF gated-access)
# Output: /opt/models/llama3-llava-next-8b-awq-domain (~5 GB).
# Pinned: vllm==0.7.3, llmcompressor==0.3.0.

import argparse
import os
import sys
from pathlib import Path

# llmcompressor's oneshot pulls in wandb at import. disable BEFORE the
# transformers/llmcompressor imports so we don't crash on missing wandb auth.
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from servers.vision import image_loader


def _build_calib_texts(datasets, per_dataset, processor):
    # Compose chat-template strings describing each teammate's HF shard.
    from datasets import load_dataset

    all_texts = []
    for alias in datasets:
        spec = image_loader.get_spec(alias)
        ds = load_dataset(spec.repo, split=spec.default_split)
        n = min(per_dataset, len(ds))
        ds = ds.select(range(n))
        prompt = (
            f"Describe the {spec.domain_hint} shown in this image."
            if spec.domain_hint else "Describe the equipment in this image."
        )
        # Each HF row yields one templated calibration string carrying both pixels + caption.
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
                   help="Dataset aliases for calibration.")
    p.add_argument("--per-dataset", type=int, default=128,
                   help="Calibration samples per dataset.")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--model-id", default=os.environ.get(
        "AWQ_MODEL_ID", "llava-hf/llama3-llava-next-8b-hf"))
    p.add_argument("--out-dir", default=os.environ.get(
        "AWQ_OUT_DIR", "/opt/models/llama3-llava-next-8b-awq-domain"))
    args = p.parse_args()

    from datasets import Dataset
    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        LlavaNextForConditionalGeneration,
    )
    from llmcompressor.transformers import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    print(f"==> Loading {args.model_id} (FP16) ...")
    # Snapshot the multimodal processor/tokenizer pairing before we mutate weights.
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype="float16",
        device_map="auto",
        # SDPA attention path crashes inside llmcompressor's calibration step
        # for LLaVA-NeXT. eager works.
        attn_implementation="eager",
    )

    print(f"==> Building domain calibration set: datasets={args.datasets}, "
          f"per_dataset={args.per_dataset}")
    calib_texts = _build_calib_texts(args.datasets, args.per_dataset, processor)
    print(f"==> Total calibration samples: {len(calib_texts)}")
    # HF Dataset ctor keeps llmcompressor off the flaky "path=list" branch.
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

    # Propagate KV cache disable onto inner Llama when the wrapper nests it.
    if hasattr(model, "language_model"):
        model.language_model.config.use_cache = False

    #  pass llmcompressor JUST the inner LlamaForCausalLM.
    inner = model.language_model
    inner.config.use_cache = False

    out = Path(args.out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"==> oneshot quantization (language_model only) -> {out}")
    
    # transformers 4.45's Trainer needs output_dir to be a real path. use a
    # throwaway temp dir and re-save the full wrapper afterwards.
    import tempfile
    _temp_outdir = tempfile.mkdtemp(prefix="llmc_inner_")
    oneshot(
        model=inner,
        tokenizer=tokenizer,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=len(calib_texts),
        output_dir=_temp_outdir,
    )

    print(f"==> Saving full LlavaNext wrapper (with quantized language_model) -> {out}")
    model.save_pretrained(str(out))
    processor.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print("==> Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

