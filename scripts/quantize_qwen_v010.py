# Qwen2.5-VL W4A16 via llmcompressor 0.10 -> compressed-tensors dump for vLLM 0.19+.
# Modes: w4a16_domain (domain text), w4a16_generic (ultrachat).

import argparse
import os
import sys
from pathlib import Path

# kill wandb chatter from llmcompressor
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

# L4 24GB: reduce allocator fragmentation during GPTQ
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _build_dataset(mode, num_samples):
    from datasets import Dataset
    from benchmark.calibration import build_domain_corpus, build_generic_corpus

    if mode == "w4a16_domain":
        texts = build_domain_corpus(num_samples=num_samples)
        print(f"==> DOMAIN calibration: {len(texts)} substation texts")
    elif mode == "w4a16_generic":
        texts = build_generic_corpus(num_samples=num_samples)
        print(f"==> GENERIC calibration: {len(texts)} ultrachat samples")
    else:
        raise ValueError(f"Unknown mode {mode!r}")
    return Dataset.from_dict({"text": texts})


def _build_recipe():
    # W4A16. keep vision FP16. layer-level sequential target for this model class
    from llmcompressor.modifiers.quantization import GPTQModifier

    return GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        sequential_targets=["Qwen2_5_VLDecoderLayer"],
        ignore=["lm_head", "re:visual.*", "re:model.visual.*"],
    )


def _save_with_offload_dance(model, processor, tokenizer, out_dir):
    # strip offload hooks then save_compressed=True (else you get fp16 dequant files)
    print("   dispatch_model: consolidating offloaded modules ...")
    # isolate errors so the rest of the call can bail cleanly
    try:
        from compressed_tensors.offload import dispatch_model
    except ImportError:
        from accelerate import dispatch_model
    dispatch_model(model)

    print("   remove_hook_from_module: stripping accelerate offload hooks ...")
    from accelerate.hooks import remove_hook_from_module
    remove_hook_from_module(model, recurse=True)

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, save_compressed=True)
    print("   save_pretrained completed (compression hooks fired -> packed INT4)")

    tokenizer.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["w4a16_domain", "w4a16_generic"], required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--model-id",
        default=os.environ.get("AWQ_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct"),
    )
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=2048)
    args = p.parse_args()

    import torch  # forces CUDA init before transformers imports
    from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq
    from llmcompressor import oneshot

    print(f"==> Loading {args.model_id} (FP16) ...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        dtype="float16",
        device_map="auto",
        attn_implementation="eager",  # avoid RoPE/SDPA mismatch in calibration
    )
    model.config.use_cache = False

    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    calib_ds = _build_dataset(args.mode, args.num_samples)
    recipe = _build_recipe()

    print(f"==> oneshot quantization (W4A16, sequential_targets=Qwen2_5_VLDecoderLayer) -> {args.out_dir}")
    import tempfile
    _tmp = tempfile.mkdtemp(prefix=f"llmc_{args.mode}_")
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=len(calib_ds),
        output_dir=_tmp,
    )

    print(f"==> Saving full Qwen2.5-VL wrapper (with quantized LM) -> {args.out_dir}")
    _save_with_offload_dance(model, processor, tokenizer, args.out_dir)

    import json
    cfg_path = Path(args.out_dir) / "config.json"
    qc = {}

    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        qc = cfg.get("quantization_config", {})
        print(
            f"   config.json quantization_config: present={bool(qc)} "
            f"method={qc.get('quant_method', 'n/a')} "
            f"format={qc.get('format', 'n/a')}"
        )

    # optional wandb artifact (configs only)
    os.environ["WANDB_DISABLED"] = "false"
    os.environ["WANDB_MODE"] = "online"
    # wrap risky IO or RPC so we can surface a useful failure
    try:
        sys.path.insert(0, str(_REPO))
        from benchmark.wandb_logger import log_checkpoint_artifact
        artifact_url = log_checkpoint_artifact(
            name=Path(args.out_dir).name,
            out_dir=args.out_dir,
            metadata={
                "model_id": args.model_id,
                "mode": args.mode,
                "num_calibration_samples": args.num_samples,
                "max_seq_len": args.max_seq_len,
                "sequential_targets": "Qwen2_5_VLDecoderLayer",
                "scheme": "W4A16",
                "quant_method": qc.get("quant_method", "compressed-tensors"),
                "format": qc.get("format", "pack-quantized"),
            },
            description=f"Qwen2.5-VL-7B {args.mode} (W4A16 GPTQ)",
        )
        if artifact_url:
            print(f"==> wandb artifact run: {artifact_url}")
    except Exception as exc:
        print(f"   (wandb artifact logging skipped: {exc})")

    print("==> Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

