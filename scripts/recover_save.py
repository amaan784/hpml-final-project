# Recovery: merge most-recent quantized inner with the FP16 LlavaNext wrapper.

import os, sys, glob, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", default="/opt/models/llama3-llava-next-8b-awq-generic")
ap.add_argument("--inner-dir", default=None, help="Auto-pick latest /tmp/llmc_inner_* if unset")
args = ap.parse_args()

if args.inner_dir is None:
    candidates = sorted(glob.glob("/tmp/llmc_inner_*"), key=os.path.getmtime, reverse=True)
    if not candidates:
        print("No /tmp/llmc_inner_* dir found", file=sys.stderr); sys.exit(1)
    INNER_DIR = candidates[0]
else:
    INNER_DIR = args.inner_dir
print(f"==> Using inner checkpoint: {INNER_DIR}")
print(f"==> Saving merged wrapper -> {args.out_dir}")

WRAPPER_ID = "llava-hf/llama3-llava-next-8b-hf"

import torch
from transformers import (
    LlavaNextForConditionalGeneration,
    LlamaForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

print(f"==> Loading FP16 wrapper {WRAPPER_ID} ...")
wrapper = LlavaNextForConditionalGeneration.from_pretrained(
    WRAPPER_ID, torch_dtype=torch.float16, device_map="cpu",
)
print(f"==> Loading quantized inner from {INNER_DIR} ...")
inner = LlamaForCausalLM.from_pretrained(
    INNER_DIR, torch_dtype=torch.float16, device_map="cpu",
)
print("==> Direct swap: wrapper.language_model = inner")
wrapper.language_model = inner
os.makedirs(args.out_dir, exist_ok=True)
wrapper.save_pretrained(args.out_dir)
AutoProcessor.from_pretrained(WRAPPER_ID).save_pretrained(args.out_dir)
AutoTokenizer.from_pretrained(WRAPPER_ID).save_pretrained(args.out_dir)
print("==> Done.")

