# Repack a 'fake-quant' (FP16 + weight_scale/zero_point) checkpoint into
# real compressed-tensors INT4 that vLLM can serve as W4A16.
# in:  /tmp/llmc_inner_* (inner LlamaForCausalLM with scales)
#      llava-hf/llama3-llava-next-8b-hf (FP16 wrapper)
# out: a vLLM-servable INT4 directory

import os, sys, glob, argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--inner-dir", required=True)
ap.add_argument("--out-dir", required=True)
args = ap.parse_args()

WRAPPER_ID = "llava-hf/llama3-llava-next-8b-hf"

import torch
from transformers import (
    LlavaNextForConditionalGeneration,
    LlamaForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)
from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.compressors import ModelCompressor

# W4A16 quant config matching our GPTQModifier recipe
quant_config_dict = {
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "input_activations": None,
            "output_activations": None,
            "weights": {
                "num_bits": 4,
                "type": "int",
                "strategy": "group",
                "group_size": 128,
                "symmetric": True,
                "dynamic": False,
                "observer": "minmax",
            },
        }
    },
    "format": "pack-quantized",
    "ignore": ["lm_head", "re:.*vision_tower.*", "re:.*multi_modal_projector.*"],
    "quant_method": "compressed-tensors",
    "quantization_status": "frozen",
}

print(f"==> Loading FP16 wrapper {WRAPPER_ID} ...")
wrapper = LlavaNextForConditionalGeneration.from_pretrained(
    WRAPPER_ID, torch_dtype=torch.float16, device_map="cpu",
)
print(f"==> Patching inner config.json with quantization_config ...")
inner_cfg_path = os.path.join(args.inner_dir, "config.json")

with open(inner_cfg_path) as f:
    inner_cfg = json.load(f)
inner_cfg["quantization_config"] = quant_config_dict

with open(inner_cfg_path, "w") as f:
    json.dump(inner_cfg, f, indent=2)
print(f"   patched {inner_cfg_path}")

print(f"==> Loading inner with compression-aware loader ...")
inner = LlamaForCausalLM.from_pretrained(
    args.inner_dir, torch_dtype=torch.float16, device_map="cpu",
)
print(f"   inner type: {type(inner).__name__}")
# Check that scales are now loaded (not dropped)
n_scales = sum(1 for n, _ in inner.named_buffers() if "weight_scale" in n)
n_scales_p = sum(1 for n, _ in inner.named_parameters() if "weight_scale" in n)
print(f"   weight_scale buffers: {n_scales}  parameters: {n_scales_p}")

print(f"==> wrapper.language_model = inner")
wrapper.language_model = inner

print(f"==> Setting wrapper config.quantization_config (text_config-scoped via top-level) ...")
wrapper.config.quantization_config = quant_config_dict

print(f"==> Saving merged checkpoint -> {args.out_dir}")
os.makedirs(args.out_dir, exist_ok=True)
wrapper.save_pretrained(args.out_dir, safe_serialization=True)
AutoProcessor.from_pretrained(WRAPPER_ID).save_pretrained(args.out_dir)
AutoTokenizer.from_pretrained(WRAPPER_ID).save_pretrained(args.out_dir)
print("==> Done.")

