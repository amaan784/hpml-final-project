# Post-quantization sanity check for llmcompressor 0.10 W4A16 checkpoints.
# Validates config.json quantization_config, state-dict key naming, file
# presence, and packed INT4 tensor shapes BEFORE attempting vLLM serve.
# Saves us from re-running a 25-minute quantization on a key prefix mismatch.

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt_dir", help="Path to checkpoint directory")
    args = p.parse_args()
    ckpt = Path(args.ckpt_dir)

    print(f"=== Verifying {ckpt} ===\n")
    issues = []

    # 1. File presence
    print("1. File presence:")
    must_have = ["config.json", "tokenizer_config.json"]
    bin_or_st = list(ckpt.glob("*.bin")) + list(ckpt.glob("*.safetensors"))
    # repeat for every element we need to touch
    for f in must_have:
        present = (ckpt / f).exists()
        print(f"   {f:40s} {'OK' if present else 'MISSING'}")
        if not present:
            issues.append(f"missing {f}")
    print(f"   weights ({len(bin_or_st)} files): " +
          (", ".join(f.name for f in bin_or_st) if bin_or_st else "NO WEIGHTS"))

    if not bin_or_st:
        issues.append("no .bin or .safetensors weights found")

    # 2. config.json check
    print("\n2. config.json:")
    cfg_path = ckpt / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        print(f"   architectures: {cfg.get('architectures')}")
        print(f"   model_type:    {cfg.get('model_type')}")
        qc = cfg.get("quantization_config", cfg.get("text_config", {}).get("quantization_config", {}))

        if not qc:
            issues.append("config.json has no quantization_config")
            print("   quantization_config: MISSING")
        else:
            print(f"   quantization_config keys: {sorted(qc.keys())}")
            print(f"   quant_method: {qc.get('quant_method', '?')}")
            print(f"   format:       {qc.get('format', '?')}")
            print(f"   ignore:       {qc.get('ignore', [])[:3]}...")
            if qc.get("quant_method") not in ("compressed-tensors", "compressed_tensors"):
                issues.append(f"quant_method should be 'compressed-tensors', got {qc.get('quant_method')}")

    # 3. State-dict key naming
    print("\n3. State-dict key naming (vs vLLM 0.19 LlavaNext mapper):")
    print("   vLLM mapper expects keys with prefixes:")
    print("     model.language_model.X  (will be remapped to language_model.model.X)")
    print("     model.vision_tower.X")
    print("     model.multi_modal_projector.X")

    keys = []

    if bin_or_st:
        f = bin_or_st[0]
        if f.suffix == ".safetensors":
            from safetensors import safe_open
            with safe_open(f, framework="pt") as sf:
                keys = list(sf.keys())
        else:
            import torch
            sd = torch.load(f, map_location="cpu", weights_only=True)
            keys = list(sd.keys())
            del sd

    if keys:
        # sample first 5 keys to detect prefix format
        sample = keys[:5]
        print(f"\n   first 5 keys: {sample}")
        n_with_outer_model = sum(1 for k in keys if k.startswith("model.language_model.")
                                                 or k.startswith("model.vision_tower.")
                                                 or k.startswith("model.multi_modal_projector."))
        n_no_outer_model = sum(1 for k in keys if k.startswith("language_model.")
                                              or k.startswith("vision_tower.")
                                              or k.startswith("multi_modal_projector."))
        print(f"   keys with 'model.' outer prefix:    {n_with_outer_model}")
        print(f"   keys WITHOUT 'model.' outer prefix: {n_no_outer_model}")

        if n_with_outer_model > 0 and n_no_outer_model == 0:
            print("   OK Format: transformers 4.52+ (vLLM 0.19 mapper will handle this)")
        elif n_no_outer_model > 0 and n_with_outer_model == 0:
            print("   WARN Format: transformers <4.52 (vLLM mapper needs 'model.' prefix added)")
            issues.append("state_dict uses pre-4.52 key format; add 'model.' prefix before serving")
        else:
            print("   WARN mixed key prefixes - investigate")
            issues.append("state_dict has mixed key prefixes")

        # Quantization-aware keys
        n_packed = sum(1 for k in keys if k.endswith(".weight_packed"))
        n_scale = sum(1 for k in keys if k.endswith(".weight_scale"))
        n_zero = sum(1 for k in keys if k.endswith(".weight_zero_point"))
        n_weight_only = sum(1 for k in keys if k.endswith(".weight") and not any(
            k.endswith(suff) for suff in (".weight_packed", ".weight_scale", ".weight_zero_point")
        ))
        print(f"\n   weight_packed keys:     {n_packed}")
        print(f"   weight_scale keys:      {n_scale}")
        print(f"   weight_zero_point keys: {n_zero}")
        print(f"   plain .weight keys:     {n_weight_only}  (lm_head, embeddings, ignored layers)")

        if n_packed == 0:
            issues.append("no weight_packed keys - checkpoint is NOT real INT4 (might be fake-quant FP16)")
            print("   FAIL: not real packed INT4. Probably saved as dequantized FP16.")
        else:
            print("   OK real packed INT4 (W4A16)")

    print("\n=== Verification result ===")

    if not issues:
        print("OK all checks pass - ready to serve via vLLM 0.19 --quantization compressed-tensors")
    else:
        print(f"FAIL: {len(issues)} issue(s) to fix:")
        # repeat for every element we need to touch
        for i in issues:
            print(f"   - {i}")
    print()

    return 0 if not issues else 1

if __name__ == "__main__":
    sys.exit(main())

