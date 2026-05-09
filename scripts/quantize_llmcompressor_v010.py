# llama3-llava-next-8b quant via llmcompressor 0.10 (domain / generic / w8a8_domain).
# Vision stays FP16. OOM -> PYTORCH_ALLOC_CONF=expandable_segments:True

import argparse, os, sys, sysconfig
from pathlib import Path

# wandb off for oneshot
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

# L4: fragmentation-friendly allocator
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["domain", "generic", "w8a8_domain"], required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--model-id", default="llava-hf/llama3-llava-next-8b-hf")
ap.add_argument("--num-samples", type=int, default=128)
ap.add_argument("--max-seq-len", type=int, default=2048)
args = ap.parse_args()


def _ensure_transformers_save_pretrained_patch() -> None:
    # `ensure_transformers_save_pretrained_patch` lives here. was getting too cramped inline.
    site_paths = [sysconfig.get_paths().get(k) for k in ("purelib", "platlib")]
    # each pass handles the next item in the sequence
    for base in filter(None, site_paths):
        path = Path(base) / "transformers" / "modeling_utils.py"
        if path.exists():
            break
    else:
        raise RuntimeError("could not find transformers/modeling_utils.py to patch")

    text = path.read_text()
    remaining = text.count("if module_map:")
    already_patched = text.count("if module_map and False:")

    if remaining == 0 and already_patched > 0:
        print(f"==> transformers save_pretrained patch already present ({already_patched}x): {path}")
        return

    if remaining == 0:
        raise RuntimeError(
            "transformers modeling_utils.py does not contain the expected "
            "`if module_map:` guard; inspect save_pretrained before quantizing."
        )
    # transformers 4.57.6 has TWO `if module_map:` branches. patch ALL of them.
    path.write_text(text.replace("if module_map:", "if module_map and False:"))
    print(f"==> patched transformers save_pretrained module_map guard ({remaining}x): {path}")


_ensure_transformers_save_pretrained_patch()

# Substation domain calibration text - same set we used for AutoAWQ.
SUBSTATION_BASE = [
    "The image shows a substation transformer with cooling radiators on the side.",
    "Identify the primary piece of substation equipment visible: it appears to be a circuit breaker.",
    "Visible defects include corrosion on the tank surface and minor oil seepage near the bushing.",
    "The equipment condition is assessed as degraded due to weathering and rust streaks.",
    "An analog pressure gauge mounted on the transformer reads 42 kPa.",
    "Disconnector switches with porcelain insulators are mounted on a steel lattice tower.",
    "Lightning arresters protect the transformer from voltage surges; ceramic discs are clean.",
    "The current transformer (CT) housing shows no visible damage; bushings are intact.",
    "Voltage transformer (VT) is mounted on a concrete pedestal with grounding strap visible.",
    "Busbars run horizontally above the equipment yard, supported by string insulators.",
    "Capacitor banks are arranged in three-phase groups within a fenced enclosure.",
    "A surge arrester at the line entrance shows minor discoloration but no cracking.",
    "Recloser cabinet is mounted on the pole; SCADA cable terminations are sealed.",
    "Isolator blade contacts appear oxidized; suggest cleaning and inspection.",
    "Switchgear cabinet doors are closed and locked; IR thermography would help inspect.",
    "Reactor coil shows uniform color with no localized hotspots from operation.",
    "Outdoor air-insulated switchgear (AIS) yard with steel bus support structures.",
    "Indoor metal-clad switchgear with red, yellow, blue phase markers.",
    "Sky is overcast; substation surroundings are fenced with clear sight lines.",
    "Ceramic insulator strings are stacked in groups of 6-8 discs along the line.",
    "Equipment is dominated by power transformers in the foreground panel.",
    "Steel lattice support structures hold horizontal busbars at high elevation.",
    "Concrete pedestals support the bushing terminals at ground potential.",
    "A bird nest is visible on top of one of the transformer bushings - foreign object.",
    "Vegetation overgrowth near the fence requires trimming to maintain clearance.",
    "No infrared imagery in this view; assessment based on RGB visual cues only.",
    "Maintenance walk-down recommended for the panel showing the most weathering.",
    "All four panels show the same general category of substation hardware.",
    "Equipment is likely 30+ years old based on visible aging and color fade.",
    "Recommend maintenance within next 6 months based on visible wear patterns.",
    "Power transformer outdoor installation with conservator tank and Buchholz relay.",
    "The HV bushing porcelain shows light dust accumulation but no surface tracking.",
    "Tertiary winding terminals are accessible via the side cable box.",
    "Tap changer housing is mounted on the side of the transformer tank.",
    "Cooling fans on the radiator bank appear stationary; need operational verification.",
    "Oil level indicator on the conservator reads within the normal operating range.",
    "Silica gel breather color is blue, indicating low moisture absorption.",
    "Earth grid connection from the tank base is visible and properly secured.",
    "Fire suppression piping runs along the tank perimeter at the upper edge.",
    "Lightning impulse waveform test results indicate insulation is intact.",
]

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, LlavaNextForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# Build calibration dataset
if args.mode == "domain" or args.mode == "w8a8_domain":
    texts = (SUBSTATION_BASE * 4)[:args.num_samples]
    calib_ds = Dataset.from_dict({"text": texts})
    print(f"==> DOMAIN calibration: {len(texts)} substation texts")
else:
    # Generic: HuggingFaceH4/ultrachat_200k is llmcompressor's default text-only
    # calibration set. Pull a small slice and flatten messages to text.
    raw = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=42).select(range(args.num_samples))

    def _to_text(ex):
        # `to_text` lives here. was getting too cramped inline.
        msgs = ex.get("messages", [])
        return {"text": "\n".join(m.get("content", "") for m in msgs)[:8000]}
    calib_ds = raw.map(_to_text, remove_columns=raw.column_names)
    print(f"==> GENERIC calibration: {len(calib_ds)} ultrachat samples")

# Load model
print(f"==> Loading {args.model_id} ...")
model = LlavaNextForConditionalGeneration.from_pretrained(
    args.model_id, dtype="auto", device_map="auto"
)
# GPTQ calibration runs only forward passes on the Linear layers. KV cache
# is dead weight that can push the 8B model + activations + Hessians over
# the L4's 22 GB ceiling. Mirror the Qwen track's setup.
model.config.use_cache = False

if hasattr(model.config, "text_config"):
    model.config.text_config.use_cache = False

# Recipe: W4A16 GPTQ for INT4 modes, SmoothQuant + GPTQ W8A8 for INT8 mode.
# Vision tower + multi_modal_projector + lm_head ignored in both cases - vision
# encoders collapse at INT4 and aren't worth the activation-quant cost at INT8
# either (well-documented across LLaVA / Qwen-VL families).
ignore = ["re:.*lm_head", "re:.*vision_tower.*", "re:.*multi_modal_projector.*"]

if args.mode == "w8a8_domain":
    # SmoothQuant first so activation outliers don't blow up the INT8
    # round-to-nearest tail. GPTQ then handles the weight scale search.
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
    recipe = [
        SmoothQuantModifier(smoothing_strength=0.7),
        GPTQModifier(
            targets="Linear",
            scheme="W8A8",
            ignore=ignore,
        ),
    ]
    scheme_label = "W8A8 (SmoothQuant + GPTQ)"
else:
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=ignore,
    )
    scheme_label = "W4A16 (GPTQ, group_size implicit)"

print(f"==> oneshot quantization ({scheme_label}) ...")
oneshot(
    model=model,
    tokenizer=args.model_id,
    dataset=calib_ds,
    recipe=recipe,
    max_seq_length=args.max_seq_len,
    num_calibration_samples=len(calib_ds),
    sequential_targets=["LlamaDecoderLayer"],
)

print(f"==> Saving to {args.out_dir} ...")
os.makedirs(args.out_dir, exist_ok=True)

# Two-step consolidation before save_pretrained:
#  1. dispatch_model: pull all weights off the offload device onto the active device
#  2. remove_hook_from_module: strip accelerate's offload hooks
#
# Without (2), save_pretrained's `if module_map:` offload-aware branch fires
# and crashes on nn.Parameter-only attributes (image_newline, class_embedding)
# with KeyError on module_map lookup. (1) alone is insufficient - confirmed
# by repeated KeyError: 'image_newline' even after dispatch_model.
#
# Saving llmcompressor-compressed weights via save_pretrained(save_compressed=True)
# is mandatory: llmcompressor's compression hooks fire during this call to pack
# weights into TRUE INT4 (weight_packed in INT32 storage). Bypassing
# save_pretrained dumps FP16-dequantized weights instead.
import torch
from compressed_tensors.offload import dispatch_model
from accelerate.hooks import remove_hook_from_module

print("   dispatch_model: consolidating offloaded modules ...")
dispatch_model(model)
print("   remove_hook_from_module: stripping accelerate offload hooks ...")
remove_hook_from_module(model, recurse=True)

model.save_pretrained(args.out_dir, save_compressed=True)
print(f"   save_pretrained completed (compression hooks fired -> packed INT4)")

# Save tokenizer + processor (their save_pretrained is unaffected by the bug).
AutoTokenizer.from_pretrained(args.model_id).save_pretrained(args.out_dir)
from transformers import AutoProcessor
AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True).save_pretrained(args.out_dir)

# Final verify: confirm quantization_config was injected and weights are packed.
import json
cfg_path = os.path.join(args.out_dir, "config.json")
cfg = json.loads(open(cfg_path).read())
qc = cfg.get("quantization_config", {})
if qc:
    # vLLM 0.19's LlavaNext loader expects the ignored vision/projector
    # modules to remain in regex form. Some llmcompressor save paths expand
    # the ignore list into individual q/k/v keys, which makes vLLM's CLIP
    # loader look for fused qkv_proj.weight and crash before readiness.
    qc["ignore"] = ["lm_head", "re:.*vision_tower.*", "re:.*multi_modal_projector.*"]
    cfg["quantization_config"] = qc
    if isinstance(cfg.get("text_config"), dict) and isinstance(cfg["text_config"].get("quantization_config"), dict):
        cfg["text_config"]["quantization_config"]["ignore"] = qc["ignore"]
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
print(f"   config.json quantization_config: present={bool(qc)} method={qc.get('quant_method', 'n/a')}")

# Log a metadata-only W&B Artifact (config + size manifest, NOT the
# 5-9 GB safetensors weights -- those live under models/ on the VM).
os.environ["WANDB_DISABLED"] = "false"
os.environ["WANDB_MODE"] = "online"
# isolate errors so the rest of the call can bail cleanly
try:
    import sys as _sys
    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sys.path.insert(0, _REPO)
    from benchmark.wandb_logger import log_checkpoint_artifact
    artifact_url = log_checkpoint_artifact(
        name=os.path.basename(args.out_dir),
        out_dir=args.out_dir,
        metadata={
            "model_id": args.model_id,
            "mode": args.mode,
            "num_calibration_samples": args.num_samples,
            "max_seq_len": args.max_seq_len,
            "scheme": "W8A8" if args.mode == "w8a8_domain" else "W4A16",
            "quant_method": qc.get("quant_method", "compressed-tensors"),
            "format": qc.get("format", "pack-quantized"),
        },
        description=f"llama3-llava-next-8b {args.mode}",
    )

    if artifact_url:
        print(f"==> wandb artifact run: {artifact_url}")
except Exception as exc:  # noqa: BLE001
    print(f"   (wandb artifact logging skipped: {exc})")

print("==> Done.")

