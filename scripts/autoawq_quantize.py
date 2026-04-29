# AutoAWQ quantization of llama3-llava-next-8b for vLLM serving.
# Produces real packed INT4 (unlike llmcompressor 0.3.0 fake-quant).
# domain  : 128 substation-inspection texts
# generic : default 'pileval' (general English from The Pile)

import os, sys, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["domain", "generic"], required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--model-id", default="llava-hf/llama3-llava-next-8b-hf")
ap.add_argument("--max-samples", type=int, default=128)
ap.add_argument("--max-seq-len", type=int, default=512)
args = ap.parse_args()

# substation calibration texts (~40 lines, replicated to 128) - hand-authored to
# match the scenario distribution: equipment ID, defects, condition, gauge
SUBSTATION_TEXTS_BASE = [
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
# replicate to AutoAWQ's default sample budget of 128
SUBSTATION_TEXTS = (SUBSTATION_TEXTS_BASE * 4)[:128]

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AutoProcessor

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",  # vLLM supports GEMM. GEMV is faster on small batches
}

print(f"==> Loading {args.model_id} ...")
model = AutoAWQForCausalLM.from_pretrained(
    args.model_id,
    safetensors=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

calib = SUBSTATION_TEXTS if args.mode == "domain" else "pileval"
mode_desc = f"DOMAIN ({len(SUBSTATION_TEXTS)} substation texts)" if args.mode == "domain" else "GENERIC (pileval)"
print(f"==> Calibration: {mode_desc}")
print(f"==> Quantizing W4A16 group_size=128 ...")

model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calib,
    max_calib_samples=args.max_samples,
    max_calib_seq_len=args.max_seq_len,
        n_parallel_calib_samples=1,
)

print(f"==> Saving to {args.out_dir} ...")
os.makedirs(args.out_dir, exist_ok=True)
model.save_quantized(args.out_dir)
tokenizer.save_pretrained(args.out_dir)
# LlavaNext needs the processor for image preprocessing during serving
try:
    AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True).save_pretrained(args.out_dir)
except Exception as exc:
    print(f"   (processor save skipped: {exc})")
print("==> Done.")
