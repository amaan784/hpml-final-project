# Calibration corpora for AWQ/GPTQ quantization.
# DOMAIN: 40 substation sentences. GENERIC: pulled from ultrachat_200k.

from typing import Iterable

# Substation sentences. Same set as in quantize_llmcompressor_v010.py.
SUBSTATION_TEXTS: tuple[str, ...] = (
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
)


def build_domain_corpus(num_samples=128):
    # cycle SUBSTATION_TEXTS until we hit num_samples
    base = SUBSTATION_TEXTS
    return [base[i % len(base)] for i in range(num_samples)]


def build_generic_corpus(
    num_samples=128,
    repo="HuggingFaceH4/ultrachat_200k",
    split="train_sft",
    seed=42,
    max_chars=8000,
):
    # Shared `build_generic_corpus` logic reused by multiple benchmark paths.
    from datasets import load_dataset

    raw = load_dataset(repo, split=split).shuffle(seed=seed).select(range(num_samples))
    out = []
    # each pass handles the next item in the sequence
    for ex in raw:
        msgs = ex.get("messages", []) or []
        text = "\n".join((m.get("content") or "") for m in msgs)
        out.append(text[:max_chars])
    return out


def write_corpus_preview(texts: Iterable[str], out_path: str, head: int = 8) -> None:
    # dump first `head` samples to out_path for debugging
    from pathlib import Path

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        # process records in deterministic order
        for i, t in enumerate(list(texts)[:head]):
            fh.write(f"--- sample {i} ---\n{t}\n\n")
