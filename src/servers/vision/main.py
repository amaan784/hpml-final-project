# Vision MCP: VLM tools (analyze, classify, defects, etc.) backed by vLLM.

import logging
import os
import re
from typing import List, Union

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from . import image_loader, vlm_client

load_dotenv()

_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("vision-mcp-server")


CONDITION_LEVELS = ["good", "degraded", "critical"]
CONFIDENCE_LEVELS = ["low", "medium", "high"]


# Result models
class ErrorResult(BaseModel):
    error: str


class AnalyzeResult(BaseModel):
    image_ref: str
    answer: str


class EquipmentResult(BaseModel):
    image_ref: str
    equipment_type: str
    domain: str
    raw_response: str


class DefectResult(BaseModel):
    image_ref: str
    defects: List[str]
    raw_response: str


class ConditionResult(BaseModel):
    image_ref: str
    condition: str
    confidence: str
    raw_response: str


class GaugeResult(BaseModel):
    image_ref: str
    reading: str
    raw_response: str


# Parsing helpers
def _normalize_equipment(text: str, taxonomy: tuple[str, ...] | list[str]) -> str:
    """Map a free-form VLM reply to one of ``taxonomy`` classes (best-effort).

    Empty taxonomy means 'open-set' - return the first cleaned line as-is.
    """
    lines = text.strip().splitlines()

    # Blank model output should not propagate as if it matched a taxonomy
    if not lines:
        return "unknown"
    head = lines[0].lower()
    head = re.sub(r"^(answer|equipment|class|label|type)\s*[:\-]\s*", "", head)
    head = head.strip(" .,'\"")

    # Longest token wins so multi-word taxonomy entries beat single-word overlaps.
    if taxonomy:
        # Longest-class-first so multi-word labels beat single-word substrings.
        for cls in sorted(taxonomy, key=len, reverse=True):
            if cls in head:
                return cls
    return head or "unknown"


def _normalize_choice(value: str, allowed: List[str], default: str) -> str:
    # VLMs sprinkle punctuation around keyword so we need to strip
    v = value.strip().lower().strip(".")
    for a in allowed:
        if a in v:
            return a
    return default


def _parse_condition(text: str) -> tuple[str, str]:
    # Split "condition: ...; confidence: ..." style responses from assess_condition.
    cond = "unknown"
    conf = "low"
    for part in re.split(r"[;\n]", text):
        if ":" not in part:
            continue
        key, _, val = part.partition(":")
        key = key.strip().lower()
        if key == "condition":
            cond = _normalize_choice(val, CONDITION_LEVELS, "unknown")
        elif key == "confidence":
            conf = _normalize_choice(val, CONFIDENCE_LEVELS, "low")
    return cond, conf


def _parse_defects(text: str) -> List[str]:
    # Treat negation keywords as "no defect list" instead of empty parsing noise.
    head = text.strip()
    low = head.lower()

    if low.startswith("none") or low in ("no defects", "n/a", "intact"):
        return []
    line = next((ln for ln in head.splitlines() if ln.strip()), "")
    return [d.strip().lower() for d in line.split(",") if d.strip()]


# Domain priming
# When the image is a local path (not ``hf://``), ``image_loader.domain_hint_for``
# returns None and tools fall back to a useless generic phrase. We patch that
# here by substring-matching the path against a table of domain hints. Each
# teammate's domain (pump / motor / turbine / transformer) drops in via the
# tables below - no other code change needed.

# Short noun phrase. Used by classify_equipment, detect_visual_defects,
# assess_condition (they wrap it as "You are an expert inspector of {domain}").
_LOCAL_PATH_DOMAIN_HINTS = {
    "pump": (
        "cast submersible pump impeller (back/hub face, top-view); "
        "the dark area in the center is a normal through-hole bore, "
        "defects appear on the outer rim"
    ),
    # Add more aliases as teammates plug in their datasets:
    # "motor": "...",
    # "turbine": "...",
    # "transformer": "...",
}

# Full system prompt used ONLY by analyze_image. Includes complete
# inspection-domain guidance: what is normal vs. what is a defect, where to
# look, common defect types, and answer-format expectations. This is the
# single biggest accuracy lever for freeform inspection queries.
_ANALYZE_IMAGE_PROMPTS = {
    "pump": (
        "You are a quality inspector for cast submersible pump impellers.\n\n"
        "You are examining the BACK FACE (hub side) of an impeller from a "
        "TOP-VIEW. The vanes are on the OPPOSITE face and are NOT visible "
        "in this view.\n\n"
        "IMPORTANT - what is normal versus what is a defect:\n"
        "- The dark area in the center is a normal THROUGH-HOLE BORE (an "
        "opening through the part). It is NOT a defect: not overheating, "
        "not a void, not corrosion, not wear, not material loss.\n"
        "- Concentric circular machining marks on the flat ring are normal "
        "post-processing artifacts, not cracks or scratches.\n\n"
        "Casting defects on this view appear on the OUTER RIM (perimeter), "
        "not the center. Common defect types:\n"
        "- Burrs / flash: jagged or excess material along the outer edge "
        "(from material escaping the mold parting line)\n"
        "- Chipping / material loss: missing chunks along the rim\n"
        "- Irregular outer profile: oval, dented, or non-circular outline\n"
        "- Surface cracks or pitting on the cast face\n\n"
        "Answer the user's question grounded in visible features. When you "
        "identify a defect, localize it using clock positions (e.g. "
        "'2-3 o'clock') or compass directions (e.g. 'upper-left'). Be "
        "specific. Do not invent defects that are not visible."
    ),
    # Teammates: add your domain prompt here when you register your scenarios.
}


def _domain_phrase(image_ref: str, fallback: str = "industrial equipment image") -> str:
    """Return a short noun-phrase domain hint for the image_ref's equipment.

    Resolution order:
      1. Registered HuggingFace dataset (``hf://alias/...``).
      2. Substring match on local file path (pump / motor / etc.).
      3. Fallback ``"industrial equipment image"``.
    """
    hint = image_loader.domain_hint_for(image_ref)

    # Prefer dataset-declared wording before sniffing filenames.
    if hint:
        return hint
    low = image_ref.lower()
    for key, val in _LOCAL_PATH_DOMAIN_HINTS.items():
        if key in low:
            return val
    return fallback


def _analyze_image_system_prompt(image_ref: str) -> str | None:
    """Return a full domain-aware system prompt for ``analyze_image``, or None.

    The full prompt encodes domain-specific normalcy guidance (e.g. "the dark
    center is a bore, not a defect") that a generalist VLM cannot infer from
    the image alone. Falls back to None if no domain match - the caller
    builds a generic prompt from ``_domain_phrase`` in that case.
    """
    low = image_ref.lower()
    for key, prompt in _ANALYZE_IMAGE_PROMPTS.items():
        if key in low:
            return prompt
    return None


# Server
mcp = FastMCP("vision")


def _safe_load(image_ref: str):
    # Normalize loader failures into the MCP error envelope every tool consumes.
    try:
        return image_loader.load_image(image_ref), None
    except Exception as exc:  # noqa: BLE001
        return None, ErrorResult(error=f"image_load_failed: {exc}")


@mcp.tool()
async def analyze_image(
    image_ref: str,
    question: str,
    max_tokens: int = 768,
) -> Union[AnalyzeResult, ErrorResult]:
    """Free-form VLM call with domain-aware system prompt priming.

    The tool prepends a domain-aware system prompt to the user's question so
    the VLM has the equipment-specific context (what is normal, where defects
    appear, common defect types) that a generalist VLM otherwise lacks.
    Without this priming, off-the-shelf VLMs commonly mistake normal
    structural features (e.g. a bore opening) for defects.

    Resolution:
      1. If a full domain prompt is registered for this image_ref's domain
         (see ``_ANALYZE_IMAGE_PROMPTS``), use it verbatim.
      2. Otherwise, build a generic "you are inspecting {domain}" framing
         using the short ``_domain_phrase`` hint.

    Use this for scenarios that need freeform descriptive answers
    (accept/reject decisions with justification, defect localization,
    surface condition narratives). For label-only outputs prefer
    ``classify_equipment`` / ``detect_visual_defects`` / ``assess_condition``.

    Args:
        image_ref: ``hf://alias/[split/]index``, absolute path, or URL.
        question: The user's question. Will be embedded after the system
            prompt.
        max_tokens: Output cap. Bumped to 768 (from the original 256) so
            that "describe + reason + verdict" answers don't get truncated
            mid-sentence - truncation otherwise produces hedged endings
            ("appears acceptable...") that don't reflect the model's full
            judgement.

    Returns:
        ``AnalyzeResult(image_ref, answer)`` on success;
        ``ErrorResult(error)`` if the image cannot load or the VLM call
        fails.
    """
    img, err = _safe_load(image_ref)

    # No pixels => bail before touching the GPU-backed VLM.
    if err is not None:
        return err

    full_prompt = _analyze_image_system_prompt(image_ref)

    # Prefer the chunky domain-specific playbook when teammates registered one.
    if full_prompt is not None:
        primed_prompt = f"{full_prompt}\n\nUser question: {question}"
    else:
        domain = _domain_phrase(image_ref)
        primed_prompt = (
            f"You are inspecting {domain}. Answer the user's question "
            "grounded in the visible features of the image. Be specific "
            "about defect locations using clock positions or compass "
            "directions. Do not invent defects that are not visible.\n\n"
            f"User question: {question}"
        )

    # Bubble OpenAI-compatible client exceptions into ErrorResult payloads.
    try:
        text = await vlm_client.vlm_call(primed_prompt, img, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001
        logger.error("vlm_call failed: %s", exc)
        return ErrorResult(error=f"vlm_call_failed: {exc}")
    return AnalyzeResult(image_ref=image_ref, answer=text.strip())


@mcp.tool()
async def classify_equipment(image_ref: str) -> Union[EquipmentResult, ErrorResult]:
    """Identify the primary piece of equipment shown in the image.

    The taxonomy and domain phrasing are pulled from the dataset spec
    registered for the image's alias, so this works across any
    register_dataset() entry without code changes.
    """
    img, err = _safe_load(image_ref)

    # No pixels => bail before touching the GPU-backed VLM.
    if err is not None:
        return err
    domain = _domain_phrase(image_ref)
    taxonomy = list(image_loader.taxonomy_for(image_ref))

    # Enumerated labels tighten logits; otherwise let the VLM invent a short class name.
    if taxonomy:
        prompt = (
            f"You are an expert inspector of {domain}. Identify the single "
            "primary piece of equipment in this image. Reply with one of "
            f"these exact labels on the first line: {', '.join(taxonomy)}. "
            "On the second line give a one-sentence reason."
        )
    else:
        prompt = (
            f"You are an expert inspector of {domain}. Identify the single "
            "primary piece of equipment in this image. Reply with the class "
            "name on the first line and a one-sentence reason on the second."
        )
    # Bubble OpenAI-compatible client exceptions into ErrorResult payloads.
    try:
        text = await vlm_client.vlm_call(prompt, img, max_tokens=128)
    except Exception as exc:  # noqa: BLE001
        logger.error("vlm_call failed: %s", exc)
        return ErrorResult(error=f"vlm_call_failed: {exc}")
    return EquipmentResult(
        image_ref=image_ref,
        equipment_type=_normalize_equipment(text, taxonomy),
        domain=domain,
        raw_response=text,
    )


@mcp.tool()
async def detect_visual_defects(image_ref: str) -> Union[DefectResult, ErrorResult]:
    """List visible defects in the image.

    Examples by domain (auto-selected via the dataset's domain_hint):
      transformer -> corrosion, oil leak, broken insulator, arcing damage
      motor       -> burn marks, casing crack, loose mount, oil seepage
      cable       -> frayed strand, burn mark, cut, splice failure
      pipe        -> surface corrosion, pitting, leak, weld defect
    Returns an empty list when intact.
    """
    img, err = _safe_load(image_ref)

    # No pixels => bail before touching the GPU-backed VLM.
    if err is not None:
        return err
    domain = _domain_phrase(image_ref)
    prompt = (
        f"You are an expert inspector of {domain}. List visible defects in "
        "this image. Reply on a single line as a comma-separated list of "
        "short defect names. If no defects are visible, reply exactly 'none'."
    )
    # Bubble OpenAI-compatible client exceptions into ErrorResult payloads.
    try:
        text = await vlm_client.vlm_call(prompt, img, max_tokens=128)
    except Exception as exc:  # noqa: BLE001
        logger.error("vlm_call failed: %s", exc)
        return ErrorResult(error=f"vlm_call_failed: {exc}")
    return DefectResult(image_ref=image_ref, defects=_parse_defects(text), raw_response=text)


@mcp.tool()
async def assess_condition(image_ref: str) -> Union[ConditionResult, ErrorResult]:
    """Assess overall equipment condition: good / degraded / critical.

    Domain-agnostic: works for any registered dataset.
    """
    img, err = _safe_load(image_ref)

    # No pixels => bail before touching the GPU-backed VLM.
    if err is not None:
        return err
    domain = _domain_phrase(image_ref)
    prompt = (
        f"You are an expert inspector of {domain}. Assess the operating "
        "condition of the equipment in this image. Reply EXACTLY in this "
        "format on one line:\n"
        "condition: <good|degraded|critical>; confidence: <low|medium|high>; "
        "reason: <one sentence>"
    )
    # Mirror other tools — network blips shouldn't crash the MCP host.
    try:
        text = await vlm_client.vlm_call(prompt, img, max_tokens=128)
    except Exception as exc:  # noqa: BLE001
        logger.error("vlm_call failed: %s", exc)
        return ErrorResult(error=f"vlm_call_failed: {exc}")
    cond, conf = _parse_condition(text)
    return ConditionResult(
        image_ref=image_ref, condition=cond, confidence=conf, raw_response=text
    )


@mcp.tool()
async def read_gauge(image_ref: str) -> Union[GaugeResult, ErrorResult]:
    """Read the numeric value shown by an analog gauge in the image.

    Domain-agnostic: works for any registered dataset.
    """
    img, err = _safe_load(image_ref)

    # No pixels => bail before touching the GPU-backed VLM.
    if err is not None:
        return err
    prompt = (
        "Read the analog gauge shown in this image. Reply on the first line "
        "with just the numeric reading and unit (e.g. '42.5 kV'). If no gauge "
        "is visible, reply 'no gauge'."
    )
    # Short timeout call still needs structured failure reporting.
    try:
        text = await vlm_client.vlm_call(prompt, img, max_tokens=64)
    except Exception as exc:  # noqa: BLE001
        logger.error("vlm_call failed: %s", exc)
        return ErrorResult(error=f"vlm_call_failed: {exc}")
    reading = text.strip().splitlines()[0].strip() if text.strip() else "unknown"
    return GaugeResult(image_ref=image_ref, reading=reading, raw_response=text)


def main():
    # Start stdio transport for MCP clients (Cursor / bench harness wrappers).
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()

