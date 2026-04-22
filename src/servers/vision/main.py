# Vision MCP server - VLM-backed inspection tools.
# Tools: analyze_image, classify_equipment, detect_visual_defects,
# assess_condition, read_gauge. VLM backend picked from VLM_BASE_URL/VLM_MODEL.

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


def _normalize_equipment(text, taxonomy):
    # does `normalize_equipment`, split out so we can reuse it from a few call sites.
    lines = text.strip().splitlines()

    if not lines:
        return "unknown"
    head = lines[0].lower()
    head = re.sub(r"^(answer|equipment|class|label|type)\s*[:\-]\s*", "", head)
    head = head.strip(" .,'\"")

    if taxonomy:
        # repeat for every element we need to touch
        for cls in sorted(taxonomy, key=len, reverse=True):
            if cls in head:
                return cls
    return head or "unknown"


def _normalize_choice(value, allowed, default):
    # `normalize_choice` lives here. was getting too cramped inline.
    v = value.strip().lower().strip(".")
    for a in allowed:
        if a in v:
            return a
    return default


def _parse_condition(text):
    # walk through `parse_condition`, kept separate so the main flow stays readable.
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


def _parse_defects(text):
    # walk through `parse_defects`, kept separate so the main flow stays readable.
    head = text.strip()
    low = head.lower()

    if low.startswith("none") or low in ("no defects", "n/a", "intact"):
        return []
    line = next((ln for ln in head.splitlines() if ln.strip()), "")
    return [d.strip().lower() for d in line.split(",") if d.strip()]


def _domain_phrase(image_ref, fallback="industrial equipment image"):
    hint = image_loader.domain_hint_for(image_ref)
    return hint or fallback


mcp = FastMCP("vision")


def _safe_load(image_ref):
    # walk through `safe_load`, kept separate so the main flow stays readable.
    try:
        return image_loader.load_image(image_ref), None
    except Exception as exc:
        return None, ErrorResult(error=f"image_load_failed: {exc}")


@mcp.tool()
async def analyze_image(
    image_ref: str,
    question: str,
    max_tokens: int = 256,
) -> Union[AnalyzeResult, ErrorResult]:
    # free-form VLM call with a generic domain hint
    img, err = _safe_load(image_ref)

    if err is not None:
        return err

    domain = _domain_phrase(image_ref)
    primed_prompt = (
        f"You are inspecting {domain}. Answer the user's question "
        "grounded in the visible features of the image.\n\n"
        f"User question: {question}"
    )

    # keep the happy path obvious by catching failures here
    try:
        text = await vlm_client.vlm_call(primed_prompt, img, max_tokens=max_tokens)
    except Exception as exc:
        logger.error("vlm_call failed: %s", exc)
        return ErrorResult(error=f"vlm_call_failed: {exc}")
    return AnalyzeResult(image_ref=image_ref, answer=text.strip())


@mcp.tool()
async def classify_equipment(image_ref: str) -> Union[EquipmentResult, ErrorResult]:
    # taxonomy + domain hint come from the registered dataset spec
    img, err = _safe_load(image_ref)

    if err is not None:
        return err
    domain = _domain_phrase(image_ref)
    taxonomy = list(image_loader.taxonomy_for(image_ref))

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
    # keep the happy path obvious by catching failures here
    try:
        text = await vlm_client.vlm_call(prompt, img, max_tokens=128)
    except Exception as exc:
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
    # list visible defects. empty list when intact
    img, err = _safe_load(image_ref)

    if err is not None:
        return err
    domain = _domain_phrase(image_ref)
    prompt = (
        f"You are an expert inspector of {domain}. List visible defects in "
        "this image. Reply on a single line as a comma-separated list of "
        "short defect names. If no defects are visible, reply exactly 'none'."
    )
    # keep the happy path obvious by catching failures here
    try:
        text = await vlm_client.vlm_call(prompt, img, max_tokens=128)
    except Exception as exc:
        logger.error("vlm_call failed: %s", exc)
        return ErrorResult(error=f"vlm_call_failed: {exc}")
    return DefectResult(image_ref=image_ref, defects=_parse_defects(text), raw_response=text)


@mcp.tool()
async def assess_condition(image_ref: str) -> Union[ConditionResult, ErrorResult]:
    # good / degraded / critical (works for any registered dataset)
    img, err = _safe_load(image_ref)

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
    # isolate errors so the rest of the call can bail cleanly
    try:
        text = await vlm_client.vlm_call(prompt, img, max_tokens=128)
    except Exception as exc:
        logger.error("vlm_call failed: %s", exc)
        return ErrorResult(error=f"vlm_call_failed: {exc}")
    cond, conf = _parse_condition(text)
    return ConditionResult(
        image_ref=image_ref, condition=cond, confidence=conf, raw_response=text
    )


@mcp.tool()
async def read_gauge(image_ref: str) -> Union[GaugeResult, ErrorResult]:
    # read the numeric value off an analog gauge
    img, err = _safe_load(image_ref)

    if err is not None:
        return err
    prompt = (
        "Read the analog gauge shown in this image. Reply on the first line "
        "with just the numeric reading and unit (e.g. '42.5 kV'). If no gauge "
        "is visible, reply 'no gauge'."
    )
    # keep the happy path obvious by catching failures here
    try:
        text = await vlm_client.vlm_call(prompt, img, max_tokens=64)
    except Exception as exc:
        logger.error("vlm_call failed: %s", exc)
        return ErrorResult(error=f"vlm_call_failed: {exc}")
    reading = text.strip().splitlines()[0].strip() if text.strip() else "unknown"
    return GaugeResult(image_ref=image_ref, reading=reading, raw_response=text)


def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()

