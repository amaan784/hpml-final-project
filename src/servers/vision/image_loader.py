# Image loader for the vision MCP server.
# Resolves image_ref strings: hf://alias/[split/]index, abs paths, file://, http(s)://.

import io
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import requests
from PIL import Image


# dataset registry

@dataclass(frozen=True)
class DatasetSpec:
    alias: str
    repo: str
    default_split: str = "train"
    image_col: str = "image"
    label_col: str | None = "label"
    domain_hint: str = "industrial equipment image"
    prompt_taxonomy: tuple = field(default_factory=tuple)


_REGISTRY: dict[str, DatasetSpec] = {}


def register_dataset(spec):
    # walk through `register_dataset`, kept separate so the main flow stays readable.
    _REGISTRY[spec.alias] = spec


def get_spec(alias):
    if alias not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset alias {alias!r}. Known: {sorted(_REGISTRY)}."
        )
    return _REGISTRY[alias]


def list_aliases():
    # does `list_aliases`, split out so we can reuse it from a few call sites.
    return sorted(_REGISTRY)


# built-in registrations (teammates add their own here or in their scenario modules)

register_dataset(DatasetSpec(
    alias="transformer",
    repo="AndrzejDD/15-class-Substation-Equipment",
    default_split="train",
    image_col="image",
    label_col=None,
    domain_hint="substation transformer / switchgear / insulator equipment",
    prompt_taxonomy=(
        "transformer", "circuit breaker", "disconnector", "lightning arrester",
        "current transformer", "voltage transformer", "busbar", "capacitor bank",
        "insulator", "switchgear", "reactor", "surge arrester",
        "recloser", "isolator", "other",
    ),
))

register_dataset(DatasetSpec(
    alias="substation",
    repo="AndrzejDD/15-class-Substation-Equipment",
    default_split="train",
    image_col="image",
    label_col=None,
    domain_hint="substation transformer / switchgear / insulator equipment",
    prompt_taxonomy=_REGISTRY["transformer"].prompt_taxonomy,
))


# templates for teammates - uncomment and fill repo:
#
# register_dataset(DatasetSpec(
#     alias="motor",
#     repo="<TEAMMATE_HF_ORG>/<motor-dataset>",
#     domain_hint="electric motor health: rotor / bearing / stator faults",
#     prompt_taxonomy=("healthy", "inner race fault", "outer race fault",
#                      "ball fault", "rotor bar fault", "stator winding fault"),
# ))
#
# register_dataset(DatasetSpec(
#     alias="cable",
#     repo="<TEAMMATE_HF_ORG>/<cable-dataset>",
#     domain_hint="power cable / conductor visual defects",
#     prompt_taxonomy=("intact", "frayed strand", "burn mark", "cut", "splice"),
# ))
#
# register_dataset(DatasetSpec(
#     alias="pipe",
#     repo="<TEAMMATE_HF_ORG>/<pipe-dataset>",
#     domain_hint="industrial pipe corrosion / leak inspection",
#     prompt_taxonomy=("intact", "surface corrosion", "pitting", "leak", "weld defect"),
# ))


# image loading

def _parse_hf_ref(ref):
    body = ref[len("hf://"):]
    parts = body.split("/")
    if len(parts) == 2:
        alias, idx = parts
        spec = get_spec(alias)
        split = spec.default_split
    elif len(parts) >= 3:
        alias, split = parts[0], parts[1]
        idx = parts[-1]
        spec = get_spec(alias)
    else:
        raise ValueError(f"Bad hf:// ref: {ref!r}. Expected hf://alias/[split/]index")
    return spec, split, int(idx)


@lru_cache(maxsize=8)
def _load_split(repo, split):
    # walk through `load_split`, kept separate so the main flow stays readable.
    from datasets import load_dataset
    return load_dataset(repo, split=split)


def load_image(image_ref):
    # resolve image_ref to a PIL.Image (always RGB)
    if image_ref.startswith("hf://"):
        spec, split, idx = _parse_hf_ref(image_ref)
        ds = _load_split(spec.repo, split)

        if idx < 0 or idx >= len(ds):
            raise IndexError(
                f"hf://{spec.alias}/{split}/{idx} out of range (size={len(ds)})"
            )
        item = ds[idx]
        img = item.get(spec.image_col)

        if img is None:
            # repeat for every element we need to touch
            for fallback in ("image", "img", "png", "jpg"):
                img = item.get(fallback)
                if img is not None:
                    break

        if img is None:
            raise KeyError(
                f"No image column found in {spec.repo}/{split}[{idx}]; "
                f"keys={list(item)}, expected '{spec.image_col}'"
            )
        return img.convert("RGB")

    if image_ref.startswith(("http://", "https://")):
        resp = requests.get(image_ref, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    path = image_ref[len("file://"):] if image_ref.startswith("file://") else image_ref
    p = Path(path)

    if not p.is_absolute():
        p = (Path(os.environ.get("VISION_IMAGE_ROOT", ".")) / p).resolve()

    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return Image.open(p).convert("RGB")


def label_for(image_ref):
    # dataset label string for an HF image, or None if not available
    if not image_ref.startswith("hf://"):
        return None
    spec, split, idx = _parse_hf_ref(image_ref)

    if not spec.label_col:
        return None
    ds = _load_split(spec.repo, split)
    item = ds[idx]
    raw = item.get(spec.label_col)

    if raw is None:
        return None
    # isolate errors so the rest of the call can bail cleanly
    try:
        return ds.features[spec.label_col].int2str(raw)
    except (AttributeError, KeyError, TypeError):
        return str(raw)


def domain_hint_for(image_ref):
    # registered domain_hint string (None for paths/URLs)
    if not image_ref.startswith("hf://"):
        return None
    spec, _, _ = _parse_hf_ref(image_ref)
    return spec.domain_hint


def taxonomy_for(image_ref):
    # registered prompt_taxonomy or empty tuple
    if not image_ref.startswith("hf://"):
        return ()
    spec, _, _ = _parse_hf_ref(image_ref)
    return spec.prompt_taxonomy

