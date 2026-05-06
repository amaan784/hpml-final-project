"""Image loader for the vision MCP server.

Resolves an ``image_ref`` string to a PIL.Image. Supported forms:
  - ``hf://<alias>/<split>/<index>``  e.g. ``hf://transformer/train/3``
  - ``hf://<alias>/<index>``          alias for the dataset's default split
  - ``/abs/path/to/file.jpg``         absolute filesystem path
  - ``file://...``                    file URI
  - ``http(s)://...``                 remote URL (downloaded once)

# Two registries: HF (calibration) + local (scenario priming)

There are two flavors of dataset:

1. **HuggingFace datasets** (``DatasetSpec``) - large, indexable by ``hf://``.
   Used for AWQ calibration (the quantization scripts pull 32-128 samples per
   dataset) and for scenarios that reference HF images directly.

   register_dataset(DatasetSpec(
       alias="transformer",
       repo="AndrzejDD/15-class-Substation-Equipment",
       domain_hint="...",
       prompt_taxonomy=("transformer", "circuit breaker", ...),
   ))

2. **Local-filesystem datasets** (``LocalDatasetSpec``) - small, hand-curated
   subset under ``data/<alias>/<class>/``. Too small for calibration; used to
   register *metadata* (domain hint, prompt taxonomy) so the VLM gets domain
   priming for hand-authored scenarios that reference local files.

   register_local_dataset(LocalDatasetSpec(
       alias="pumps",
       root="data/pumps",
       domain_hint="cast submersible pump impeller, top-view ...",
       prompt_taxonomy=("acceptable", "defective", "burr", ...),
   ))

The new alias is now usable in scenario JSON (``hf://motor/...`` for HF or
``data/pumps/<class>/<file>.jpeg`` for local), in the AWQ calibration scripts
(``--datasets motor transformer``), and in the benchmark harness without any
further changes. ``domain_hint_for(image_ref)`` works on both flavors.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import requests
from PIL import Image


# Dataset registry
@dataclass(frozen=True)
class DatasetSpec:
    """Declarative description of a HuggingFace image dataset.

    Attributes:
        alias:          Short name used in image_refs (``hf://<alias>/...``).
        repo:           HuggingFace dataset repo id.
        default_split:  Split used when an image_ref omits it.
        image_col:      Dataset column holding the PIL image.
        label_col:      Optional column holding an integer / string label.
        domain_hint:    One-sentence description of what's in the images.
                        Used by the generic VLM prompt and AWQ calibration prompt.
        prompt_taxonomy: Optional list of canonical class labels for prompt
                        seeding. Empty -> let the VLM answer freely.
    """

    alias: str
    repo: str
    default_split: str = "train"
    image_col: str = "image"
    label_col: str | None = "label"
    domain_hint: str = "industrial equipment image"
    prompt_taxonomy: tuple[str, ...] = field(default_factory=tuple)


_REGISTRY: dict[str, DatasetSpec] = {}


def register_dataset(spec: DatasetSpec) -> None:
    """Register (or overwrite) a dataset by its alias."""
    _REGISTRY[spec.alias] = spec


def get_spec(alias: str) -> DatasetSpec:
    if alias not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset alias {alias!r}. "
            f"Known: {sorted(_REGISTRY)}. "
            f"Register with register_dataset(DatasetSpec(...)) - see image_loader.py."
        )
    return _REGISTRY[alias]


def list_aliases() -> list[str]:
    # does `list_aliases`, split out so we can reuse it from a few call sites.
    return sorted(_REGISTRY)


# Built-in registrations
# Eric's part of the project. Teammates add their own here (or via
# ``register_dataset()`` in their own scenario module).

register_dataset(DatasetSpec(
    alias="transformer",
    repo="AndrzejDD/15-class-Substation-Equipment",
    default_split="train",
    image_col="image",
    label_col=None,  # this dataset ships only the 'image' column
    domain_hint="substation transformer / switchgear / insulator equipment",
    prompt_taxonomy=(
        "transformer", "circuit breaker", "disconnector", "lightning arrester",
        "current transformer", "voltage transformer", "busbar", "capacitor bank",
        "insulator", "switchgear", "reactor", "surge arrester",
        "recloser", "isolator", "other",
    ),
))

# Back-compat: the original code referenced this alias before the rename.
register_dataset(DatasetSpec(
    alias="substation",
    repo="AndrzejDD/15-class-Substation-Equipment",
    default_split="train",
    image_col="image",
    label_col=None,
    domain_hint="substation transformer / switchgear / insulator equipment",
    prompt_taxonomy=_REGISTRY["transformer"].prompt_taxonomy,
))


# Local-filesystem dataset registry
# Hand-curated subsets under data/<alias>/<class>/. Used for scenario priming.
# too small for AWQ calibration (which wants 100+ samples per dataset).


@dataclass(frozen=True)
class LocalDatasetSpec:
    """Declarative description of a local-filesystem image dataset.

    Attributes:
        alias:          Short name, e.g. ``pumps`` or ``turbine``.
        root:           Directory path relative to repo root, e.g. ``data/pumps``.
        domain_hint:    Multi-sentence description of what the model is looking
                        at and what counts as a defect. Used by the vision MCP
                        server to prime the VLM before answering questions.
        prompt_taxonomy: Canonical class labels for prompt seeding.
        classes:        Subdirectory class names, e.g. ``("defective", "fine")``
                        for ``data/pumps/{defective,fine}/``.
        owner:          Team member who authored / owns the dataset (for docs).
    """

    alias: str
    root: str
    domain_hint: str
    prompt_taxonomy: tuple[str, ...] = field(default_factory=tuple)
    classes: tuple[str, ...] = field(default_factory=tuple)
    owner: str = ""


_LOCAL_REGISTRY: dict[str, LocalDatasetSpec] = {}


def register_local_dataset(spec: LocalDatasetSpec) -> None:
    """Register (or overwrite) a local dataset by its alias."""
    _LOCAL_REGISTRY[spec.alias] = spec


def get_local_spec(alias: str) -> LocalDatasetSpec:
    if alias not in _LOCAL_REGISTRY:
        raise KeyError(
            f"Unknown local dataset alias {alias!r}. "
            f"Known: {sorted(_LOCAL_REGISTRY)}. "
            f"Register with register_local_dataset(LocalDatasetSpec(...))."
        )
    return _LOCAL_REGISTRY[alias]


def list_local_aliases() -> list[str]:
    return sorted(_LOCAL_REGISTRY)


def _classify_local_path(path: str) -> str | None:
    """Match a filesystem path to a registered local-dataset alias.

    Example: ``data/pumps/defective/cast_def_0_994.jpeg`` -> ``pumps``.
    """
    norm = path.replace("\\", "/").lower()
    # repeat for every element we need to touch
    for alias, spec in _LOCAL_REGISTRY.items():
        root = spec.root.replace("\\", "/").lower().rstrip("/")
        # Match either the bare root or root-as-segment (avoid prefix collisions).
        if (
            f"/{root}/" in f"/{norm}/"
            or norm.startswith(root + "/")
            or norm == root
        ):
            return alias
    return None


# Built-in local-dataset registrations.
# Add new ones here when teammates contribute hand-authored scenarios.

register_local_dataset(LocalDatasetSpec(
    alias="pumps",
    root="data/pumps",
    domain_hint=(
        "cast submersible pump impeller, top-view of the back/hub face. "
        "The dark area in the center is a normal through-hole bore (an "
        "opening), NOT a defect. Casting defects appear on the OUTER RIM "
        "(burrs, flash, chipping, irregular edge, material loss). The vanes "
        "are on the OPPOSITE face and are not visible in this view."
    ),
    prompt_taxonomy=(
        "acceptable", "defective",
        "burr", "flash", "chipping", "rim damage", "casting void",
        "irregular edge", "material loss", "porosity",
    ),
    classes=("defective", "fine"),
    owner="Amaan",
))

register_local_dataset(LocalDatasetSpec(
    alias="turbine",
    root="data/turbine",
    domain_hint=(
        "wind-turbine blade close-up (Blade30 dataset). Damage modes "
        "include leading-edge erosion, surface cracks, lightning strike "
        "damage, and gelcoat delamination. Normal blades show smooth, "
        "uniform gelcoat. Inspect for surface continuity, color/texture "
        "uniformity, and absence of crack lines or peeling layers."
    ),
    prompt_taxonomy=(
        "normal", "defective",
        "leading-edge erosion", "surface crack", "lightning damage",
        "delamination", "gelcoat damage", "trailing-edge damage",
    ),
    classes=("normal", "defective"),
    owner="Eric",
))

register_local_dataset(LocalDatasetSpec(
    alias="transformer-local",
    root="data/transformer",
    domain_hint=(
        "transformer / substation equipment image. Inspect for visible "
        "anomalies: oil leaks, corona discharge marks, bushing damage, "
        "thermal hotspots, corrosion, or cracked insulators."
    ),
    prompt_taxonomy=(
        "transformer", "circuit breaker", "disconnector", "lightning arrester",
        "current transformer", "voltage transformer", "busbar", "capacitor bank",
        "insulator", "switchgear", "reactor", "surge arrester",
        "recloser", "isolator", "other",
    ),
    classes=(),
    owner="Eric",
))

# Reference templates for teammates (commented out. uncomment + fill repo)
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


# Image loading
def _parse_hf_ref(ref: str) -> Tuple[DatasetSpec, str, int]:
    """Parse ``hf://<alias>/[<split>/]<index>`` -> (spec, split, index)."""
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
def _load_split(repo: str, split: str):
    # Memoize HF downloads; scenarios hit the same split many times per bench run.
    from datasets import load_dataset
    return load_dataset(repo, split=split)


def load_image(image_ref: str) -> Image.Image:
    """Resolve an image_ref to a PIL.Image (always RGB)."""
    # Hugging Face references stream a particular row out of DatasetSpec-backed repos.
    if image_ref.startswith("hf://"):
        spec, split, idx = _parse_hf_ref(image_ref)
        ds = _load_split(spec.repo, split)

        # Clamp index against the lazily-materialized HF split size.
        if idx < 0 or idx >= len(ds):
            raise IndexError(
                f"hf://{spec.alias}/{split}/{idx} out of range (size={len(ds)})"
            )
        item = ds[idx]
        img = item.get(spec.image_col)

        # Some HF cards omit the advertised column — try the usual suspects.
        if img is None:
            for fallback in ("image", "img", "png", "jpg"):
                img = item.get(fallback)
                if img is not None:
                    break

        # Still nothing usable — surface sample keys so we can tweak DatasetSpec.image_col.
        if img is None:
            raise KeyError(
                f"No image column found in {spec.repo}/{split}[{idx}]; "
                f"keys={list(item)}, expected '{spec.image_col}'"
            )
        return img.convert("RGB")

    # Plain URL download path for demos or remote fixtures.
    if image_ref.startswith(("http://", "https://")):
        resp = requests.get(image_ref, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    path = image_ref[len("file://"):] if image_ref.startswith("file://") else image_ref
    p = Path(path)

    # Allow repo-relative snippets when teammates pass bare data/... paths.
    if not p.is_absolute():
        p = (Path(os.environ.get("VISION_IMAGE_ROOT", ".")) / p).resolve()

    # Fail fast before PIL hands back a vague "cannot identify".
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return Image.open(p).convert("RGB")


def label_for(image_ref: str) -> str | None:
    """Return the dataset label string for an HF image, if available."""

    # Labels ride along only on curated HF shards (local files untracked).
    if not image_ref.startswith("hf://"):
        return None
    spec, split, idx = _parse_hf_ref(image_ref)

    # Some teaser datasets omit supervision entirely.
    if not spec.label_col:
        return None
    ds = _load_split(spec.repo, split)
    item = ds[idx]
    raw = item.get(spec.label_col)

    # Sparse rows genuinely lack a packed label tensor.
    if raw is None:
        return None
    # Fall back gracefully when Feature metadata is unconventional.
    try:
        return ds.features[spec.label_col].int2str(raw)
    except (AttributeError, KeyError, TypeError):
        return str(raw)


def domain_hint_for(image_ref: str) -> str | None:
    """Return the domain_hint string registered for the dataset.

    Works for both ``hf://`` references and local filesystem paths whose
    parent directory matches a registered ``LocalDatasetSpec.root``.
    Returns None for unregistered URLs or paths.
    """

    # HF registrations already stash the teammate-written domain blurb on DatasetSpec.
    if image_ref.startswith("hf://"):
        spec, _, _ = _parse_hf_ref(image_ref)
        return spec.domain_hint

    # Plain URLs have no curator metadata we can latch onto.
    if image_ref.startswith(("http://", "https://")):
        return None
    # Local file path - try to match against registered local datasets.
    path = image_ref[len("file://"):] if image_ref.startswith("file://") else image_ref
    alias = _classify_local_path(path)

    # Filename landed inside a teammate's curated local corpus.
    if alias:
        return _LOCAL_REGISTRY[alias].domain_hint
    return None


def taxonomy_for(image_ref: str) -> tuple[str, ...]:
    """Return the prompt_taxonomy registered for the dataset, or empty tuple.

    Works for both ``hf://`` references and local filesystem paths.
    """

    # hf:// lookups inherit whatever taxonomy DatasetSpec seeded.
    if image_ref.startswith("hf://"):
        spec, _, _ = _parse_hf_ref(image_ref)
        return spec.prompt_taxonomy

    # Remote URLs behave like anonymous assets — no scripted label hints.
    if image_ref.startswith(("http://", "https://")):
        return ()
    path = image_ref[len("file://"):] if image_ref.startswith("file://") else image_ref
    alias = _classify_local_path(path)

    # Local corpuses optionally pin deterministic class strings for prompting.
    if alias:
        return _LOCAL_REGISTRY[alias].prompt_taxonomy
    return ()

