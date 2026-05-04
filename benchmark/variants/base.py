# Variant registry: each module calls register(Variant(...)).

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Variant:
    name: str
    short: str
    family: str
    description: str
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vllm_extra: tuple[str, ...] = ()
    image_max_side: int = 1024
    env: dict[str, str] = field(default_factory=dict)
    requires: tuple[str, ...] = ()
    how_it_works: str = ""
    howto_test: str = ""


REGISTRY: dict[str, Variant] = {}


def register(v: Variant) -> None:
    if v.name in REGISTRY:
        raise ValueError(f"Variant {v.name!r} already registered")
    REGISTRY[v.name] = v


def get(name: str) -> Variant:
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown variant {name!r}. "
            f"Known: {sorted(REGISTRY)}. "
            f"List with: python -m benchmark.variants list"
        )
    return REGISTRY[name]


def all_variants() -> list[Variant]:
    # CLI/helper entry for `all_variants`.
    return [REGISTRY[k] for k in sorted(REGISTRY)]


def by_family(family: str) -> list[Variant]:
    return [v for v in all_variants() if v.family == family]


# Placeholder resolution
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

_PLACEHOLDERS = {
    "$AWQ_DOMAIN":        str(_DEFAULT_MODEL_DIR / "qwen2.5-vl-7b-awq-domain"),
    "$AWQ_GENERIC":       str(_DEFAULT_MODEL_DIR / "qwen2.5-vl-7b-awq-generic"),
    "$AWQ_LLAMA_DOMAIN":  str(_DEFAULT_MODEL_DIR / "llama3-llava-next-8b-awq-domain-real"),
    "$AWQ_LLAMA_GENERIC": str(_DEFAULT_MODEL_DIR / "llama3-llava-next-8b-awq-generic-real"),
    # W8A8 entries removed when those variants were dropped
}


def resolve_model_id(spec: str) -> str:
    import os
    for placeholder, default in _PLACEHOLDERS.items():
        if spec == placeholder:
            value = os.environ.get(placeholder.lstrip("$"), default)
            return os.path.expanduser(os.path.expandvars(value))
    return os.path.expanduser(os.path.expandvars(spec))
