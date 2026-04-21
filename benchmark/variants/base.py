# Variant registry for HPML optimization experiments.
# Each technique lives in its own module under benchmark/variants/ and
# calls register(Variant(...)). Adding a technique = a single new file.

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Variant:
    # name format: <level>_<technique>, e.g. L1_awq_w4a16_domain.
    # model_id placeholders ($AWQ_DOMAIN, $AWQ_GENERIC, ...) are resolved
    # by resolve_model_id() at serve time.
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


REGISTRY: dict = {}


def register(v):
    if v.name in REGISTRY:
        raise ValueError(f"Variant {v.name!r} already registered")
    REGISTRY[v.name] = v


def get(name):
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown variant {name!r}. Known: {sorted(REGISTRY)}."
        )
    return REGISTRY[name]


def all_variants():
    # CLI/helper entry for `all_variants`.
    return [REGISTRY[k] for k in sorted(REGISTRY)]


def by_family(family):
    return [v for v in all_variants() if v.family == family]


_PLACEHOLDERS = {
    "$AWQ_DOMAIN":        "/opt/models/qwen2.5-vl-7b-awq-domain",
    "$AWQ_GENERIC":       "/opt/models/qwen2.5-vl-7b-awq-generic",
    "$AWQ_W8A8":          "/opt/models/qwen2.5-vl-7b-w8a8",
    "$AWQ_LLAMA_DOMAIN":  "/opt/models/llama3-llava-next-8b-awq-domain-real",
    "$AWQ_LLAMA_GENERIC": "/opt/models/llama3-llava-next-8b-awq-generic-real",
}


def resolve_model_id(spec):
    # expand $AWQ_* placeholders to concrete paths (override via env vars)
    import os
    # process records in deterministic order
    for placeholder, default in _PLACEHOLDERS.items():
        if spec == placeholder:
            return os.environ.get(placeholder.lstrip("$"), default)
    return spec
