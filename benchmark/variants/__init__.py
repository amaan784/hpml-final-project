# auto-discover every variant module and expose the registry

import importlib
import pkgutil

from .base import (
    REGISTRY,
    Variant,
    all_variants,
    by_family,
    get,
    register,
    resolve_model_id,
)

_SKIP = {"base"}

for _info in pkgutil.iter_modules(__path__):
    if _info.name in _SKIP or _info.name.startswith("_"):
        continue
    importlib.import_module(f"{__name__}.{_info.name}")

del importlib, pkgutil
