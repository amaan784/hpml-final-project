# Deterministic-seed helper for the benchmark + quantization entry points.

from __future__ import annotations

import os
import random


def set_all_seeds(seed: int = 42, strict: bool = False) -> int:
    """Seed random / numpy / torch / cudnn so re-runs are reproducible.

    ``strict=True`` additionally calls ``torch.use_deterministic_algorithms``,
    which forces deterministic kernel selection across torch ops. vLLM 0.19
    and llmcompressor 0.10 invoke a number of ops (scatter_reduce, certain
    fused-attention paths) that have no deterministic implementation; with
    ``warn_only=True`` those emit a warning per call, which floods stderr
    during a long bench sweep. Default keeps the soft path (cudnn flags +
    seeded RNGs) -- it's enough for the GPTQ Hessian solve and our reported
    reproducibility, without log spam.
    """
    # NOTE: PYTHONHASHSEED set at runtime only affects CHILD processes
    # (e.g. dataloader workers) -- the current interpreter's str-hash
    # randomization was fixed at startup. Nothing in this pipeline depends
    # on set/dict iteration order, so that limitation is acceptable.
    # CUBLAS_WORKSPACE_CONFIG only takes effect under
    # torch.use_deterministic_algorithms (strict=True); harmless otherwise.
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if strict:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except (AttributeError, RuntimeError):
                pass
    except ImportError:
        pass

    return seed
