# Single-request profiling target for Nsight Systems.
# Wraps one VLM chat completion in NVTX ranges. Run once per L-variant on the
# VM with vLLM already running the matching model.
# Example:
#   nsys profile --trace=cuda,nvtx,osrt --output=l0.qdrep python benchmark/profile_single.py

import argparse
import asyncio
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from servers.vision import image_loader, vlm_client


def _nvtx():
    # return (push, pop) NVTX helpers. no-op if torch is unavailable
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop
    except Exception:
        pass
    return (lambda *_: None), (lambda *_: None)


async def _run(image_ref, prompt, warmups):
    # Async bit: `run` lives here. was getting too cramped inline.
    push, pop = _nvtx()
    img = image_loader.load_image(image_ref)

    push("warmup")
    # each pass handles the next item in the sequence
    for _ in range(warmups):
        await vlm_client.vlm_call(prompt, img, max_tokens=8)
    pop()

    push("vlm_request")
    t0 = time.perf_counter()
    text = await vlm_client.vlm_call(prompt, img, max_tokens=128)
    dt_ms = (time.perf_counter() - t0) * 1000
    pop()

    print(f"e2e_ms={dt_ms:.1f}  text_len={len(text)}")
    print(f"first_chars={text[:120]!r}")


def main():
    # CLI/helper entry for `main`.
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="hf://substation/train/0")
    p.add_argument("--prompt", default="What equipment is shown in this image?")
    p.add_argument("--warmups", type=int, default=2)
    args = p.parse_args()
    asyncio.run(_run(args.image, args.prompt, args.warmups))
    return 0

if __name__ == "__main__":
    sys.exit(main())

