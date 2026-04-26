# Concurrent throughput probe for vLLM-served VLMs.
# Fires N copies of (image + prompt) concurrently and reports rps and p50/p95.
# Usage: python benchmark/concurrent_load.py --image hf://substation/train/0

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from servers.vision import image_loader, vlm_client

PROMPT = "Identify the equipment in this image in one word."


async def _one_request(img):
    # Async `one_request` isolated for readability.
    t0 = time.perf_counter()
    text = await vlm_client.vlm_call(PROMPT, img, max_tokens=16)
    return (time.perf_counter() - t0) * 1000, len(text)


async def _run_at_concurrency(img, n, repeats):
    latencies_ms = []
    t0 = time.perf_counter()
    for _ in range(repeats):
        results = await asyncio.gather(*[_one_request(img) for _ in range(n)])
        latencies_ms.extend(r[0] for r in results)
    wall_s = time.perf_counter() - t0
    completed = len(latencies_ms)
    return {
        "concurrency": n,
        "completed": completed,
        "wall_s": round(wall_s, 3),
        "rps": round(completed / wall_s, 2) if wall_s > 0 else 0.0,
        "p50_ms": round(statistics.median(latencies_ms), 1),
        "p95_ms": round(statistics.quantiles(latencies_ms, n=20)[-1], 1)
                  if len(latencies_ms) >= 20 else round(max(latencies_ms), 1),
    }


async def _main_async(image_ref, levels, repeats):
    # Helper for `main_async`.
    img = image_loader.load_image(image_ref)
    print(f"==> image_ref={image_ref}  levels={levels}  repeats={repeats}")
    print(f"==> VLM: {vlm_client.DEFAULT_BASE_URL}")
    print(f"{'concurrency':>11}  {'rps':>6}  {'p50_ms':>7}  {'p95_ms':>7}  {'wall_s':>7}")
    for n in levels:
        row = await _run_at_concurrency(img, n, repeats)
        print(f"{row['concurrency']:>11}  {row['rps']:>6}  "
              f"{row['p50_ms']:>7}  {row['p95_ms']:>7}  {row['wall_s']:>7}")


def main():
    # CLI/helper entry for `main`.
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="hf://substation/train/0")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--repeats", type=int, default=2,
                   help="Number of full-concurrency rounds at each level")
    args = p.parse_args()
    asyncio.run(_main_async(args.image, args.levels, args.repeats))
    return 0

if __name__ == "__main__":
    sys.exit(main())

