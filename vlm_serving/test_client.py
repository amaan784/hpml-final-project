"""Smoke test for the vLLM Qwen2.5-VL server.

Sends ONE impeller image and ONE question to the OpenAI-compatible endpoint
and prints the response. Use this to verify the server is up before running
the full scenario sweep.

Run:
    python vlm_serving/test_client.py
    python vlm_serving/test_client.py --image pump_data/defective/cast_def_0_1055.jpeg

Env:
    VLLM_BASE_URL (default http://localhost:8000/v1)
    VLLM_MODEL    (default Qwen/Qwen2.5-VL-7B-Instruct)
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import sys
from pathlib import Path

from openai import OpenAI

DEFAULT_IMAGE = "pump_data/defective/cast_def_0_1055.jpeg"
DEFAULT_QUESTION = (
    "This is a top-view image of a cast submersible pump impeller. "
    "Is this casting defective or acceptable? Answer in one sentence."
)


def image_to_data_url(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(image_path.name)
    if mime is None:
        mime = "image/jpeg"
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1

    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    model = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

    # vLLM does not check the key, but the SDK requires one to be set.
    client = OpenAI(base_url=base_url, api_key="EMPTY")

    resp = client.chat.completions.create(
        model=model,
        max_tokens=args.max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    {"type": "text", "text": args.question},
                ],
            }
        ],
    )

    print(f"--- {model} @ {base_url} ---")
    print(f"Image:    {image_path}")
    print(f"Question: {args.question}")
    print()
    print("Response:")
    print(resp.choices[0].message.content)
    usage = resp.usage
    if usage is not None:
        print()
        print(f"Tokens: prompt={usage.prompt_tokens}  completion={usage.completion_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
