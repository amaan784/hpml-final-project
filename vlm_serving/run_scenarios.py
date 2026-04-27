"""Run every scenario in scenarios/vlm_impeller_scenarios.json through the
Qwen2.5-VL vLLM endpoint and save the responses.

This is the L0 (FP16) baseline producer. The exact same script is reused for
L1 (AWQ) and L2 (serving-tuned) by pointing VLLM_MODEL / VLLM_BASE_URL at the
respective server.

Run (after start_vllm_server.sh is up):
    python vlm_serving/run_scenarios.py \
        --scenarios scenarios/vlm_impeller_scenarios.json \
        --out qwen_vllm_results.json

Env:
    VLLM_BASE_URL (default http://localhost:8000/v1)
    VLLM_MODEL    (default Qwen/Qwen2.5-VL-7B-Instruct)
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path

from openai import OpenAI


def image_to_data_url(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(image_path.name)
    if mime is None:
        mime = "image/jpeg"
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def analyze_pump_impeller(
    client: OpenAI, model: str, image_path: Path, question: str, max_tokens: int
) -> dict:
    """One scenario -> one model call. This is the function we will later expose
    as an MCP tool (step 4)."""
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    {"type": "text", "text": question},
                ],
            }
        ],
    )
    elapsed = time.perf_counter() - t0
    return {
        "response": resp.choices[0].message.content,
        "latency_s": elapsed,
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
        "completion_tokens": getattr(resp.usage, "completion_tokens", None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True, help="Path to vlm_impeller_scenarios.json")
    ap.add_argument("--out", required=True, help="Where to write the results JSON")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    scenarios_path = Path(args.scenarios)
    out_path = Path(args.out)
    payload = json.loads(scenarios_path.read_text(encoding="utf-8"))
    scenarios = payload["scenarios"]

    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    model = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    client = OpenAI(base_url=base_url, api_key="EMPTY")

    print(f"Endpoint: {base_url}")
    print(f"Model:    {model}")
    print(f"Scenarios to run: {len(scenarios)}")
    print()

    results = {
        "_meta": {
            "model": model,
            "base_url": base_url,
            "scenarios_file": str(scenarios_path),
            "n_scenarios": len(scenarios),
        },
        "results": {},
    }

    for s in scenarios:
        sid = s["id"]
        img = Path(s["image_path"])
        q = s["text"]

        print(f"[{sid}] {img.name}")
        if not img.exists():
            print(f"  MISSING IMAGE: {img}", file=sys.stderr)
            results["results"][sid] = {"error": f"missing image: {img}"}
            continue

        try:
            out = analyze_pump_impeller(client, model, img, q, args.max_tokens)
        except Exception as e:  # noqa: BLE001 - we want to keep going and capture the error
            print(f"  ERROR: {e}", file=sys.stderr)
            results["results"][sid] = {"error": str(e)}
            continue

        print(f"  latency={out['latency_s']:.2f}s  "
              f"prompt_tok={out['prompt_tokens']}  "
              f"completion_tok={out['completion_tokens']}")
        print(f"  -> {out['response'][:200]}{'...' if len(out['response']) > 200 else ''}")
        print()

        results["results"][sid] = {
            "image_path": str(img),
            "question": q,
            "expected_answer": s.get("expected_answer"),
            **out,
        }

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
