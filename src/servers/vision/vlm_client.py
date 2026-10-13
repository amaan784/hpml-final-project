# Async OpenAI-compatible client for vLLM with multimodal content.
# Env: VLM_BASE_URL, VLM_API_KEY, VLM_MODEL.

import base64
import io
import os

from PIL import Image

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

_client = None  # lazy AsyncOpenAI

# Diagnostics for the most recent vlm_call: finish_reason ("stop" vs "length"
# tells a complete answer from one cut at max_tokens) and token counts
# (prompt_tokens grows with image size, so it exposes image-token cost). The
# benchmark harness calls the MCP tools in-process and reads this right after
# each call; it is deliberately not part of any tool's return schema, so
# agent-facing outputs are unchanged. Mutated in place, never rebound.
LAST_CALL: dict = {}


def _get_client():
    # `get_client` lives here. was getting too cramped inline.
    global _client
    if _client is None:
        from openai import AsyncOpenAI

        _client = AsyncOpenAI(
            base_url=os.environ.get("VLM_BASE_URL", DEFAULT_BASE_URL),
            api_key=os.environ.get("VLM_API_KEY", "EMPTY"),
        )
    return _client


def _max_side():
    # resolution cap. override per-variant via VLM_IMAGE_MAX_SIDE
    raw = os.environ.get("VLM_IMAGE_MAX_SIDE")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return 1024


def _encode_image_b64(img, max_side=None):
    cap = max_side if max_side is not None else _max_side()

    if max(img.size) > cap:
        img = img.copy()
        img.thumbnail((cap, cap))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


async def vlm_call(prompt, image, max_tokens=256, temperature=0.0):
    # send one (text + image) chat-completion and return the assistant text
    LAST_CALL.clear()  # a failed call must not leave the previous call's stats
    client = _get_client()
    model = os.environ.get("VLM_MODEL", DEFAULT_MODEL)
    b64 = _encode_image_b64(image)
    resp = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
    )
    choice = resp.choices[0]
    usage = getattr(resp, "usage", None)
    LAST_CALL.update({
        "finish_reason": getattr(choice, "finish_reason", None) or "",
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    })
    return choice.message.content or ""

