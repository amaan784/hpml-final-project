# Async OpenAI-compatible client for vLLM with multimodal content.
# Env: VLM_BASE_URL, VLM_API_KEY, VLM_MODEL.

import base64
import io
import os

from PIL import Image

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

_client = None  # lazy AsyncOpenAI


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
    return resp.choices[0].message.content or ""

