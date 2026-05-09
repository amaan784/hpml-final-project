import json
import os
from unittest.mock import patch

import pytest
from PIL import Image


requires_vlm = pytest.mark.skipif(
    os.environ.get("VLM_BASE_URL") is None,
    reason="No live VLM endpoint (set VLM_BASE_URL to enable integration tests)",
)


@pytest.fixture
def anyio_backend():
    # walk through `anyio_backend`, kept separate so the main flow stays readable.
    return "asyncio"


async def call_tool(mcp_instance, tool_name: str, args: dict) -> dict:
    contents, _ = await mcp_instance.call_tool(tool_name, args)
    return json.loads(contents[0].text)


@pytest.fixture
def fake_image(monkeypatch):
    img = Image.new("RGB", (64, 64), color=(120, 120, 120))

    def _load(image_ref):
        # `load` lives here. was getting too cramped inline.
        return img

    monkeypatch.setattr("servers.vision.image_loader.load_image", _load)
    return img


@pytest.fixture
def mock_vlm():
    queue: list = []

    async def _vlm(prompt, image, max_tokens=256, temperature=0.0):
        # Async bit: `vlm` lives here. was getting too cramped inline.
        if not queue:
            raise AssertionError("mock_vlm response queue empty")
        nxt = queue.pop(0)
        return nxt(prompt, image) if callable(nxt) else nxt

    with patch("servers.vision.main.vlm_client.vlm_call", _vlm):
        yield queue


@pytest.fixture
def vlm_error():
    # `vlm_error` lives here. was getting too cramped inline.
    async def _raise(*args, **kwargs):
        # Async bit: does `raise`, split out so we can reuse it from a few call sites.
        raise RuntimeError("connection refused")

    with patch("servers.vision.main.vlm_client.vlm_call", _raise):
        yield
