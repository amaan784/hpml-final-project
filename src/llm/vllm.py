# Direct OpenAI-compatible client for a local vLLM endpoint.
# Two modes: generate(prompt) -> str (LLMBackend), and
# generate_with_metrics(prompt) -> (str, GenerationMetrics).
# Metrics path uses streaming + include_usage to capture TTFT and decode time
# for the NFR CSV. Env vars: VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY.

import os
import time
from dataclasses import dataclass
from typing import Optional

from .base import LLMBackend

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


@dataclass
class GenerationMetrics:
    e2e_s: float
    ttft_s: Optional[float]
    decode_s: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    decode_tps: Optional[float]


class VLLMBackend(LLMBackend):
    # direct vLLM client via the openai SDK (no litellm provider routing)

    def __init__(
        self,
        model_id: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        # caller args win over env and client starts unset
        self._model_id = model_id or os.environ.get("VLLM_MODEL", DEFAULT_MODEL)
        self._base_url = base_url or os.environ.get("VLLM_BASE_URL", DEFAULT_BASE_URL)
        self._api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        self._client = None

    @property
    def model_id(self) -> str:
        # which model string we hand to the server
        return self._model_id

    def _get_client(self):
        # import openai and build client on first call only
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        # simple path that matches LLMBackend
        client = self._get_client()
        # wait for the full reply in one response object
        resp = client.chat.completions.create(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""

    def generate_with_metrics(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> tuple[str, GenerationMetrics]:
        # streaming call so we can time first token and still read usage
        client = self._get_client()
        t_start = time.perf_counter()
        # include_usage asks for token totals on the stream
        stream = client.chat.completions.create(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        # running totals updated as chunks arrive
        ttft: Optional[float] = None
        text_chunks: list[str] = []
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None

        # some chunks carry usage only some carry deltas
        for chunk in stream:
            if chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)

            if content:
                if ttft is None:
                    # first real text ends the prefill wait
                    ttft = time.perf_counter() - t_start
                text_chunks.append(content)

        e2e = time.perf_counter() - t_start
        text = "".join(text_chunks)
        # decode time is everything after first token when we saw one
        decode_s = (e2e - ttft) if ttft is not None else None
        decode_tps = (
            completion_tokens / decode_s
            if decode_s and decode_s > 0 and completion_tokens
            else None
        )
        # shape NFR fields for the CSV layer
        metrics = GenerationMetrics(
            e2e_s=e2e,
            ttft_s=ttft,
            decode_s=decode_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            decode_tps=decode_tps,
        )
        return text, metrics

