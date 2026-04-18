# Unified LLM backend via the litellm library. Provider is encoded in the
# model-string prefix, e.g. "watsonx/meta-llama/..." or "litellm_proxy/GCP/...".
# Credentials come from env vars: WATSONX_APIKEY/WATSONX_PROJECT_ID for watsonx/*,
# LITELLM_API_KEY/LITELLM_BASE_URL otherwise.

import os

from .base import LLMBackend


class LiteLLMBackend(LLMBackend):
    def __init__(self, model_id: str) -> None:
        # remember which model string litellm will route
        self._model_id = model_id

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        # import here so jobs that only use vllm avoid the extra dependency
        import litellm

        # same message shape no matter which provider
        kwargs: dict = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 2048,
        }

        if self._model_id.startswith("watsonx/"):
            # watsonx needs project id and its own base url
            kwargs["api_key"] = os.environ["WATSONX_APIKEY"]
            kwargs["project_id"] = os.environ["WATSONX_PROJECT_ID"]
            if url := os.environ.get("WATSONX_URL"):
                kwargs["api_base"] = url
        else:
            # default path hits our litellm proxy with shared secrets
            kwargs["api_key"] = os.environ["LITELLM_API_KEY"]
            kwargs["api_base"] = os.environ["LITELLM_BASE_URL"]

        # one completion call then read assistant text
        response = litellm.completion(**kwargs)
        return response.choices[0].message.content
