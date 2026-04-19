# Abstract LLM backend interface.

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate text given a prompt."""
        ...
