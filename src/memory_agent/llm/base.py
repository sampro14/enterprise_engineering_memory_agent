"""Provider-agnostic LLM and embedding interfaces."""
from __future__ import annotations

from typing import Protocol, TypeVar

import numpy as np
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base class for provider failures surfaced by this package."""


class TransientLLMError(LLMError):
    """Retryable failure (rate limit, timeout, 5xx)."""


class StructuredOutputError(LLMError):
    """The model did not return output matching the requested schema."""


class ProviderConfigError(LLMError):
    """Missing key / unknown provider / bad configuration."""


class LLMClient(Protocol):
    model: str

    def generate(self, prompt: str, system: str | None = None) -> str: ...

    def generate_json(self, prompt: str, schema: type[T], system: str | None = None) -> T: ...


class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized vectors."""
        ...
