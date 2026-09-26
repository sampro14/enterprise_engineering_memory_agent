"""Provider-agnostic LLM and embedding interfaces."""
from __future__ import annotations

from typing import Protocol, TypeVar

import numpy as np
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def generate(self, prompt: str, system: str | None = None) -> str: ...

    def generate_json(self, prompt: str, schema: type[T], system: str | None = None) -> T: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float32 array of L2-normalized vectors."""
        ...
