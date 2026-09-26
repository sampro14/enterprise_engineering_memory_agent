"""Deterministic offline LLM/embedder for tests."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

import numpy as np

from .base import T


class FakeLLM:
    """handler(prompt, schema) -> instance of schema. Records calls for assertions."""

    def __init__(self, handler: Callable[[str, type], object] | None = None):
        self.handler = handler
        self.calls: list[str] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls.append(prompt)
        return self.handler(prompt, str) if self.handler else ""  # type: ignore[return-value]

    def generate_json(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        self.calls.append(prompt)
        assert self.handler is not None, "FakeLLM needs a handler for generate_json"
        return self.handler(prompt, schema)  # type: ignore[return-value]


class FakeEmbedder:
    """Hashed bag-of-words embedding: similar token sets -> high cosine similarity."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)
