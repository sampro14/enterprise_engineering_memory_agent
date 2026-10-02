"""Gemini adapter (LLM + embeddings). Retries/caching are applied by the factory, not here."""
from __future__ import annotations

import numpy as np
from google import genai
from google.genai import types
from pydantic import ValidationError

from ..base import StructuredOutputError, T


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) or 1.0)


class GeminiLLM:
    def __init__(self, api_key: str, model: str, client=None):
        self.model = model
        self.client = client or genai.Client(api_key=api_key)

    def _cfg(self, system: str | None, **kw) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system, temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=0), **kw,
        )

    def generate(self, prompt: str, system: str | None = None) -> str:
        r = self.client.models.generate_content(model=self.model, contents=prompt, config=self._cfg(system))
        return r.text or ""

    def generate_json(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        r = self.client.models.generate_content(
            model=self.model, contents=prompt,
            config=self._cfg(system, response_mime_type="application/json", response_schema=schema),
        )
        try:
            return schema.model_validate_json(r.text)
        except (ValidationError, ValueError, TypeError) as e:
            raise StructuredOutputError(f"gemini returned output that does not match {schema.__name__}: {e}") from e


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str, dim: int, client=None):
        self.model, self.dim = model, dim
        self.client = client or genai.Client(api_key=api_key)

    def embed(self, texts: list[str]) -> np.ndarray:
        kwargs = {"config": types.EmbedContentConfig(output_dimensionality=self.dim)} if self.dim != 3072 else {}
        r = self.client.models.embed_content(model=self.model, contents=texts, **kwargs)
        vecs = np.vstack([_normalize(np.asarray(e.values, dtype=np.float32)) for e in r.embeddings])
        if vecs.shape[1] != self.dim:
            raise ValueError(f"gemini returned {vecs.shape[1]}-dim embeddings but EMBED_DIM={self.dim}")
        return vecs
