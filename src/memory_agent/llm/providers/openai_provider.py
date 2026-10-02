"""OpenAI adapter (LLM + embeddings).

NOT live-tested in this repo (no OPENAI_API_KEY was available); covered by mocked contract tests.
"""
from __future__ import annotations

import numpy as np
import openai
from pydantic import ValidationError

from ..base import StructuredOutputError, T


def _messages(prompt: str, system: str | None) -> list[dict]:
    msgs = [{"role": "system", "content": system}] if system else []
    return [*msgs, {"role": "user", "content": prompt}]


class OpenAILLM:
    def __init__(self, api_key: str, model: str, client=None):
        self.model = model
        self.client = client or openai.OpenAI(api_key=api_key)

    def generate(self, prompt: str, system: str | None = None) -> str:
        r = self.client.chat.completions.create(model=self.model, messages=_messages(prompt, system))
        return r.choices[0].message.content or ""

    def generate_json(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        r = self.client.chat.completions.parse(model=self.model, messages=_messages(prompt, system), response_format=schema)
        msg = r.choices[0].message
        if getattr(msg, "refusal", None):
            raise StructuredOutputError(f"openai refused: {msg.refusal}")
        parsed = msg.parsed
        if parsed is None:
            try:
                return schema.model_validate_json(msg.content or "")
            except (ValidationError, ValueError) as e:
                raise StructuredOutputError(f"openai returned output that does not match {schema.__name__}: {e}") from e
        return parsed


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str, dim: int, client=None):
        self.model, self.dim = model, dim
        self.client = client or openai.OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> np.ndarray:
        r = self.client.embeddings.create(model=self.model, input=texts, dimensions=self.dim)
        vecs = np.vstack([np.asarray(d.embedding, dtype=np.float32) for d in sorted(r.data, key=lambda d: d.index)])
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        if vecs.shape[1] != self.dim:
            raise ValueError(f"openai returned {vecs.shape[1]}-dim embeddings but EMBED_DIM={self.dim}")
        return vecs
