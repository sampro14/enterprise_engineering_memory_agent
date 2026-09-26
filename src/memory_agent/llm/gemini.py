"""Gemini implementations of LLMClient and Embedder."""
from __future__ import annotations

import os
import time

import numpy as np
from google import genai
from google.genai import types

from ..config import Settings, get_settings
from .base import T
from .cache import DiskCache


def _retry(fn, attempts: int = 5):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # rate limits / transient errors
            if i == attempts - 1:
                raise
            time.sleep(min(2**i, 20))
            print(f"[gemini] retry {i + 1}: {type(e).__name__}")


class GeminiLLM:
    def __init__(self, settings: Settings | None = None, cache: DiskCache | None = None):
        self.s = settings or get_settings()
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self.cache = cache or DiskCache(self.s.cache_path)

    def _cfg(self, system: str | None, **kw) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            **kw,
        )

    def generate(self, prompt: str, system: str | None = None) -> str:
        k = DiskCache.key("gen", self.s.llm_model, system or "", prompt)
        if (hit := self.cache.get(k)) is not None:
            return hit.decode()
        r = _retry(
            lambda: self.client.models.generate_content(
                model=self.s.llm_model, contents=prompt, config=self._cfg(system)
            )
        )
        text = r.text or ""
        self.cache.set(k, text.encode())
        return text

    def generate_json(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        k = DiskCache.key("json", self.s.llm_model, system or "", schema.__name__, prompt)
        if (hit := self.cache.get(k)) is not None:
            return schema.model_validate_json(hit)
        r = _retry(
            lambda: self.client.models.generate_content(
                model=self.s.llm_model,
                contents=prompt,
                config=self._cfg(
                    system, response_mime_type="application/json", response_schema=schema
                ),
            )
        )
        obj = schema.model_validate_json(r.text)
        self.cache.set(k, obj.model_dump_json().encode())
        return obj


class GeminiEmbedder:
    def __init__(self, settings: Settings | None = None, cache: DiskCache | None = None):
        self.s = settings or get_settings()
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self.cache = cache or DiskCache(self.s.cache_path)

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[np.ndarray | None] = [None] * len(texts)
        todo: list[int] = []
        for i, t in enumerate(texts):
            hit = self.cache.get(DiskCache.key("emb", self.s.embed_model, t))
            if hit is not None:
                out[i] = np.frombuffer(hit, dtype=np.float32)
            else:
                todo.append(i)
        for start in range(0, len(todo), 50):
            batch = todo[start : start + 50]
            r = _retry(
                lambda b=batch: self.client.models.embed_content(
                    model=self.s.embed_model, contents=[texts[i] for i in b]
                )
            )
            for i, e in zip(batch, r.embeddings, strict=True):
                v = np.asarray(e.values, dtype=np.float32)
                v /= np.linalg.norm(v) or 1.0
                out[i] = v
                self.cache.set(DiskCache.key("emb", self.s.embed_model, texts[i]), v.tobytes())
        return np.vstack(out) if out else np.zeros((0, 1), dtype=np.float32)
