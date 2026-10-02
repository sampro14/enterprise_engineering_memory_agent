"""Cross-cutting behavior applied once, around any provider: retries and on-disk caching."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from .base import Embedder, LLMClient, T, TransientLLMError
from .cache import DiskCache

log = logging.getLogger(__name__)

_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
_TRANSIENT_NAMES = ("ratelimit", "timeout", "connection", "unavailable", "overloaded", "internalserver", "servererror")


def default_is_transient(exc: BaseException) -> bool:
    if isinstance(exc, TransientLLMError):
        return True
    for attr in ("status_code", "code", "http_status"):
        code = getattr(exc, attr, None)
        if isinstance(code, int) and code in _TRANSIENT_STATUS:
            return True
    name = type(exc).__name__.lower()
    return any(n in name for n in _TRANSIENT_NAMES)


def _retry(fn: Callable[[], Any], attempts: int, base_delay: float, sleep: Callable[[float], None],
           is_transient: Callable[[BaseException], bool]) -> Any:
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1 or not is_transient(e):
                raise
            delay = min(base_delay * 2**i, 20.0)
            log.warning("transient provider error (%s); retry %d/%d in %.1fs", type(e).__name__, i + 1, attempts - 1, delay)
            sleep(delay)


class RetryingLLM:
    def __init__(self, inner: LLMClient, attempts: int = 5, base_delay: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep,
                 is_transient: Callable[[BaseException], bool] = default_is_transient):
        self.inner, self.attempts, self.base_delay, self.sleep, self.is_transient = inner, attempts, base_delay, sleep, is_transient
        self.model = inner.model

    def generate(self, prompt: str, system: str | None = None) -> str:
        return _retry(lambda: self.inner.generate(prompt, system), self.attempts, self.base_delay, self.sleep, self.is_transient)

    def generate_json(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        return _retry(lambda: self.inner.generate_json(prompt, schema, system), self.attempts, self.base_delay,
                      self.sleep, self.is_transient)


class CachedLLM:
    """Key layout is stable (and matches the PoC), so existing caches stay valid."""

    def __init__(self, inner: LLMClient, cache: DiskCache):
        self.inner, self.cache = inner, cache
        self.model = inner.model

    def generate(self, prompt: str, system: str | None = None) -> str:
        k = DiskCache.key("gen", self.model, system or "", prompt)
        if (hit := self.cache.get(k)) is not None:
            return hit.decode()
        text = self.inner.generate(prompt, system)
        self.cache.set(k, text.encode())
        return text

    def generate_json(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        k = DiskCache.key("json", self.model, system or "", schema.__name__, prompt)
        if (hit := self.cache.get(k)) is not None:
            return schema.model_validate_json(hit)
        obj = self.inner.generate_json(prompt, schema, system)
        self.cache.set(k, obj.model_dump_json().encode())
        return obj


class RetryingEmbedder:
    def __init__(self, inner: Embedder, attempts: int = 5, base_delay: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep,
                 is_transient: Callable[[BaseException], bool] = default_is_transient):
        self.inner, self.attempts, self.base_delay, self.sleep, self.is_transient = inner, attempts, base_delay, sleep, is_transient
        self.model, self.dim = inner.model, inner.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        return _retry(lambda: self.inner.embed(texts), self.attempts, self.base_delay, self.sleep, self.is_transient)


class CachedEmbedder:
    """Per-text cache; only cache misses reach the provider, in batches."""

    def __init__(self, inner: Embedder, cache: DiskCache, batch_size: int = 50):
        self.inner, self.cache, self.batch_size = inner, cache, batch_size
        self.model, self.dim = inner.model, inner.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[np.ndarray | None] = [None] * len(texts)
        todo: list[int] = []
        for i, t in enumerate(texts):
            hit = self.cache.get(DiskCache.key("emb", self.model, t))
            if hit is not None and len(hit) == 4 * self.dim:
                out[i] = np.frombuffer(hit, dtype=np.float32)
            else:
                todo.append(i)
        for start in range(0, len(todo), self.batch_size):
            batch = todo[start : start + self.batch_size]
            vecs = self.inner.embed([texts[i] for i in batch])
            for i, v in zip(batch, vecs, strict=True):
                out[i] = v
                self.cache.set(DiskCache.key("emb", self.model, texts[i]), v.astype(np.float32).tobytes())
        return np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)
