"""Runtime context shared by the write path, read path and graph."""
from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field

from .config import Settings, get_settings
from .llm.base import Embedder, LLMClient
from .store.sqlite import MemoryStore


def _today() -> str:
    return dt.date.today().isoformat()  # noqa: DTZ011


@dataclass
class Context:
    store: MemoryStore
    llm: LLMClient
    embedder: Embedder
    settings: Settings = field(default_factory=get_settings)
    # Injectable so evals can simulate months of history (system time).
    clock: Callable[[], str] = _today

    def now(self) -> str:
        return self.clock()
