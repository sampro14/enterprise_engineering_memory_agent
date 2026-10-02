"""Environment-driven settings."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class ConfidenceWeights:
    """Weights for the confidence formula (design 14C). Hyperparameters, to be tuned."""

    authority: float = 0.40
    sources: float = 0.20
    verification: float = 0.20
    recency: float = 0.20
    contradiction: float = 0.30


@dataclass(frozen=True)
class Settings:
    # ---- providers (see llm/factory.py). Model "" means: use the provider's default. ----
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "gemini").lower())
    embed_provider: str = field(default_factory=lambda: _env("EMBED_PROVIDER", "gemini").lower())
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL"))
    embed_dim: int = field(default_factory=lambda: int(_env("EMBED_DIM", "3072")))
    google_api_key: str = field(default_factory=lambda: _env("GOOGLE_API_KEY"), repr=False)
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"), repr=False)
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"), repr=False)
    # ---- storage ----
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL"), repr=False)  # empty -> SQLite file
    db_path: str = field(default_factory=lambda: _env("MEMORY_DB", "data/memory.db"))
    cache_path: str = field(default_factory=lambda: os.getenv("LLM_CACHE", "data/llm_cache.db"))  # "" disables
    tenant_id: str = field(default_factory=lambda: _env("TENANT_ID", "default"))
    # ---- service ----
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    worker_poll_seconds: float = field(default_factory=lambda: float(_env("WORKER_POLL_SECONDS", "1.0")))
    job_max_attempts: int = field(default_factory=lambda: int(_env("JOB_MAX_ATTEMPTS", "3")))
    job_visibility_timeout: int = field(default_factory=lambda: int(_env("JOB_VISIBILITY_TIMEOUT_SECONDS", "300")))
    # ---- memory behavior ----
    # Entity resolution thresholds (design 14B)
    er_match_threshold: float = 0.90
    er_review_threshold: float = 0.75
    # Embeddings alone confuse similar-looking names (InventoryService/InvoiceService: string ratio 0.50, true
    # variants >= 0.82), so embedding matches also require this string similarity on the distinctive name.
    er_ratio_guard: float = 0.75
    # Two candidates whose confidence differs by less than this go to human review (14C step 5)
    conflict_margin: float = 0.05
    # Experience promotion (design 19)
    experience_min_evidence: int = 3
    # Failure-class names: unrelated ones sit at 0.55-0.69 cosine, same-cause variants at 0.80-0.92 (measured).
    signature_similarity: float = 0.78
    weights: ConfidenceWeights = field(default_factory=ConfidenceWeights)
    # Component ablations for experiments (design 36): no_temporal | no_authority | no_graph | no_entity_resolution
    ablate: frozenset[str] = frozenset()


def get_settings() -> Settings:
    return Settings()
