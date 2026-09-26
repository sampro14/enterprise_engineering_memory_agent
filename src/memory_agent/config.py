"""Environment-driven settings."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


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
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gemini-2.5-flash"))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "gemini-embedding-001"))
    db_path: str = field(default_factory=lambda: os.getenv("MEMORY_DB", "data/memory.db"))
    cache_path: str = field(default_factory=lambda: os.getenv("LLM_CACHE", "data/llm_cache.db"))
    tenant_id: str = field(default_factory=lambda: os.getenv("TENANT_ID", "default"))
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


def get_settings() -> Settings:
    return Settings()
