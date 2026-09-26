"""Core data models (design 25, PoC subset)."""
from __future__ import annotations

import uuid
from enum import StrEnum

from pydantic import BaseModel, Field

# valid_from for facts whose start is unknown (e.g. "used Stripe until 2026-09-11")
UNKNOWN_START = "0001-01-01"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class SourceType(StrEnum):
    """Source-authority tiers, highest first (design 21)."""

    PRODUCTION_CONFIG = "production_config"
    ARCHITECTURE_REPO = "architecture_repo"
    OFFICIAL_DOCS = "official_docs"
    INCIDENT_REPORT = "incident_report"
    DEVELOPER = "developer"
    AGENT_INFERENCE = "agent_inference"


AUTHORITY: dict[SourceType, float] = {
    SourceType.PRODUCTION_CONFIG: 1.0,
    SourceType.ARCHITECTURE_REPO: 0.85,
    SourceType.OFFICIAL_DOCS: 0.70,
    SourceType.INCIDENT_REPORT: 0.60,
    SourceType.DEVELOPER: 0.50,
    SourceType.AGENT_INFERENCE: 0.20,
}


class Decision(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    UPDATE_EXISTING = "UPDATE_EXISTING"
    MARK_CONFLICT = "MARK_CONFLICT"
    REQUIRE_HUMAN_REVIEW = "REQUIRE_HUMAN_REVIEW"


class MemoryStatus(StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    CONFLICTED = "conflicted"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"
    ARCHIVED = "archived"


class Entity(BaseModel):
    id: str
    tenant_id: str
    entity_type: str
    name: str
    description: str | None = None


class Memory(BaseModel):
    id: str
    tenant_id: str
    memory_type: str = "technical_fact"
    subject_id: str
    relation: str
    object_id: str | None = None
    object_value: str | None = None
    content: str = ""
    confidence: float = 0.5
    importance: float = 0.5
    source_authority: float = 0.5
    valid_from: str | None = None
    valid_until: str | None = None
    observed_at: str
    created_at: str
    recorded_until: str | None = None
    supersedes_id: str | None = None
    last_verified: str | None = None
    last_accessed: str | None = None
    status: str = MemoryStatus.ACTIVE
    access_policy: str | None = None


# ---- LLM extraction schema ------------------------------------------------------------


class ExtractedFact(BaseModel):
    """One fact as the extractor sees it, before entity resolution and validation."""

    subject: str = Field(description="Name of the entity the fact is about, as written in the text")
    subject_type: str = Field(description="service, database, team, provider, incident, ...")
    relation: str = Field(description="One relation name from the provided ontology")
    object: str = Field(description="Object entity name, or free text if the relation is literal")
    object_type: str | None = Field(default=None, description="Entity type of the object; null for literals")
    valid_from: str | None = Field(default=None, description="YYYY-MM-DD when this became true, only if stated in the text")
    valid_until: str | None = Field(default=None, description="YYYY-MM-DD when this stopped being true, only if stated")
    evidence: str = Field(description="Verbatim sentence or span from the text supporting the fact")


class ExtractionResult(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list)
