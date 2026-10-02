"""Request/response models for the HTTP API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..models import SourceType

DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"


class IngestRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500_000, description="Document text to learn from (treated as untrusted data)")
    uri: str = Field(min_length=1, max_length=1000, description="Where the text came from; recorded as evidence")
    source_type: SourceType = Field(default=SourceType.DEVELOPER, description="Authority tier of the source")
    doc_date: str | None = Field(default=None, pattern=DATE_PATTERN, description="Document date YYYY-MM-DD (default: today)")


class IngestAccepted(BaseModel):
    job_id: str
    status: str
    status_url: str


class JobView(BaseModel):
    id: str
    kind: str
    status: Literal["queued", "running", "succeeded", "failed"]
    attempts: int
    max_attempts: int
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


class FactOutcomeView(BaseModel):
    subject: str
    relation: str
    object: str
    decision: str
    reason: str
    memory_id: str | None = None


class IngestResult(BaseModel):
    skipped_duplicate: bool
    outcomes: list[FactOutcomeView]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=10, ge=1, le=50)
    as_of: str | None = Field(default=None, pattern=DATE_PATTERN)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class AskResponse(BaseModel):
    answer: str
    values: list[str]
    unknown: bool
    citations: list[str]
    conflicts: list[str]
    used_memory: bool
    context: str
    as_of: str | None = None


class MergeRequest(BaseModel):
    merged_entity_id: str = Field(description="Entity to merge INTO the entity in the path (same type only)")


class ReviewDecision(BaseModel):
    decision: Literal["approve", "reject"]
    note: str | None = Field(default=None, max_length=2000)


class TaskCreate(BaseModel):
    task: str = Field(min_length=1, max_length=8000)
    allowed_steps: list[str] = Field(min_length=1, max_length=100)
    with_facts: bool = True


class TaskOutcome(BaseModel):
    outcome: Literal["success", "failure"]
    error: str | None = Field(default=None, max_length=8000)


class ErrorBody(BaseModel):
    error: str
    detail: str
    request_id: str | None = None
