"""Resolved candidate memory passed between validation and consolidation."""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import Entity, ExtractedFact, Memory
from .ontology import RelationSpec


@dataclass
class Candidate:
    fact: ExtractedFact
    spec: RelationSpec | None
    subject_name: str
    subject_type: str
    subject: Entity | None  # None => new entity to create on acceptance
    object_name: str
    object_type: str | None
    object_entity: Entity | None
    object_value: str | None  # for literal relations
    valid_from: str
    valid_until: str | None
    source_type: str
    source_uri: str
    authority: float
    observed_at: str
    source_text: str
    subject_resolution_note: str = ""
    extras: dict = field(default_factory=dict)


@dataclass
class Validation:
    decision: str
    reason: str = ""
    existing: list[Memory] = field(default_factory=list)  # for UPDATE_EXISTING / conflicts
    supersede: list[Memory] = field(default_factory=list)  # existing memories the candidate replaces
    clip_until: str | None = None  # candidate.valid_until override (older fact clipped by newer one)
    confidence: float = 0.5
