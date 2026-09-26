"""Relation schema and cardinality (design 14A).

Contradiction detection only applies to single-valued relations. Unknown relations are not
silently accepted: they go to human review as `proposed` relations.

Note on `previously_used`-style relations: the temporal model makes them redundant. A superseded
fact is the same relation with `valid_until` set, so the ontology has no `previously_used`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelationSpec:
    name: str
    single_valued: bool
    object_is_entity: bool = True
    description: str = ""
    subject_type: str = "service"
    object_type: str | None = None  # entity type of the object; None for free-text objects
    # High-impact facts cannot be established by agent inference alone (design 23)
    high_impact: bool = False


RELATIONS: dict[str, RelationSpec] = {
    r.name: r
    for r in [
        RelationSpec("payment_provider", True, description="External payment provider a service uses", object_type="provider", high_impact=True),
        RelationSpec("primary_database", True, description="Primary datastore of a service", object_type="database", high_impact=True),
        RelationSpec("owned_by", True, description="Team that owns a service", object_type="team"),
        RelationSpec("language", True, description="Main implementation language of a service", object_type="language"),
        RelationSpec("depends_on", False, description="A service depends on another service or system", object_type="system"),
        RelationSpec("exposes_api", False, description="A service exposes an API", object_type="api"),
        RelationSpec("has_incident", False, description="An incident affected a service", object_type="incident"),
        RelationSpec("migration_reason", False, object_is_entity=False, description="Free-text reason for a migration or decision"),
    ]
}


def get_relation(name: str) -> RelationSpec | None:
    return RELATIONS.get(name.strip().lower())


def ontology_prompt() -> str:
    lines = []
    for r in RELATIONS.values():
        kind = "single-valued" if r.single_valued else "multi-valued"
        obj = "entity" if r.object_is_entity else "free text"
        lines.append(f"- {r.name} ({kind}, object is {obj}): {r.description}")
    return "\n".join(lines)
