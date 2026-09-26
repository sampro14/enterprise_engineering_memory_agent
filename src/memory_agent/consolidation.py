"""Apply validated candidates to the store (design 9.6): write memories, close old validity, index."""
from __future__ import annotations

import json

from .candidates import Candidate, Validation
from .confidence import VERIFIED_AUTHORITY, compute
from .context import Context
from .entities import create_entity
from .models import Decision, Entity, Memory, MemoryStatus, new_id


def _ensure_entities(ctx: Context, c: Candidate) -> tuple[Entity, Entity | None]:
    subj = c.subject or create_entity(ctx, c.subject_name, c.subject_type)
    obj = c.object_entity
    if obj is None and c.spec and c.spec.object_is_entity:
        obj = create_entity(ctx, c.object_name, c.object_type or "system")
    return subj, obj


def memory_text(subject: str, relation: str, obj: str, evidence: str) -> str:
    return f"{subject} {relation.replace('_', ' ')} {obj}. {evidence}"


def index_memory(ctx: Context, m: Memory, subject_name: str, obj_name: str, evidence: str) -> None:
    text = memory_text(subject_name, m.relation, obj_name, evidence)
    ctx.store.put_vector("memory", m.id, ctx.embedder.embed([text])[0])


def _candidate_json(c: Candidate) -> str:
    d = {"fact": c.fact.model_dump(), "valid_from": c.valid_from, "valid_until": c.valid_until,
         "source_type": c.source_type, "source_uri": c.source_uri, "authority": c.authority,
         "observed_at": c.observed_at, "subject_name": c.subject_name, "object_name": c.object_name,
         "note": c.subject_resolution_note}
    return json.dumps(d)


def apply(ctx: Context, c: Candidate, v: Validation) -> str | None:
    """Returns the id of the memory written or updated (None when nothing was stored)."""
    now = ctx.now()
    store = ctx.store

    if v.decision == Decision.REJECT:
        return None
    if v.decision == Decision.REQUIRE_HUMAN_REVIEW:
        store.add_review(v.reason, _candidate_json(c), now)
        return None

    if v.decision == Decision.UPDATE_EXISTING:
        ex = v.existing[0]
        seen = {r["source_uri"] for r in store.evidence_of(ex.id)}
        if c.source_uri not in seen:
            store.add_evidence(ex.id, c.source_type, c.source_uri, c.fact.evidence, c.authority, now)
        n = len({r["source_uri"] for r in store.evidence_of(ex.id)})
        auth = max(ex.source_authority, c.authority)
        conf = compute(ctx.settings.weights, auth, n, now if auth >= VERIFIED_AUTHORITY else ex.last_verified, now)
        store.update("memories", ex.id, {
            "confidence": max(conf, ex.confidence), "source_authority": auth,
            "last_verified": now if c.authority >= VERIFIED_AUTHORITY else ex.last_verified,
        })
        # The candidate may end a fact that is still open (e.g. "used Stripe until 2026-09-11")
        if c.valid_until and (ex.valid_until is None or c.valid_until < ex.valid_until):
            closed = store.close_validity(ex, c.valid_until, now)
            _reindex(ctx, closed, c)
            return closed.id
        return ex.id

    # ACCEPT or MARK_CONFLICT: write a new memory
    subj, obj = _ensure_entities(ctx, c)
    conflicted = v.decision == Decision.MARK_CONFLICT
    valid_until = v.clip_until or c.valid_until
    m = Memory(
        id=new_id("mem"), tenant_id=store.tenant,
        memory_type="decision" if c.spec and not c.spec.object_is_entity else "technical_fact",
        subject_id=subj.id, relation=c.fact.relation.strip().lower(),
        object_id=obj.id if obj else None, object_value=c.object_value,
        content=c.fact.evidence, confidence=v.confidence,
        importance=0.9 if c.spec and c.spec.high_impact else 0.6,
        source_authority=c.authority, valid_from=c.valid_from, valid_until=valid_until,
        observed_at=now, created_at=now,
        last_verified=now if c.authority >= VERIFIED_AUTHORITY else None,
        status=MemoryStatus.CONFLICTED if conflicted else MemoryStatus.ACTIVE,
    )
    store.insert_memory(m)
    store.add_evidence(m.id, c.source_type, c.source_uri, c.fact.evidence, c.authority, now)
    _reindex(ctx, m, c, subj, obj)

    for old in v.supersede:  # the newer fact ends the older one's validity
        closed = store.close_validity(old, c.valid_from, now)
        _reindex(ctx, closed, None)
    if conflicted:
        store.add_review(v.reason, _candidate_json(c), now, memory_id=m.id)
    return m.id


def _reindex(ctx: Context, m: Memory, c: Candidate | None, subj: Entity | None = None, obj: Entity | None = None) -> None:
    subj = subj or ctx.store.get_entity(m.subject_id)
    obj_name = m.object_value or (ctx.store.get_entity(m.object_id).name if m.object_id else "")
    ev = ctx.store.evidence_of(m.id)
    evidence = ev[0]["excerpt"] if ev else m.content
    index_memory(ctx, m, subj.name if subj else "", obj_name, evidence)
