"""Human review queue (design 9.5 REQUIRE_HUMAN_REVIEW / MARK_CONFLICT)."""
from __future__ import annotations

import json

from .consolidation import _reindex
from .context import Context
from .ingest import ingest_text
from .models import MemoryStatus, SourceType
from .validation import overlaps


class ReviewError(Exception):
    pass


def _get(ctx: Context, review_id: str):
    rows = ctx.store.fetch("SELECT * FROM reviews WHERE id=? AND tenant_id=?", (review_id, ctx.store.tenant))
    if not rows:
        raise ReviewError(f"no such review {review_id}")
    if rows[0]["status"] != "pending":
        raise ReviewError(f"review {review_id} already {rows[0]['status']}")
    return rows[0]


def _close(ctx: Context, review_id: str, status: str, reviewer: str, note: str | None) -> None:
    ctx.store.update("reviews", review_id, {"status": status, "reviewer": reviewer, "note": note,
                                            "decided_at": ctx.now()})


def approve(ctx: Context, review_id: str, reviewer: str = "cli", note: str | None = None) -> str:
    """Approve a flagged memory. A conflicted memory becomes active and supersedes what it contradicted."""
    r = _get(ctx, review_id)
    now = ctx.now()
    if r["memory_id"]:
        m = ctx.store.get_memory(r["memory_id"])
        if m is None or m.status != MemoryStatus.CONFLICTED:
            raise ReviewError("memory is no longer conflicted")
        clash = ctx.store.memories(
            "subject_id=? AND relation=? AND status='active' AND recorded_until IS NULL AND id<>?",
            (m.subject_id, m.relation, m.id),
        )
        for old in clash:
            if old.object_id != m.object_id and overlaps(m.valid_from or "", m.valid_until, old.valid_from or "", old.valid_until):
                closed = ctx.store.close_validity(old, m.valid_from or now, now)
                _reindex(ctx, closed, None)
        # Human confirmation lifts authority to the developer-confirmation tier at minimum.
        ctx.store.update("memories", m.id, {"status": MemoryStatus.ACTIVE, "last_verified": now,
                                            "source_authority": max(m.source_authority, 0.5)})
        _close(ctx, review_id, "approved", reviewer, note)
        return m.id
    cand = json.loads(r["candidate_json"])
    if "fact" not in cand or "agent inference" not in (r["reason"] or ""):
        raise ReviewError("only conflicts and agent-inference candidates can be approved in the PoC; reject instead")
    text = cand["fact"]["evidence"]
    rep = ingest_text(ctx, text, f"review:{review_id}", SourceType.DEVELOPER, cand.get("valid_from"))
    _close(ctx, review_id, "approved", reviewer, note)
    return rep.outcomes[0].memory_id or ""


def reject(ctx: Context, review_id: str, reviewer: str = "cli", note: str | None = None) -> None:
    r = _get(ctx, review_id)
    if r["memory_id"]:
        ctx.store.update("memories", r["memory_id"], {"status": MemoryStatus.INVALIDATED})
    _close(ctx, review_id, "rejected", reviewer, note)
