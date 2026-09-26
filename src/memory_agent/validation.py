"""Memory validation (design 9.5, 14C, 23). Nothing becomes trusted memory without passing here."""
from __future__ import annotations

import re

from .candidates import Candidate, Validation
from .confidence import VERIFIED_AUTHORITY, compute
from .context import Context
from .models import Decision, Memory, SourceType

FAR_FUTURE = "9999-12-31"
SENSITIVE = re.compile(
    r"(?i)\b(password|passwd|secret|api[_-]?key|token)\b\s*[:=]\s*\S+|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)


def _norm_tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def grounded(evidence: str, source_text: str) -> bool:
    """Evidence must be (nearly) verbatim from the source. Guards against hallucinated facts."""
    ev, src = _norm_tokens(evidence), _norm_tokens(source_text)
    if not ev:
        return False
    if " ".join(ev) in " ".join(src):
        return True
    srcset = set(src)
    return sum(t in srcset for t in ev) / len(ev) >= 0.85


def overlaps(a_from: str, a_until: str | None, b_from: str, b_until: str | None) -> bool:
    return a_from < (b_until or FAR_FUTURE) and b_from < (a_until or FAR_FUTURE)


def _same_object(m: Memory, c: Candidate) -> bool:
    if c.spec and not c.spec.object_is_entity:
        return (m.object_value or "").strip().lower() == (c.object_value or "").strip().lower()
    return c.object_entity is not None and m.object_id == c.object_entity.id


def _active(ctx: Context, c: Candidate) -> list[Memory]:
    if c.subject is None:
        return []
    return ctx.store.memories(
        "subject_id=? AND relation=? AND status='active' AND recorded_until IS NULL",
        (c.subject.id, c.fact.relation.strip().lower()),
    )


def candidate_confidence(ctx: Context, c: Candidate, contradiction: float = 0.0) -> float:
    verified_at = c.observed_at if c.authority >= VERIFIED_AUTHORITY else None
    return compute(ctx.settings.weights, c.authority, 1, verified_at, c.observed_at, contradiction)


def validate(ctx: Context, c: Candidate) -> Validation:
    text = f"{c.fact.evidence} {c.object_name}"
    if SENSITIVE.search(text):
        return Validation(Decision.REJECT, "sensitive information")
    if c.spec is None:
        return Validation(Decision.REQUIRE_HUMAN_REVIEW, f"unknown relation '{c.fact.relation}' (proposed)")
    if not grounded(c.fact.evidence, c.source_text):
        return Validation(Decision.REJECT, "evidence not found in source text (unsupported)")
    if c.source_type == SourceType.AGENT_INFERENCE and c.spec.high_impact:
        return Validation(Decision.REQUIRE_HUMAN_REVIEW, "high-impact fact supported only by agent inference")

    if c.valid_until is not None and c.valid_until <= c.valid_from:
        return Validation(Decision.REJECT, f"invalid validity interval ({c.valid_from} .. {c.valid_until})")

    conf = candidate_confidence(ctx, c)
    active = _active(ctx, c)

    # Duplicate: same triple with overlapping validity -> add evidence instead of a new memory
    dups = [m for m in active if _same_object(m, c) and overlaps(c.valid_from, c.valid_until, m.valid_from or "", m.valid_until)]
    if dups:
        return Validation(Decision.UPDATE_EXISTING, "already known; adding evidence", existing=dups, confidence=conf)

    if not c.spec.single_valued:
        return Validation(Decision.ACCEPT, "multi-valued relation; no conflict possible", confidence=conf)

    conflicts = [
        m for m in active
        if not _same_object(m, c) and overlaps(c.valid_from, c.valid_until, m.valid_from or "", m.valid_until)
    ]
    if not conflicts:
        return Validation(Decision.ACCEPT, "no conflicting fact", confidence=conf)

    supersede: list[Memory] = []
    clip: str | None = None
    unresolved: list[str] = []
    for ex in conflicts:
        ex_from = ex.valid_from or ""
        if c.valid_from > ex_from:  # candidate is newer in valid time
            if c.authority >= VERIFIED_AUTHORITY or c.authority >= ex.source_authority:
                supersede.append(ex)
            else:
                unresolved.append(
                    f"newer but lower-authority claim ({c.source_type}) contradicts {ex.id} (authority {ex.source_authority})"
                )
        elif c.valid_from < ex_from:  # candidate describes an older period: clip it, keep as history
            clip = min(clip, ex_from) if clip else ex_from
        else:  # same start date: fall back to confidence comparison
            conf_c = candidate_confidence(ctx, c, contradiction=1.0)
            margin = ctx.settings.conflict_margin
            if conf_c > ex.confidence + margin:
                supersede.append(ex)
            else:
                unresolved.append(f"same-date conflict with {ex.id}; confidence {conf_c:.2f} vs {ex.confidence:.2f}")
    if unresolved:
        return Validation(
            Decision.MARK_CONFLICT, "; ".join(unresolved), existing=conflicts,
            confidence=candidate_confidence(ctx, c, contradiction=1.0),
        )
    reason = "supersedes " + ", ".join(m.id for m in supersede) if supersede else "older period, clipped"
    return Validation(Decision.ACCEPT, reason, existing=conflicts, supersede=supersede, clip_until=clip, confidence=conf)
