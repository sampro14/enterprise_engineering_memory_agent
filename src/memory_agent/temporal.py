"""Temporal queries over bitemporal memories (design 15).

- current:        valid now, and the system still believes it
- as_of(T):       true in the real world on date T (valid time)
- as_known_on(D): what the system believed on date D about date T (system time + valid time)
"""
from __future__ import annotations

from .context import Context
from .models import Memory

FAR_FUTURE = "9999-12-31"
_BASE = "subject_id=? AND relation=?"


def current(ctx: Context, subject_id: str, relation: str) -> list[Memory]:
    now = ctx.now()
    return ctx.store.memories(
        f"{_BASE} AND status='active' AND recorded_until IS NULL "
        "AND (valid_until IS NULL OR valid_until > ?) AND valid_from <= ?",
        (subject_id, relation, now, now),
    )


def as_of(ctx: Context, subject_id: str, relation: str, date: str) -> list[Memory]:
    return ctx.store.memories(
        f"{_BASE} AND status='active' AND recorded_until IS NULL "
        "AND valid_from <= ? AND (valid_until IS NULL OR ? < valid_until)",
        (subject_id, relation, date, date),
    )


def as_known_on(ctx: Context, subject_id: str, relation: str, known_date: str, valid_date: str | None = None) -> list[Memory]:
    t = valid_date or known_date
    return ctx.store.memories(
        f"{_BASE} AND status IN ('active','superseded') AND observed_at <= ? "
        "AND (recorded_until IS NULL OR recorded_until > ?) "
        "AND valid_from <= ? AND (valid_until IS NULL OR ? < valid_until)",
        (subject_id, relation, known_date, known_date, t, t),
    )


def history(ctx: Context, subject_id: str, relation: str) -> list[Memory]:
    ms = ctx.store.memories(f"{_BASE} AND status='active' AND recorded_until IS NULL", (subject_id, relation))
    return sorted(ms, key=lambda m: (m.valid_from or "", m.valid_until or FAR_FUTURE))


def open_conflicts(ctx: Context, subject_id: str, relation: str) -> list[Memory]:
    return ctx.store.memories(f"{_BASE} AND status='conflicted'", (subject_id, relation))
