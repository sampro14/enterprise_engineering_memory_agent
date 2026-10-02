"""Hybrid retrieval (design 16-17): entity + graph traversal + vector, temporal filter, re-rank, conflict check."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .context import Context
from .entities import ERDecision, resolve
from .models import UNKNOWN_START, Memory
from .temporal import open_conflicts

MAX_HOPS = 2


def mentioned_entities(ctx: Context, text: str) -> list[str]:
    """Names of known entities that appear in `text` (by name or alias, case-insensitive, whole-token). A deterministic
    fallback so retrieval never depends on an LLM noticing a service name."""
    found: list[str] = []
    for e in ctx.store.list_entities():
        for alias in {e.name, *ctx.store.aliases_of(e.id)}:
            if len(alias) >= 3 and re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", text, re.IGNORECASE):
                found.append(e.name)
                break
    return found


@dataclass
class Retrieved:
    memory: Memory
    subject: str
    object: str
    state: str  # current | historical | future | valid_at
    score: float
    relevance: float
    proximity: float
    sources: list[str] = field(default_factory=list)


@dataclass
class Retrieval:
    items: list[Retrieved] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)


def _state(m: Memory, now: str, as_of: str | None) -> str:
    if as_of:
        ok = (m.valid_from or "") <= as_of and (m.valid_until is None or as_of < m.valid_until)
        return "valid_at" if ok else "historical"
    if (m.valid_from or "") > now:
        return "future"
    if m.valid_until is not None and m.valid_until <= now:
        return "historical"
    return "current"


def _graph_expand(ctx: Context, seeds: list[str]) -> dict[str, int]:
    """BFS over memories that link two entities. Returns {entity_id: hop_distance}."""
    dist = {e: 0 for e in seeds}
    frontier = set(seeds)
    for hop in range(1, MAX_HOPS + 1):
        if not frontier:
            break
        ph = ",".join("?" * len(frontier))
        rows = ctx.store.memories(
            f"status='active' AND recorded_until IS NULL AND object_id IS NOT NULL "
            f"AND (subject_id IN ({ph}) OR object_id IN ({ph}))",
            [*frontier, *frontier],
        )
        nxt: set[str] = set()
        for m in rows:
            for e in (m.subject_id, m.object_id):
                if e and e not in dist:
                    dist[e] = hop
                    nxt.add(e)
        frontier = nxt
    return dist


def retrieve(
    ctx: Context, question: str, entities: list[str], relations: list[str] | None = None,
    as_of: str | None = None, k: int = 14,
) -> Retrieval:
    store, now = ctx.store, ctx.now()
    out = Retrieval()

    # 1. entity resolution (MATCH only: retrieval must not guess)
    seeds: list[str] = []
    for name in entities:
        for etype in {e.entity_type for e in store.list_entities()} or {"service"}:
            res = resolve(ctx, name, etype)
            if res.decision == ERDecision.MATCH and res.entity and res.entity.id not in seeds:
                seeds.append(res.entity.id)
                break
    out.entity_ids = seeds
    dist = _graph_expand(ctx, seeds) if seeds else {}

    # 2. candidate memories: everything touching a reached entity, plus vector hits
    base = "status='active' AND recorded_until IS NULL"
    cands: dict[str, Memory] = {}
    if dist:
        ph = ",".join("?" * len(dist))
        for m in store.memories(f"{base} AND (subject_id IN ({ph}) OR object_id IN ({ph}))", [*dist, *dist]):
            cands[m.id] = m
    qv = ctx.embedder.embed([question])[0]
    sims = dict(store.search_vectors("memory", qv, k=40))
    valid_ids = {m.id for m in store.memories(base)}
    for mid, s in sims.items():
        if mid in valid_ids and mid not in cands and s > 0.3:
            m = store.get_memory(mid)
            if m:
                cands[mid] = m

    # 3. score and label
    rel_set = {r.lower() for r in (relations or [])}
    for m in cands.values():
        d = min(dist.get(m.subject_id, 9), dist.get(m.object_id or "", 9))
        proximity = {0: 1.0, 1: 0.7, 2: 0.4}.get(d, 0.1)
        relevance = sims.get(m.id, 0.0) + (0.25 if m.relation in rel_set else 0.0)
        state = _state(m, now, as_of)
        temporal_fit = {"current": 1.0, "valid_at": 1.0, "historical": 0.5, "future": 0.2}[state]
        if as_of and state == "historical":
            temporal_fit = 0.3
        freshness = 1.0 if m.last_verified else 0.7
        score = 0.35 * relevance + 0.20 * m.confidence + 0.20 * temporal_fit + 0.15 * proximity + 0.10 * freshness
        subj = store.get_entity(m.subject_id)
        obj = m.object_value or (store.get_entity(m.object_id).name if m.object_id else "")
        sources = sorted({f"{r['source_uri']} [{r['source_type']}]" for r in store.evidence_of(m.id)})
        out.items.append(Retrieved(m, subj.name if subj else "?", obj, state, score, relevance, proximity, sources))

    out.items.sort(key=lambda r: -r.score)
    # Directly matched entities' current/valid facts are always kept, even if low-ranked.
    keep = out.items[:k]
    kept_ids = {r.memory.id for r in keep}
    for r in out.items[k:]:
        if r.memory.subject_id in seeds and r.state in ("current", "valid_at") and r.memory.id not in kept_ids:
            keep.append(r)
    out.items = keep
    store.touch([r.memory.id for r in out.items], now)

    # 4. conflict check
    seen: set[tuple[str, str]] = set()
    for r in out.items:
        key = (r.memory.subject_id, r.memory.relation)
        if key in seen:
            continue
        seen.add(key)
        for cm in open_conflicts(ctx, *key):
            cobj = cm.object_value or (store.get_entity(cm.object_id).name if cm.object_id else "")
            src = ", ".join(sorted({f"{e['source_uri']} [{e['source_type']}]" for e in store.evidence_of(cm.id)}))
            out.conflicts.append(
                f"{r.subject} {cm.relation} = {cobj} (claimed {cm.valid_from}, unverified, from {src}) "
                f"contradicts the accepted fact"
            )
    return out


def _fmt_range(m: Memory) -> str:
    start = "unknown start" if m.valid_from == UNKNOWN_START else f"from {m.valid_from}"
    return f"{start}" + (f" until {m.valid_until}" if m.valid_until else "")


def build_context(ret: Retrieval, experiences: list[str] | None = None) -> str:
    def line(r: Retrieved) -> str:
        m = r.memory
        return (f"- {r.subject} {m.relation} = {r.object} ({_fmt_range(m)}; confidence {m.confidence:.2f}; "
                f"sources: {', '.join(r.sources) or 'n/a'})")

    groups = {"current": [], "valid_at": [], "historical": [], "future": []}
    for r in ret.items:
        groups[r.state].append(r)
    parts: list[str] = []
    titles = {"current": "CURRENT", "valid_at": "VALID AT THE REQUESTED DATE", "historical": "HISTORICAL",
              "future": "SCHEDULED / NOT YET IN EFFECT"}
    for key in ("current", "valid_at", "historical", "future"):
        if groups[key]:
            body = "\n".join(line(r) for r in sorted(groups[key], key=lambda x: x.memory.valid_from or ""))
            parts.append(f"{titles[key]}:\n{body}")
    if ret.conflicts:
        parts.append("UNRESOLVED CONFLICTS:\n" + "\n".join(f"- {c}" for c in ret.conflicts))
    if experiences:
        parts.append("EXPERIENCE (lessons from previous tasks):\n" + "\n".join(f"- {e}" for e in experiences))
    return "\n\n".join(parts)
