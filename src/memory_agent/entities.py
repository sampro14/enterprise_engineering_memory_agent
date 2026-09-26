"""Entity resolution (design 14B): normalize -> alias -> fuzzy/embedding -> MATCH / NEW / REVIEW."""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum

from .context import Context
from .models import Entity


class ERDecision(StrEnum):
    MATCH = "MATCH"
    NEW_ENTITY = "NEW_ENTITY"
    REVIEW = "REVIEW"


@dataclass
class Resolution:
    decision: ERDecision
    entity: Entity | None = None
    score: float = 0.0
    note: str = ""


def normalize_key(name: str) -> str:
    """Lowercase alphanumerics only; 'svc' is expanded to 'service'."""
    tokens = re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower())
    tokens = ["service" if t == "svc" else t for t in tokens]
    return "".join(tokens)


GENERIC_TOKENS = {"the", "service", "svc", "team"}


def core_key(name: str) -> str:
    """Distinctive part of a name for fuzzy comparison: generic suffixes ('Service', 'svc') would otherwise
    make every 'XService' look similar to every 'YService'."""
    tokens = re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower())
    core = [t for t in tokens if t not in GENERIC_TOKENS]
    return "".join(core) or normalize_key(name)


def resolve(ctx: Context, name: str, entity_type: str, source: str = "extraction") -> Resolution:
    """Resolve a mention to an existing entity. Never creates entities; see `resolve_or_create`."""
    key = normalize_key(name)
    if not key:
        return Resolution(ERDecision.REVIEW, note="empty name")
    # 1-2. exact name / alias match (type-agnostic: the same key is the same thing)
    hit = ctx.store.entity_by_alias_key(key)
    if hit:
        return Resolution(ERDecision.MATCH, hit, 1.0, "alias")

    # 3. fuzzy string + embedding match among same-type entities
    same_type = {e.id: e for e in ctx.store.list_entities(entity_type)}
    if not same_type:
        return Resolution(ERDecision.NEW_ENTITY)
    core = core_key(name)
    best_ratio, best_ent = 0.0, None
    for e in same_type.values():
        for alias in [e.name, *ctx.store.aliases_of(e.id)]:
            r = SequenceMatcher(None, core, core_key(alias)).ratio()
            if r > best_ratio:
                best_ratio, best_ent = r, e
    if best_ratio >= 0.90 and best_ent:
        return Resolution(ERDecision.MATCH, best_ent, best_ratio, "fuzzy")

    q = ctx.embedder.embed([name])[0]
    top = ctx.store.search_vectors("entity", q, k=1, ref_ids=set(same_type))
    if top:
        ref_id, cos = top[0]
        ent = same_type[ref_id]
        ratio = SequenceMatcher(None, core, core_key(ent.name)).ratio()
        s = ctx.settings
        # The string-ratio guard stops semantically-similar-but-different entities
        # (Stripe vs Razorpay, InventoryService vs InvoiceService) from merging on embeddings alone.
        if ratio >= s.er_ratio_guard and cos >= s.er_match_threshold:
            return Resolution(ERDecision.MATCH, ent, cos, "embedding")
        if ratio >= s.er_ratio_guard and cos >= s.er_review_threshold:
            return Resolution(ERDecision.REVIEW, ent, cos, f"ambiguous with '{ent.name}'")
    return Resolution(ERDecision.NEW_ENTITY)


def create_entity(ctx: Context, name: str, entity_type: str, source: str = "extraction") -> Entity:
    now = ctx.now()
    e = ctx.store.add_entity(entity_type, name, now)
    ctx.store.add_alias(e.id, name, normalize_key(name), source, 1.0, now)
    ctx.store.put_vector("entity", e.id, ctx.embedder.embed([name])[0])
    return e


def register_alias(ctx: Context, entity: Entity, alias: str, confidence: float, source: str) -> None:
    ctx.store.add_alias(entity.id, alias, normalize_key(alias), source, confidence, ctx.now())
