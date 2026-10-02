"""Write path orchestration: extract -> resolve entities -> validate -> consolidate (design 8, 27)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .candidates import Candidate
from .consolidation import apply
from .context import Context
from .entities import ERDecision, register_alias, resolve
from .extraction import extract
from .models import AUTHORITY, UNKNOWN_START, Decision, ExtractedFact, SourceType
from .ontology import get_relation
from .validation import validate


@dataclass
class FactOutcome:
    subject: str
    relation: str
    object: str
    decision: str
    reason: str
    memory_id: str | None = None


@dataclass
class IngestReport:
    uri: str
    skipped_duplicate: bool = False
    outcomes: list[FactOutcome] = field(default_factory=list)

    def count(self, decision: str) -> int:
        return sum(o.decision == decision for o in self.outcomes)


def _resolve_entity(ctx: Context, name: str, etype: str):
    res = resolve(ctx, name, etype)
    if res.decision == ERDecision.MATCH:
        if res.entity and name != res.entity.name:
            register_alias(ctx, res.entity, name, res.score, "entity_resolution")
        return res.entity, res
    return None, res


def build_candidate(
    ctx: Context, fact: ExtractedFact, text: str, uri: str, source_type: SourceType, doc_date: str
) -> tuple[Candidate | None, str | None]:
    """Returns (candidate, review_reason). review_reason set when an entity mention is ambiguous."""
    spec = get_relation(fact.relation)
    s_type = spec.subject_type if spec else fact.subject_type.strip().lower()
    subj, s_res = _resolve_entity(ctx, fact.subject, s_type)
    if s_res.decision == ERDecision.REVIEW:
        return None, s_res.note
    obj, obj_value, o_type = None, None, None
    if spec is None or spec.object_is_entity:
        o_type = (spec.object_type if spec else fact.object_type) or "system"
        obj, o_res = _resolve_entity(ctx, fact.object, o_type.lower())
        if o_res.decision == ERDecision.REVIEW:
            return None, o_res.note
    else:
        obj_value = fact.object
    valid_from = fact.valid_from
    if valid_from is None:
        # A fact that already ended before the document date has an unknown start, not "today".
        ended_before_doc = fact.valid_until is not None and fact.valid_until <= doc_date
        valid_from = UNKNOWN_START if ended_before_doc else doc_date
    c = Candidate(
        fact=fact, spec=spec, subject_name=(subj.name if subj else fact.subject), subject_type=s_type, subject=subj,
        object_name=(obj.name if obj else fact.object), object_type=o_type, object_entity=obj, object_value=obj_value,
        valid_from=valid_from, valid_until=fact.valid_until,
        source_type=source_type.value, source_uri=uri, authority=AUTHORITY[source_type], observed_at=ctx.now(),
        source_text=text,
    )
    return c, None


def ingest_text(
    ctx: Context, text: str, uri: str, source_type: SourceType = SourceType.DEVELOPER, doc_date: str | None = None,
    unknown_relation: str = "review",
) -> IngestReport:
    """unknown_relation: "review" queues facts with relations outside the ontology for a human (default);
    "drop" discards them (used for machine-generated text such as task outcomes, which would flood the queue)."""
    report = IngestReport(uri=uri)
    h = hashlib.sha256(f"{source_type.value}|{text}".encode()).hexdigest()
    if ctx.store.source_seen(h):
        report.skipped_duplicate = True
        return report
    doc_date = doc_date or ctx.now()
    facts = extract(ctx, text, doc_date).facts
    # Process older facts first so a same-document change (old until X, new from X) is coherent.
    facts.sort(key=lambda f: (f.valid_from or f.valid_until or doc_date, f.valid_from is not None))
    for fact in facts:
        if unknown_relation == "drop" and get_relation(fact.relation) is None:
            report.outcomes.append(FactOutcome(fact.subject, fact.relation, fact.object, Decision.REJECT,
                                               "relation not in ontology (dropped)"))
            continue
        cand, review_reason = build_candidate(ctx, fact, text, uri, source_type, doc_date)
        if cand is None:
            ctx.store.add_review(f"ambiguous_entity: {review_reason}", fact.model_dump_json(), ctx.now())
            report.outcomes.append(FactOutcome(fact.subject, fact.relation, fact.object,
                                               Decision.REQUIRE_HUMAN_REVIEW, f"ambiguous_entity: {review_reason}"))
            continue
        v = validate(ctx, cand)
        mem_id = apply(ctx, cand, v)
        report.outcomes.append(FactOutcome(cand.subject_name, fact.relation, cand.object_name, v.decision, v.reason, mem_id))
    ctx.store.record_source(uri, h, source_type.value, doc_date, ctx.now())
    return report
