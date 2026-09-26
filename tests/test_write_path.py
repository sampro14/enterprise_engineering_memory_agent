from conftest import fact
from memory_agent import temporal
from memory_agent.entities import normalize_key
from memory_agent.ingest import ingest_text
from memory_agent.models import Decision, SourceType


def svc(ctx, name="PaymentService"):
    return ctx.store.entity_by_alias_key(normalize_key(name))


def provider_names(ctx, ms):
    return sorted(ctx.store.get_entity(m.object_id).name for m in ms)


def test_migration_current_historical_and_as_of(ctx, clock, script):
    script["DOC-JAN"] = [fact("PaymentService", "payment_provider", "Stripe",
                              "PaymentService uses Stripe. DOC-JAN", valid_from="2025-01-01")]
    clock.d = "2025-01-15"
    ingest_text(ctx, "PaymentService uses Stripe. DOC-JAN", "jan.md", SourceType.OFFICIAL_DOCS, "2025-01-01")

    script["DOC-SEP"] = [fact("PaymentService", "payment_provider", "Razorpay",
                              "Migration to Razorpay completed on 2026-09-11. DOC-SEP", valid_from="2026-09-11")]
    clock.d = "2026-10-01"
    r = ingest_text(ctx, "Migration to Razorpay completed on 2026-09-11. DOC-SEP", "sep.md", SourceType.OFFICIAL_DOCS, "2026-09-11")
    assert r.outcomes[0].decision == Decision.ACCEPT

    s = svc(ctx)
    assert provider_names(ctx, temporal.current(ctx, s.id, "payment_provider")) == ["Razorpay"]
    assert provider_names(ctx, temporal.as_of(ctx, s.id, "payment_provider", "2026-03-01")) == ["Stripe"]
    assert provider_names(ctx, temporal.as_of(ctx, s.id, "payment_provider", "2026-09-20")) == ["Razorpay"]
    hist = temporal.history(ctx, s.id, "payment_provider")
    assert [(ctx.store.get_entity(m.object_id).name, m.valid_until) for m in hist] == [
        ("Stripe", "2026-09-11"), ("Razorpay", None)]
    # As known on 2026-06-01 the system still believed Stripe was current, and had no end date for it
    known = temporal.as_known_on(ctx, s.id, "payment_provider", "2026-06-01")
    assert provider_names(ctx, known) == ["Stripe"] and known[0].valid_until is None


def test_multi_valued_relation_does_not_conflict(ctx, script):
    script["DOC"] = [fact("PaymentService", "depends_on", "PostgreSQL", "PaymentService depends on PostgreSQL. DOC"),
                     fact("PaymentService", "depends_on", "Redis", "PaymentService depends on Redis. DOC")]
    r = ingest_text(ctx, "PaymentService depends on PostgreSQL. PaymentService depends on Redis. DOC", "d.md")
    assert [o.decision for o in r.outcomes] == [Decision.ACCEPT, Decision.ACCEPT]
    assert len(temporal.current(ctx, svc(ctx).id, "depends_on")) == 2


def test_duplicate_adds_evidence_and_raises_confidence(ctx, script, clock):
    script["A-DOC"] = [fact("PaymentService", "primary_database", "Oracle", "PaymentService primary database is Oracle. A-DOC")]
    script["B-DOC"] = [fact("PaymentService", "primary_database", "Oracle", "PaymentService primary database is Oracle. B-DOC")]
    ingest_text(ctx, "PaymentService primary database is Oracle. A-DOC", "a.md", SourceType.OFFICIAL_DOCS)
    m1 = temporal.current(ctx, svc(ctx).id, "primary_database")[0]
    r = ingest_text(ctx, "PaymentService primary database is Oracle. B-DOC", "b.md", SourceType.ARCHITECTURE_REPO)
    assert r.outcomes[0].decision == Decision.UPDATE_EXISTING
    m2 = temporal.current(ctx, svc(ctx).id, "primary_database")
    assert len(m2) == 1 and m2[0].confidence > m1.confidence
    assert len(ctx.store.evidence_of(m2[0].id)) == 2


def test_idempotent_ingest(ctx, script):
    script["DOC"] = [fact("PaymentService", "owned_by", "PaymentsTeam", "PaymentService is owned by PaymentsTeam. DOC")]
    txt = "PaymentService is owned by PaymentsTeam. DOC"
    ingest_text(ctx, txt, "d.md")
    assert ingest_text(ctx, txt, "d.md").skipped_duplicate
    assert len(ctx.store.memories()) == 1


def test_poisoning_low_authority_contradiction_is_flagged_not_applied(ctx, script, clock):
    script["CFG"] = [fact("PaymentService", "payment_provider", "Razorpay", "provider: Razorpay CFG", valid_from="2026-09-11")]
    ingest_text(ctx, "provider: Razorpay CFG", "deploy.yaml", SourceType.PRODUCTION_CONFIG, "2026-09-11")
    script["CHAT"] = [fact("PaymentService", "payment_provider", "Stripe", "we use Stripe for payments CHAT")]
    clock.d = "2026-10-05"
    r = ingest_text(ctx, "we use Stripe for payments CHAT", "chat", SourceType.DEVELOPER, "2026-10-05")
    assert r.outcomes[0].decision == Decision.MARK_CONFLICT
    s = svc(ctx)
    assert provider_names(ctx, temporal.current(ctx, s.id, "payment_provider")) == ["Razorpay"]
    assert len(temporal.open_conflicts(ctx, s.id, "payment_provider")) == 1
    assert len(ctx.store.pending_reviews()) == 1


def test_older_document_after_newer_is_clipped_as_history(ctx, script, clock):
    clock.d = "2026-10-01"
    script["NEW"] = [fact("PaymentService", "payment_provider", "Razorpay", "uses Razorpay NEW", valid_from="2026-09-11")]
    ingest_text(ctx, "uses Razorpay NEW", "new.md", SourceType.OFFICIAL_DOCS, "2026-09-11")
    script["OLD"] = [fact("PaymentService", "payment_provider", "Stripe", "uses Stripe OLD", valid_from="2025-01-01")]
    r = ingest_text(ctx, "uses Stripe OLD", "old.md", SourceType.OFFICIAL_DOCS, "2025-01-01")
    assert r.outcomes[0].decision == Decision.ACCEPT
    s = svc(ctx)
    assert provider_names(ctx, temporal.current(ctx, s.id, "payment_provider")) == ["Razorpay"]
    assert provider_names(ctx, temporal.as_of(ctx, s.id, "payment_provider", "2026-01-01")) == ["Stripe"]


def test_guardrails(ctx, script):
    script["UNK"] = [fact("PaymentService", "sla_tier", "gold", "sla tier is gold UNK")]
    script["SEC"] = [fact("PaymentService", "depends_on", "Vault", "depends_on Vault password=hunter2 SEC")]
    script["HAL"] = [fact("PaymentService", "primary_database", "MongoDB", "totally made up sentence HAL")]
    script["INF"] = [fact("PaymentService", "primary_database", "Oracle", "primary database is Oracle INF")]
    assert ingest_text(ctx, "sla tier is gold UNK", "u").outcomes[0].decision == Decision.REQUIRE_HUMAN_REVIEW
    assert ingest_text(ctx, "depends_on Vault password=hunter2 SEC", "s").outcomes[0].decision == Decision.REJECT
    assert "unsupported" in ingest_text(ctx, "unrelated HAL text", "h").outcomes[0].reason
    r = ingest_text(ctx, "primary database is Oracle INF", "i", SourceType.AGENT_INFERENCE)
    assert r.outcomes[0].decision == Decision.REQUIRE_HUMAN_REVIEW
    assert len(ctx.store.pending_reviews()) == 2


def test_alias_resolution_merges_variants(ctx, script):
    script["ONE"] = [fact("PaymentService", "owned_by", "PaymentsTeam", "PaymentService owned by PaymentsTeam ONE")]
    script["TWO"] = [fact("payments-svc", "depends_on", "Redis", "payments-svc depends on Redis TWO")]
    ingest_text(ctx, "PaymentService owned by PaymentsTeam ONE", "1")
    ingest_text(ctx, "payments-svc depends on Redis TWO", "2")
    assert len(ctx.store.list_entities("service")) == 1
    assert "payments-svc" in ctx.store.aliases_of(svc(ctx).id)


def test_ended_fact_without_start_matches_existing_history(ctx, script, clock):
    script["JAN"] = [fact("PaymentService", "payment_provider", "Stripe", "uses Stripe JAN", valid_from="2025-01-01")]
    ingest_text(ctx, "uses Stripe JAN", "jan.md", SourceType.OFFICIAL_DOCS, "2025-01-01")
    script["ADR"] = [fact("PaymentService", "payment_provider", "Razorpay", "moved to Razorpay ADR", valid_from="2026-09-11"),
                     fact("PaymentService", "payment_provider", "Stripe", "moved from Stripe ADR", valid_until="2026-09-11")]
    clock.d = "2026-09-15"
    r = ingest_text(ctx, "moved to Razorpay moved from Stripe ADR", "adr.md", SourceType.ARCHITECTURE_REPO, "2026-09-15")
    assert [o.decision for o in r.outcomes] == [Decision.UPDATE_EXISTING, Decision.ACCEPT]
    hist = temporal.history(ctx, svc(ctx).id, "payment_provider")
    assert [(ctx.store.get_entity(m.object_id).name, m.valid_from, m.valid_until) for m in hist] == [
        ("Stripe", "2025-01-01", "2026-09-11"), ("Razorpay", "2026-09-11", None)]


def test_ended_fact_without_any_other_knowledge_keeps_unknown_start(ctx, script, clock):
    from memory_agent.models import UNKNOWN_START
    script["ADR"] = [fact("PaymentService", "payment_provider", "Stripe", "moved from Stripe ADR", valid_until="2026-09-11")]
    clock.d = "2026-09-15"
    ingest_text(ctx, "moved from Stripe ADR", "adr.md", SourceType.ARCHITECTURE_REPO, "2026-09-15")
    m = temporal.history(ctx, svc(ctx).id, "payment_provider")[0]
    assert m.valid_from == UNKNOWN_START and m.valid_until == "2026-09-11"


def test_distinct_services_sharing_a_suffix_are_not_ambiguous(ctx, script):
    """Regression: 'BillingService' vs 'SubscriptionService' must not be flagged as the same/ambiguous."""
    script["ONE"] = [fact("SubscriptionService", "owned_by", "CoreTeam", "SubscriptionService owned ONE")]
    script["TWO"] = [fact("BillingService", "owned_by", "MobileTeam", "BillingService owned TWO")]
    script["THREE"] = [fact("the Billing service", "language", "Go", "the Billing service is written in Go THREE")]
    ingest_text(ctx, "SubscriptionService owned ONE", "1")
    r = ingest_text(ctx, "BillingService owned TWO", "2")
    assert r.outcomes[0].decision == Decision.ACCEPT
    assert len(ctx.store.list_entities("service")) == 2
    ingest_text(ctx, "the Billing service is written in Go THREE", "3")
    assert len(ctx.store.list_entities("service")) == 2  # 'the Billing service' resolved to BillingService
    assert ctx.store.pending_reviews() == []


def test_similar_looking_but_distinct_names_stay_distinct(ctx, script):
    for i, (svc_name, team) in enumerate([("InventoryService", "Growth team"), ("InvoiceService", "Data team")]):
        script[f"K{i}"] = [fact(svc_name, "owned_by", team, f"{svc_name} owned by {team} K{i}")]
        ingest_text(ctx, f"{svc_name} owned by {team} K{i}", f"k{i}")
    assert len(ctx.store.list_entities("service")) == 2 and len(ctx.store.list_entities("team")) == 2
    assert ctx.store.pending_reviews() == []
