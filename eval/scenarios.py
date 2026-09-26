"""Seeded synthetic 'world' generator for the temporal / contradiction benchmark (design 33-34).

All scenarios of one seed share a single world (one memory store), so systems must find the right
entity among many. Ground truth is known by construction. Every system sees the same documents in
the same order with the same dates and source types.
"""
from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field

from memory_agent.models import SourceType

PREFIXES = ["Billing", "Invoice", "Catalog", "Search", "Auth", "Notification", "Shipping", "Inventory", "Ledger",
            "Fraud", "Pricing", "Checkout", "Refund", "Loyalty", "Analytics", "Reporting", "Media", "Chat", "Email",
            "Recommendation", "Tax", "Subscription", "Identity", "Onboarding", "Support", "Scheduler", "Gateway",
            "Audit", "Export", "Sync", "Cart", "Wishlist", "Review", "Coupon", "Returns", "Warehouse", "Routing",
            "Tracking", "Payment", "Orders"]
PROVIDERS = ["Stripe", "Razorpay", "Adyen", "PayPal", "Braintree", "Square", "Worldpay", "Mollie"]
DATABASES = ["PostgreSQL", "Oracle", "MySQL", "MongoDB", "DynamoDB", "CockroachDB", "Cassandra", "SQLServer"]
TEAMS = ["Payments", "Platform", "Growth", "Core", "Infra", "Data", "Commerce", "Trust", "Mobile", "Search"]
LANGS = ["Java", "Go", "Python", "Kotlin", "Rust", "TypeScript", "Scala", "Ruby"]
DEPS = ["Redis", "Kafka", "RabbitMQ", "Elasticsearch", "S3", "Vault", "Consul", "Memcached", "ClickHouse"]

REL = {
    "payment_provider": dict(pool=PROVIDERS, phrase="payment provider"),
    "primary_database": dict(pool=DATABASES, phrase="primary database"),
    "owned_by": dict(pool=TEAMS, phrase="owning team"),
    "language": dict(pool=LANGS, phrase="implementation language"),
}
NOISE = [
    "Reminder: sprint review is on Friday. I am testing the {s} staging flow today, nothing to record yet.",
    "The weekly on-call rotation for {s} changes every Monday. Lunch is at noon.",
    "Quick note: I'm debugging a flaky unit test in {s} this afternoon.",
]


@dataclass
class Doc:
    ingest_date: str
    doc_date: str
    source_type: SourceType
    uri: str
    text: str
    scenario: str = ""


@dataclass
class Question:
    id: str
    scenario: str
    category: str  # change | alias | multi | poison | out_of_order | transient
    kind: str  # current | as_of | before | when | multi
    text: str
    expected: list[str]
    forbidden: list[str] = field(default_factory=list)


@dataclass
class World:
    seed: int
    docs: list[Doc]
    questions: list[Question]
    asked_at: str


def _d(x: dt.date) -> str:
    return x.isoformat()


def _v1(rel: str, s: str, v: str) -> str:
    return {
        "payment_provider": f"{s} uses {v} as its payment provider.",
        "primary_database": f"The primary database of {s} is {v}.",
        "owned_by": f"{s} is owned by the {v} team.",
        "language": f"{s} is implemented in {v}.",
    }[rel]


def _change(rel: str, s: str, old: str, new: str, when: dt.date) -> str:
    return {
        "payment_provider": f"Migration update: {s} moved from {old} to {new}. The cutover completed on {_d(when)}.",
        "primary_database": f"The primary database of {s} was migrated from {old} to {new} on {_d(when)}.",
        "owned_by": f"Ownership of {s} transferred from the {old} team to the {new} team, effective {_d(when)}.",
        "language": f"{s} was rewritten from {old} to {new}; the rewrite went live on {_d(when)}.",
    }[rel]


def _confirm(rel: str, s: str, v: str) -> str:
    return {
        "payment_provider": f"Architecture review: the current payment provider for {s} is {v}.",
        "primary_database": f"Architecture review: {s} currently runs on {v} as its primary database.",
        "owned_by": f"Architecture review: {s} is currently owned by the {v} team.",
        "language": f"Architecture review: {s} is currently written in {v}.",
    }[rel]


def _aliases(name: str) -> list[str]:
    base = name.removesuffix("Service")
    return [name, f"{base.lower()}-svc", f"the {base} service"]


def generate(seed: int, n_change: int = 6, n_alias: int = 3, n_multi: int = 3, n_poison: int = 4,
             n_ooo: int = 2, n_transient: int = 2) -> World:
    rng = random.Random(seed)
    names = [p + "Service" for p in rng.sample(PREFIXES, len(PREFIXES))]
    docs: list[Doc] = []
    qs: list[Question] = []
    day0 = dt.date(2025, 1, 10) + dt.timedelta(days=rng.randint(0, 60))
    last = day0

    def add_doc(when: dt.date, st: SourceType, uri: str, text: str, scen: str, doc_date: dt.date | None = None):
        nonlocal last
        docs.append(Doc(_d(when), _d(doc_date or when), st, uri, text, scen))
        last = max(last, when)

    def noise(scen: str, s: str, when: dt.date):
        add_doc(when, SourceType.DEVELOPER, f"chat/{scen}-noise", rng.choice(NOISE).format(s=s), scen)

    def versions(rel: str, n: int):
        vals = rng.sample(REL[rel]["pool"], n)
        starts = [day0 + dt.timedelta(days=rng.randint(0, 30))]
        for _ in range(n - 1):
            starts.append(starts[-1] + dt.timedelta(days=rng.randint(120, 240)))
        return vals, starts

    def change_scenario(idx: int, category: str, use_alias: bool):
        scen = f"{category}{idx}"
        s = names.pop()
        rel = rng.choice(list(REL))
        n = rng.choice([2, 2, 3])
        vals, starts = versions(rel, n)
        alias_pool = _aliases(s)
        mention = (lambda: rng.choice(alias_pool)) if use_alias else (lambda: s)
        add_doc(starts[0], SourceType.OFFICIAL_DOCS, f"docs/{scen}-v1.md", _v1(rel, mention(), vals[0]), scen)
        for i in range(1, n):
            ann = starts[i] + dt.timedelta(days=rng.randint(1, 5))
            st = rng.choice([SourceType.ARCHITECTURE_REPO, SourceType.OFFICIAL_DOCS])
            add_doc(ann, st, f"adr/{scen}-v{i + 1}.md", _change(rel, mention(), vals[i - 1], vals[i], starts[i]), scen)
        conf = starts[-1] + dt.timedelta(days=rng.randint(20, 60))
        if rng.random() < 0.5:
            add_doc(conf, SourceType.PRODUCTION_CONFIG, f"config/{scen}.yaml", _confirm(rel, mention(), vals[-1]), scen)
        noise(scen, s, starts[0] + dt.timedelta(days=rng.randint(2, 20)))
        q_alias = rng.choice(alias_pool) if use_alias else s
        ph = REL[rel]["phrase"]
        cat = category
        qs.append(Question(f"{scen}-cur", scen, cat, "current", f"Which {ph} does {q_alias} currently have?",
                           [vals[-1]], vals[:-1]))
        i = rng.randrange(n - 1) if n > 2 else 0  # a past version
        mid = starts[i] + (starts[i + 1] - starts[i]) // 2
        qs.append(Question(f"{scen}-asof", scen, cat, "as_of", f"What was the {ph} of {q_alias} on {_d(mid)}?",
                           [vals[i]], [v for j, v in enumerate(vals) if j != i]))
        qs.append(Question(f"{scen}-before", scen, cat, "before",
                           f"What was the {ph} of {q_alias} before it changed to {vals[-1]}?", [vals[-2]], [vals[-1]]))
        qs.append(Question(f"{scen}-when", scen, cat, "when",
                           f"On what date did {q_alias} change its {ph} from {vals[-2]} to {vals[-1]}? Answer as YYYY-MM-DD.",
                           [_d(starts[-1])]))

    for i in range(n_change):
        change_scenario(i, "change", False)
    for i in range(n_alias):
        change_scenario(i, "alias", True)

    for i in range(n_multi):
        scen, s = f"multi{i}", names.pop()
        deps = rng.sample(DEPS, 4)
        t1 = day0 + dt.timedelta(days=rng.randint(0, 40))
        t2 = t1 + dt.timedelta(days=rng.randint(60, 150))
        add_doc(t1, SourceType.OFFICIAL_DOCS, f"docs/{scen}-a.md", f"{s} depends on {deps[0]} and {deps[1]}.", scen)
        add_doc(t2, SourceType.ARCHITECTURE_REPO, f"adr/{scen}-b.md",
                f"{s} now also depends on {deps[2]} and {deps[3]}.", scen)
        qs.append(Question(f"{scen}-q", scen, "multi", "multi", f"List everything {s} depends on.", deps))

    for i in range(n_poison):
        scen, s = f"poison{i}", names.pop()
        rel = rng.choice(["payment_provider", "primary_database"])
        v_old, v_new, v_wrong = rng.sample(REL[rel]["pool"], 3)
        t1 = day0 + dt.timedelta(days=rng.randint(0, 40))
        t2 = t1 + dt.timedelta(days=rng.randint(100, 200))
        t3 = t2 + dt.timedelta(days=rng.randint(20, 60))
        ph = REL[rel]["phrase"]
        add_doc(t1, SourceType.OFFICIAL_DOCS, f"docs/{scen}.md", _v1(rel, s, v_old), scen)
        add_doc(t2, SourceType.PRODUCTION_CONFIG, f"config/{scen}.yaml", _confirm(rel, s, v_new), scen)
        stale = i % 2 == 0  # stale-memory trap vs plain wrong claim
        wrong = v_old if stale else v_wrong
        add_doc(t3, SourceType.DEVELOPER, f"chat/{scen}.txt",
                f"pretty sure the {ph} of {s} is {wrong}, I remember setting that up.", scen)
        qs.append(Question(f"{scen}-cur", scen, "poison", "current",
                           f"Which {ph} does {s} currently have?", [v_new], [wrong]))

    for i in range(n_ooo):
        scen, s = f"ooo{i}", names.pop()
        rel = "payment_provider" if i % 2 == 0 else "primary_database"
        v_old, v_new = rng.sample(REL[rel]["pool"], 2)
        t_old = day0 + dt.timedelta(days=rng.randint(0, 40))
        t_new = t_old + dt.timedelta(days=rng.randint(150, 250))
        ph = REL[rel]["phrase"]
        # The NEWER document is ingested first; the older one is discovered later (backfilled archive).
        add_doc(t_new + dt.timedelta(days=3), SourceType.ARCHITECTURE_REPO, f"adr/{scen}.md",
                _change(rel, s, v_old, v_new, t_new), scen, doc_date=t_new + dt.timedelta(days=3))
        add_doc(t_new + dt.timedelta(days=40), SourceType.OFFICIAL_DOCS, f"archive/{scen}.md", _v1(rel, s, v_old),
                scen, doc_date=t_old)
        qs.append(Question(f"{scen}-cur", scen, "out_of_order", "current",
                           f"Which {ph} does {s} currently have?", [v_new], [v_old]))
        qs.append(Question(f"{scen}-asof", scen, "out_of_order", "as_of",
                           f"What was the {ph} of {s} on {_d(t_old + dt.timedelta(days=30))}?", [v_old], [v_new]))

    for i in range(n_transient):
        scen, s = f"transient{i}", names.pop()
        rel = "primary_database" if i % 2 == 0 else "payment_provider"
        v_cur, v_plan = rng.sample(REL[rel]["pool"], 2)
        ph = REL[rel]["phrase"]
        t1 = day0 + dt.timedelta(days=rng.randint(0, 40))
        t2 = t1 + dt.timedelta(days=rng.randint(60, 120))
        add_doc(t1, SourceType.OFFICIAL_DOCS, f"docs/{scen}.md", _v1(rel, s, v_cur), scen)
        add_doc(t2, SourceType.DEVELOPER, f"chat/{scen}.txt",
                f"We are considering moving the {ph} of {s} to {v_plan} next year. Today I'm only trying {v_plan} in a "
                f"local sandbox; no decision has been made.", scen)
        qs.append(Question(f"{scen}-cur", scen, "transient", "current", f"Which {ph} does {s} currently have?",
                           [v_cur], [v_plan]))

    docs.sort(key=lambda d: (d.ingest_date, d.uri))
    asked_at = _d(last + dt.timedelta(days=30))
    return World(seed, docs, qs, asked_at)
