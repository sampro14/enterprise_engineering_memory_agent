"""Deployment simulator with hidden ORGANIZATION-SPECIFIC rules (design 32).

The rules are deliberately not guessable from general engineering knowledge, so avoiding a
mistake requires organizational experience. The simulator is deterministic.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

STEPS = ["run_tests", "contract_tests", "freeze_reconciliation", "unfreeze_reconciliation", "run_migration",
         "build_image", "deploy_staging", "smoke_tests", "warm_cache", "deploy_production", "notify_team"]

# rule id -> (failure class label for metrics, error text shown to the agent)
RULES = {
    "R1": ("ledger_migration_without_freeze",
           "ERR-4021: nightly reconciliation job corrupted by a concurrent schema change on ledger tables. Rolled back."),
    "R2": ("cold_cache_outage",
           "ERR-5310: p99 latency SLO breached after promote; edge cache was cold. Rolled back."),
    "R3": ("consumer_contract_broken",
           "ERR-3007: downstream consumers rejected the new public API response in staging. Deployment halted."),
}
ORDER = ["R3", "R1", "R2"]  # the order in which errors surface


@dataclass
class Task:
    id: str
    text: str
    ledger: bool
    edge: bool
    api: bool

    def applies(self, rule: str) -> bool:
        return {"R1": self.ledger, "R2": self.edge, "R3": self.api}[rule]


def _idx(plan: list[str], step: str) -> int | None:
    return plan.index(step) if step in plan else None


def violated_rules(t: Task, plan: list[str]) -> list[str]:
    v = []
    if t.api:
        c, s = _idx(plan, "contract_tests"), _idx(plan, "deploy_staging")
        if c is None or (s is not None and c > s):
            v.append("R3")
    if t.ledger:
        f, m = _idx(plan, "freeze_reconciliation"), _idx(plan, "run_migration")
        if m is not None and (f is None or f > m):
            v.append("R1")
    if t.edge:
        w, p = _idx(plan, "warm_cache"), _idx(plan, "deploy_production")
        if p is not None and (w is None or w > p):
            v.append("R2")
    return [r for r in ORDER if r in v]


def baseline_error(t: Task, plan: list[str]) -> str | None:
    for step in ("run_tests", "build_image", "deploy_staging", "deploy_production"):
        if step not in plan:
            return f"ERR-1000: pipeline incomplete, required step '{step}' missing."
    if t.ledger and "run_migration" not in plan:
        return "ERR-2001: schema mismatch, ledger migration was not run."
    if plan.index("deploy_staging") > plan.index("deploy_production"):
        return "ERR-1001: production deployed before staging."
    return None


def execute(t: Task, plan: list[str]) -> tuple[str, str | None, list[str]]:
    """Returns (outcome, error_text, all violated org rules)."""
    err = baseline_error(t, plan)
    v = violated_rules(t, plan)
    if err:
        return "failure", err, v
    if v:
        return "failure", RULES[v[0]][1], v
    return "success", None, []


def _desc(rng: random.Random, svc: str, ledger: bool, edge: bool, api: bool) -> str:
    feat = rng.choice(["adds a new reporting endpoint", "improves retry handling", "updates the pricing logic",
                       "refactors the notification templates", "upgrades a client library"])
    parts = [f"Release {svc} v{rng.randint(2, 9)}.{rng.randint(0, 20)}: {feat}."]
    parts.append(rng.choice(["This release alters the general-ledger tables.", "It changes the ledger schema (new column)."])
                 if ledger else rng.choice(["No database changes.", "There are no schema changes in this release."]))
    parts.append(rng.choice(["The service is served from the edge tier.", "It runs in the edge tier."])
                 if edge else rng.choice(["It runs in the internal tier.", "The service is internal-only (not edge)."]))
    parts.append(rng.choice(["The public API response format changes.", "It modifies the public REST API contract."])
                 if api else rng.choice(["No public API changes.", "The API contract is unchanged."]))
    return " ".join(parts)


def make_sequence(seed: int, n: int = 24) -> list[Task]:
    rng = random.Random(1000 + seed)
    pool = ["OrdersService", "BillingService", "SearchService", "AuthService", "CartService", "TaxService",
            "ShippingService", "LedgerService", "PricingService", "ReportingService", "CatalogService", "InvoiceService"]
    svcs = [rng.choice(pool) for _ in range(n)]
    tasks = []
    for i in range(n):
        ledger, edge, api = (rng.random() < 0.5 for _ in range(3))
        tasks.append(Task(f"task{i}", _desc(rng, svcs[i], ledger, edge, api), ledger, edge, api))
    return tasks
