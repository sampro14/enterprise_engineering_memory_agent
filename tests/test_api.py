"""HTTP API tests: FastAPI TestClient + fake providers on each storage backend (Postgres variants need DATABASE_URL)."""
import re

import pytest
from fastapi.testclient import TestClient

from conftest import fact
from memory_agent import tenants
from memory_agent.answering import Answer, QueryAnalysis
from memory_agent.api.app import create_app
from memory_agent.config import Settings
from memory_agent.episodes import Reflection
from memory_agent.experience import LessonCheck, LessonDraft
from memory_agent.llm.base import LLMError, ProviderConfigError
from memory_agent.llm.fake import FakeEmbedder, FakeLLM
from memory_agent.models import ExtractionResult
from memory_agent.services.memory_service import MemoryService
from memory_agent.task_agent import Plan
from memory_agent.worker import Worker

STEPS = ["run_tests", "build_image", "deploy_staging", "deploy_production"]


class Env:
    def __init__(self, backend):
        self.script: dict[str, list] = {}
        self.fail_with: Exception | None = None
        self.settings = Settings(llm_provider="fake", embed_provider="fake", embed_dim=256, job_max_attempts=1)
        self.llm = FakeLLM(self.handler)
        self.service = MemoryService(self.settings, self.llm, FakeEmbedder(), backend)
        self.client = TestClient(create_app(self.service), raise_server_exceptions=False)
        self.worker = Worker(self.service, "test-worker")
        self.keys = {}

    def handler(self, prompt, schema):
        if self.fail_with:
            raise self.fail_with
        if schema is ExtractionResult:
            for marker, facts in self.script.items():
                if marker in prompt:
                    return ExtractionResult(facts=facts)
            return ExtractionResult()
        if schema is QueryAnalysis:
            return QueryAnalysis(needs_memory=True, entities=["PaymentService"], relations=["payment_provider"])
        if schema is Answer:
            m = re.search(r"CURRENT:\n- \S+ \w+ = (.+?) \(", prompt)
            return Answer(answer=f"It is {m.group(1)}" if m else "unknown", values=[m.group(1)] if m else [], unknown=not m)
        if schema is Plan:
            return Plan(steps=list(STEPS))
        if schema is Reflection:
            return Reflection(failure_class="db_error", conditions=["db"], summary="failed")
        if schema is LessonDraft:
            return LessonDraft(pattern="Deploying a service that uses db", lesson="db errors recur",
                               conditions=["db"], recommended_action="check db first")
        if schema is LessonCheck:
            return LessonCheck(consistent=True)
        raise AssertionError(schema)

    def tenant(self, name: str) -> dict:
        tid = tenants.create_tenant(self.service.store, name, tenant_id=f"ten_{name}")
        key = tenants.create_api_key(self.service.store, tid)
        self.keys[name] = key
        return {"Authorization": f"Bearer {key}"}

    def drain(self) -> int:
        n = 0
        while self.worker.run_once():
            n += 1
        return n


@pytest.fixture
def env(backend):
    return Env(backend)


@pytest.fixture
def auth(env):
    return env.tenant("acme")


def ingest(env, auth, text, uri="doc.md", source_type="official_docs", doc_date="2025-01-01"):
    r = env.client.post("/api/v1/ingest", json={"text": text, "uri": uri, "source_type": source_type, "doc_date": doc_date},
                        headers=auth)
    assert r.status_code == 202, r.text
    env.drain()
    return env.client.get(r.json()["status_url"], headers=auth).json()


# ---------------------------------------------------------------- system + auth
def test_health_and_ready_are_open_and_request_id_is_echoed(env):
    assert env.client.get("/healthz").json() == {"status": "ok"}
    r = env.client.get("/readyz", headers={"X-Request-ID": "abc123"})
    assert r.status_code == 200 and r.json()["status"] == "ready" and r.headers["X-Request-ID"] == "abc123"


def test_authentication_is_required_and_keys_can_be_revoked(env, auth):
    assert env.client.get("/api/v1/reviews").status_code == 401
    bad = env.client.get("/api/v1/reviews", headers={"Authorization": "Bearer mem_nope"})
    assert bad.status_code == 401 and bad.headers["WWW-Authenticate"] == "Bearer"
    assert env.client.get("/api/v1/reviews", headers=auth).status_code == 200
    tenants.revoke_api_key(env.service.store, env.keys["acme"])
    assert env.client.get("/api/v1/reviews", headers=auth).status_code == 401


def test_tenant_is_never_taken_from_the_request_body(env, auth):
    r = env.client.post("/api/v1/ingest", headers=auth, json={"text": "x", "uri": "u", "tenant_id": "ten_other"})
    assert r.status_code == 202
    assert env.client.get(f"/api/v1/jobs/{r.json()['job_id']}", headers=auth).json()["status"] == "queued"
    row = env.service.store.fetch("SELECT tenant_id FROM jobs")[0]
    assert row["tenant_id"] == "ten_acme"


# ---------------------------------------------------------------- async ingest -> query flow
def test_ingest_job_worker_then_query_end_to_end(env, auth):
    env.script["JAN"] = [fact("PaymentService", "payment_provider", "Stripe", "uses Stripe JAN", valid_from="2025-01-01")]
    r = env.client.post("/api/v1/ingest", headers=auth, json={
        "text": "PaymentService uses Stripe JAN", "uri": "arch.md", "source_type": "official_docs", "doc_date": "2025-01-01"})
    assert r.status_code == 202 and r.json()["status"] == "queued"
    jid = r.json()["job_id"]
    assert env.client.get(f"/api/v1/jobs/{jid}", headers=auth).json()["status"] == "queued"
    assert env.drain() == 1
    job = env.client.get(f"/api/v1/jobs/{jid}", headers=auth).json()
    assert job["status"] == "succeeded" and job["result"]["decisions"] == {"Decision.ACCEPT": 1} or job["result"]["facts"] == 1
    assert [j["id"] for j in env.client.get("/api/v1/jobs?status=succeeded", headers=auth).json()] == [jid]

    ent = env.client.get("/api/v1/entities", params={"name": "paymentservice"}, headers=auth).json()
    cur = ent["facts"]["payment_provider"]["current"]
    assert [m["object"] for m in cur] == ["Stripe"] and cur[0]["evidence"][0]["source_uri"] == "arch.md"
    hits = env.client.post("/api/v1/memories/search", headers=auth, json={"query": "which payment provider PaymentService"}).json()
    assert hits and hits[0]["object"] == "Stripe" and hits[0]["state"] == "current"
    ans = env.client.post("/api/v1/agent/ask", headers=auth, json={"question": "Which provider does PaymentService use?"}).json()
    assert ans["values"] == ["Stripe"] and ans["used_memory"] and not ans["unknown"]
    g = env.client.get(f"/api/v1/entities/{ent['id']}/graph", headers=auth).json()
    assert {n["name"] for n in g["nodes"]} == {"PaymentService", "Stripe"} and len(g["edges"]) == 1


def test_reingesting_identical_content_is_idempotent(env, auth):
    env.script["DOC"] = [fact("PaymentService", "owned_by", "PaymentsTeam", "owned by PaymentsTeam DOC")]
    ingest(env, auth, "owned by PaymentsTeam DOC")
    second = ingest(env, auth, "owned by PaymentsTeam DOC", uri="again.md")
    assert second["result"]["skipped_duplicate"] is True
    assert len(env.service.ctx("ten_acme").store.memories("status='active'")) == 1


def test_failed_job_is_recorded_with_its_error(env, auth):
    env.fail_with = RuntimeError("provider exploded")
    job = ingest(env, auth, "anything")
    assert job["status"] == "failed" and "provider exploded" in job["error"] and job["attempts"] == 1
    env.fail_with = None


def test_sync_memory_endpoint_returns_decisions(env, auth):
    env.script["SYNC"] = [fact("PaymentService", "primary_database", "Oracle", "primary database is Oracle SYNC")]
    r = env.client.post("/api/v1/memories", headers=auth, json={
        "text": "primary database is Oracle SYNC", "uri": "d", "source_type": "official_docs", "doc_date": "2025-01-01"})
    assert r.status_code == 200 and r.json()["outcomes"][0]["decision"] in ("ACCEPT", "Decision.ACCEPT")


# ---------------------------------------------------------------- isolation
def test_tenants_cannot_see_each_others_data(env, auth):
    other = env.tenant("globex")
    env.script["DOC"] = [fact("PaymentService", "payment_provider", "Stripe", "uses Stripe DOC", valid_from="2025-01-01")]
    job = ingest(env, auth, "uses Stripe DOC")
    ent = env.client.get("/api/v1/entities", params={"name": "PaymentService"}, headers=auth).json()
    assert env.client.get(f"/api/v1/jobs/{job['id']}", headers=other).status_code == 404
    assert env.client.get(f"/api/v1/entities/{ent['id']}", headers=other).status_code == 404
    assert env.client.get("/api/v1/entities", params={"name": "PaymentService"}, headers=other).status_code == 404
    assert env.client.post("/api/v1/memories/search", headers=other, json={"query": "PaymentService Stripe"}).json() == []
    assert env.client.get("/api/v1/jobs", headers=other).json() == []
    mem_id = ent["facts"]["payment_provider"]["current"][0]["id"]
    assert env.client.post(f"/api/v1/memories/{mem_id}/invalidate", headers=other).status_code == 404
    assert env.client.delete(f"/api/v1/memories/{mem_id}", headers=other).status_code == 404
    # the same document under the second tenant creates an independent copy
    ingest(env, other, "uses Stripe DOC")
    assert len(env.service.ctx("ten_acme").store.memories("status='active'")) == 1
    assert len(env.service.ctx("ten_globex").store.memories("status='active'")) == 1


# ---------------------------------------------------------------- validation and errors
@pytest.mark.parametrize("body", [
    {"text": "", "uri": "u"},
    {"text": "x", "uri": ""},
    {"text": "x", "uri": "u", "source_type": "wizard"},
    {"text": "x", "uri": "u", "doc_date": "01/02/2025"},
    {"uri": "u"},
])
def test_invalid_ingest_requests_are_422(env, auth, body):
    r = env.client.post("/api/v1/ingest", headers=auth, json=body)
    assert r.status_code == 422 and r.json()["error"] == "validation_error"


def test_not_found_and_error_shape(env, auth):
    r = env.client.get("/api/v1/jobs/job_missing", headers=auth)
    assert r.status_code == 404 and set(r.json()) == {"error", "detail", "request_id"} and r.json()["error"] == "not_found"
    assert env.client.get("/api/v1/agent/tasks/task_missing", headers=auth).status_code == 404
    assert env.client.get("/api/v1/entities/ent_missing/graph", headers=auth).status_code == 404


def test_provider_failures_map_to_http_errors(env, auth):
    env.fail_with = ProviderConfigError("GOOGLE_API_KEY is not set (required by provider 'gemini')")
    r = env.client.post("/api/v1/agent/ask", headers=auth, json={"question": "q"})
    assert r.status_code == 503 and r.json()["error"] == "provider_not_configured"
    env.fail_with = LLMError("upstream down")
    assert env.client.post("/api/v1/agent/ask", headers=auth, json={"question": "q"}).status_code == 502
    env.fail_with = None


# ---------------------------------------------------------------- review flow (poisoning)
def seed_poisoning(env, auth):
    env.script["CFG"] = [fact("PaymentService", "payment_provider", "Razorpay", "provider: Razorpay CFG", valid_from="2025-01-01")]
    ingest(env, auth, "provider: Razorpay CFG", uri="deploy.yaml", source_type="production_config")
    env.script["CHAT"] = [fact("PaymentService", "payment_provider", "Stripe", "we use Stripe CHAT", valid_from="2025-06-01")]
    ingest(env, auth, "we use Stripe CHAT", uri="chat", source_type="developer", doc_date="2025-06-01")


def test_low_authority_contradiction_goes_to_review_and_can_be_approved(env, auth):
    seed_poisoning(env, auth)
    conflicts = env.client.get("/api/v1/conflicts", headers=auth).json()
    assert len(conflicts) == 1 and conflicts[0]["conflicted"]["object"] == "Stripe"
    assert [m["object"] for m in conflicts[0]["accepted"]] == ["Razorpay"]
    reviews = env.client.get("/api/v1/reviews", headers=auth).json()
    assert len(reviews) == 1
    d = env.client.post(f"/api/v1/reviews/{reviews[0]['id']}/decision", headers=auth, json={"decision": "approve"})
    assert d.status_code == 200 and d.json()["decision"] == "approved"
    assert env.client.get("/api/v1/reviews", headers=auth).json() == []
    assert len(env.client.get("/api/v1/reviews?status=approved", headers=auth).json()) == 1
    ent = env.client.get("/api/v1/entities", params={"name": "PaymentService"}, headers=auth).json()
    assert [m["object"] for m in ent["facts"]["payment_provider"]["current"]] == ["Stripe"]  # human decision applied
    assert env.client.post(f"/api/v1/reviews/{reviews[0]['id']}/decision", headers=auth, json={"decision": "reject"}).status_code == 409


def test_rejecting_a_review_keeps_the_accepted_fact(env, auth):
    seed_poisoning(env, auth)
    rid = env.client.get("/api/v1/reviews", headers=auth).json()[0]["id"]
    assert env.client.post(f"/api/v1/reviews/{rid}/decision", headers=auth, json={"decision": "reject"}).status_code == 200
    ent = env.client.get("/api/v1/entities", params={"name": "PaymentService"}, headers=auth).json()
    assert [m["object"] for m in ent["facts"]["payment_provider"]["current"]] == ["Razorpay"]
    assert env.client.get("/api/v1/conflicts", headers=auth).json() == []


# ---------------------------------------------------------------- memory management
def test_verify_invalidate_history_and_delete(env, auth):
    env.script["DOC-ONE"] = [fact("PaymentService", "payment_provider", "Stripe", "uses Stripe DOC-ONE", valid_from="2025-01-01")]
    ingest(env, auth, "uses Stripe DOC-ONE", source_type="developer")
    env.script["DOC-TWO"] = [fact("PaymentService", "payment_provider", "Razorpay", "uses Razorpay DOC-TWO", valid_from="2025-09-01")]
    ingest(env, auth, "uses Razorpay DOC-TWO", source_type="architecture_repo", doc_date="2025-09-01")
    ent = env.client.get("/api/v1/entities", params={"name": "PaymentService"}, headers=auth).json()
    cur = ent["facts"]["payment_provider"]["current"][0]
    assert cur["object"] == "Razorpay"
    hist = env.client.get(f"/api/v1/memories/{cur['id']}/history", headers=auth).json()
    assert [(h["object"], h["valid_until"]) for h in hist if h["status"] == "active"] == [("Stripe", "2025-09-01"), ("Razorpay", None)]
    v = env.client.post(f"/api/v1/memories/{cur['id']}/verify", headers=auth).json()
    assert v["source_authority"] >= 0.85 and v["last_verified"]
    assert env.client.post(f"/api/v1/memories/{cur['id']}/invalidate", headers=auth).json()["status"] == "invalidated"
    assert env.client.delete(f"/api/v1/memories/{cur['id']}", headers=auth).status_code == 204
    assert env.service.ctx("ten_acme").store.evidence_of(cur["id"]) == []


def test_entity_merge_moves_aliases_and_facts_and_is_reversible(env, auth):
    store = env.service.ctx("ten_acme").store
    a, b = store.add_entity("service", "PaymentService", "2025-01-01"), store.add_entity("service", "PayService", "2025-01-01")
    store.add_alias(a.id, "PaymentService", "paymentservice", "t", 1.0, "2025-01-01")
    store.add_alias(b.id, "PayService", "payservice", "t", 1.0, "2025-01-01")
    env.script["MERGE-DOC"] = [fact("PayService", "owned_by", "PaymentsTeam", "PayService owned by PaymentsTeam MERGE-DOC")]
    ingest(env, auth, "PayService owned by PaymentsTeam MERGE-DOC")
    m = env.client.post(f"/api/v1/entities/{a.id}/merge", headers=auth, json={"merged_entity_id": b.id}).json()
    assert m["moved_aliases"] == 1 and m["moved_memories"] >= 1
    merged = env.client.get(f"/api/v1/entities/{a.id}", headers=auth).json()
    assert set(merged["aliases"]) == {"PaymentService", "PayService"} and merged["facts"]["owned_by"]["current"]
    assert env.client.get("/api/v1/entities", params={"name": "PayService"}, headers=auth).json()["id"] == a.id
    assert env.client.post(f"/api/v1/entities/merges/{m['merge_id']}/revert", headers=auth).json()["reverted"] is True
    assert env.client.get(f"/api/v1/entities/{b.id}", headers=auth).json()["facts"]["owned_by"]["current"]
    assert env.client.post(f"/api/v1/entities/merges/{m['merge_id']}/revert", headers=auth).status_code == 409


def test_merge_rules(env, auth):
    store = env.service.ctx("ten_acme").store
    svc, team = store.add_entity("service", "S", "2025-01-01"), store.add_entity("team", "T", "2025-01-01")
    assert env.client.post(f"/api/v1/entities/{svc.id}/merge", headers=auth, json={"merged_entity_id": team.id}).status_code == 409
    assert env.client.post(f"/api/v1/entities/{svc.id}/merge", headers=auth, json={"merged_entity_id": svc.id}).status_code == 409
    assert env.client.post(f"/api/v1/entities/{svc.id}/merge", headers=auth, json={"merged_entity_id": "ent_x"}).status_code == 404


# ---------------------------------------------------------------- agent tasks (external execution)
def test_task_two_phase_flow_learns_from_reported_outcomes(env, auth):
    r = env.client.post("/api/v1/agent/tasks", headers=auth, json={"task": "Deploy PaymentService", "allowed_steps": STEPS,
                                                                   "with_facts": False})
    assert r.status_code == 201
    t = r.json()
    assert t["status"] == "planned" and t["plan"] == STEPS and t["lessons"] == []
    assert env.client.get(f"/api/v1/agent/tasks/{t['task_id']}", headers=auth).json()["status"] == "planned"
    assert env.client.get(f"/api/v1/tasks/{t['task_id']}/memory-context", headers=auth).json()["task_id"] == t["task_id"]
    out = env.client.post(f"/api/v1/agent/tasks/{t['task_id']}/outcome", headers=auth,
                          json={"outcome": "failure", "error": "ERR-1 db"})
    assert out.status_code == 200 and out.json()["status"] == "failed" and out.json()["episode_id"]
    assert env.client.post(f"/api/v1/agent/tasks/{t['task_id']}/outcome", headers=auth,
                           json={"outcome": "success"}).status_code == 409
    for i in range(2):  # two more failures of the same kind -> a lesson is promoted (min evidence 3)
        t2 = env.client.post("/api/v1/agent/tasks", headers=auth, json={"task": f"Deploy X{i}", "allowed_steps": STEPS,
                                                                        "with_facts": False}).json()
        env.client.post(f"/api/v1/agent/tasks/{t2['task_id']}/outcome", headers=auth, json={"outcome": "failure", "error": "ERR-1 db"})
    t4 = env.client.post("/api/v1/agent/tasks", headers=auth, json={"task": "Deploy a service that uses db", "allowed_steps": STEPS,
                                                                    "with_facts": False}).json()
    assert len(t4["lessons"]) == 1


def test_task_validation_and_isolation(env, auth):
    assert env.client.post("/api/v1/agent/tasks", headers=auth, json={"task": "x", "allowed_steps": []}).status_code == 422
    assert env.client.post("/api/v1/agent/tasks/task_x/outcome", headers=auth, json={"outcome": "maybe"}).status_code == 422
    other = env.tenant("globex")
    t = env.client.post("/api/v1/agent/tasks", headers=auth, json={"task": "x", "allowed_steps": STEPS, "with_facts": False}).json()
    assert env.client.get(f"/api/v1/agent/tasks/{t['task_id']}", headers=other).status_code == 404
    assert env.client.post(f"/api/v1/agent/tasks/{t['task_id']}/outcome", headers=other, json={"outcome": "success"}).status_code == 404
