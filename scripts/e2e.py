"""Live end-to-end scenario against the running stack (`make up` first; needs a real LLM provider key).

Drives everything over HTTP exactly as a client would: async ingest through the queue and worker, temporal questions,
alias resolution, the poisoning claim landing in review, the review decision, and the two-phase task flow.
Exit code 0 only if every check passes.

    make up && make e2e
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import time

import httpx

BASE = os.getenv("API_URL", f"http://localhost:{os.getenv('API_PORT', '8090')}")
DB_URL = os.getenv("E2E_DATABASE_URL", f"postgresql://memory:memory@localhost:{os.getenv('POSTGRES_PORT', '5439')}/memory")
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def wait_ready(client: httpx.Client, timeout: float = 90) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = client.get("/readyz")
            if r.status_code == 200:
                print(f"api ready: {r.json()['checks']}")
                return
            last = r.text
        except httpx.HTTPError as e:
            last = str(e)
        time.sleep(2)
    sys.exit(f"API not ready after {timeout}s: {last}")


def bootstrap_tenant() -> tuple[str, str]:
    """Create a fresh tenant + key directly in the database (the operator step: `memory-agent tenant create`)."""
    from memory_agent import tenants
    from memory_agent.store.postgres import PostgresStore

    store = PostgresStore(DB_URL)
    tid = tenants.create_tenant(store, f"e2e-{int(time.time())}")
    key = tenants.create_api_key(store, tid, "e2e")
    store.close()
    return tid, key


def poll_job(client: httpx.Client, job_id: str, timeout: float = 180) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/v1/jobs/{job_id}").json()
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(1.5)
    raise TimeoutError(f"job {job_id} still running after {timeout}s")


def main() -> int:
    today = dt.datetime.now(dt.UTC).date()
    d_old, d_mig, d_chat = today - dt.timedelta(days=600), today - dt.timedelta(days=15), today - dt.timedelta(days=5)
    d_mid = d_old + (d_mig - d_old) // 2
    anon = httpx.Client(base_url=BASE, timeout=60)
    wait_ready(anon)
    tid, key = bootstrap_tenant()
    print(f"tenant {tid}")
    client = httpx.Client(base_url=BASE, timeout=120, headers={"Authorization": f"Bearer {key}"})

    print("\n1. authentication")
    check("no key -> 401", anon.get("/api/v1/reviews").status_code == 401)
    check("bad key -> 401", anon.get("/api/v1/reviews", headers={"Authorization": "Bearer mem_bad"}).status_code == 401)

    print("\n2. async ingest through queue + worker")
    docs = [
        (d_old, "official_docs", "arch-2025.md",
         ("PaymentService uses Stripe as its payment provider. It is owned by the Payments Team and its primary "
          "database is PostgreSQL.")),
        (d_mig, "architecture_repo", "adr-042.md",
         (f"ADR-042: On {d_mig} the payments-svc migration completed. We moved from Stripe to Razorpay because enterprise "
          f"customers required a specific settlement workflow. The primary database was migrated from PostgreSQL to "
          f"Oracle on the same date. Tomorrow I will be testing the service locally.")),
        (d_chat, "developer", "chat/thread-77",
         "hey, I think the payment service still uses Stripe, right? pretty sure that's what we run in prod"),
    ]
    job_ids = []
    for d, st, uri, text in docs:
        r = client.post("/api/v1/ingest", json={"text": text, "uri": uri, "source_type": st, "doc_date": d.isoformat()})
        check(f"POST /ingest {uri} -> 202", r.status_code == 202, r.text)
        job_ids.append(r.json()["job_id"])
    for (_, _, uri, _), jid in zip(docs, job_ids, strict=True):
        j = poll_job(client, jid)
        check(f"job for {uri} succeeded", j["status"] == "succeeded", str(j.get("error")))
        if j["status"] == "succeeded":
            print(f"      {uri}: {j['result']['decisions']}")

    print("\n3. questions")
    def ask(q: str) -> dict:
        r = client.post("/api/v1/agent/ask", json={"question": q})
        assert r.status_code == 200, r.text
        return r.json()

    a = ask("Which payment provider does PaymentService currently use?")
    print(f"      current: {a['answer']}")
    check("current provider is Razorpay", "razorpay" in a["answer"].lower(), a["answer"])
    check("dispute is surfaced (unverified Stripe claim)", bool(a["conflicts"]) or "disput" in a["answer"].lower()
          or "conflict" in a["answer"].lower() or "unverified" in a["answer"].lower(), a["answer"])
    a = ask(f"Which payment provider did PaymentService use on {d_mid}?")
    print(f"      as of {d_mid}: {a['answer']}")
    check("historical provider is Stripe", "stripe" in a["answer"].lower(), a["answer"])
    a = ask("When did PaymentService switch from Stripe to Razorpay?")
    print(f"      when: {a['answer']}")
    check("change date found", d_mig.isoformat() in a["answer"] or d_mig.strftime("%B") in a["answer"], a["answer"])
    a = ask("What is the primary database of PaymentService?")
    check("database is Oracle", "oracle" in a["answer"].lower(), a["answer"])

    print("\n4. entities, aliases and graph")
    r = client.get("/api/v1/entities", params={"name": "payments-svc"})
    check("alias 'payments-svc' resolves to PaymentService", r.status_code == 200 and r.json()["name"] == "PaymentService", r.text)
    ent = r.json()
    hist = ent["facts"]["payment_provider"]["history"]
    check("timeline has Stripe (ended) then Razorpay (current)",
          [(h["object"], h["valid_until"] is not None) for h in hist] == [("Stripe", True), ("Razorpay", False)], str(hist))
    g = client.get(f"/api/v1/entities/{ent['id']}/graph").json()
    check("graph links service to Razorpay and Oracle", {"Razorpay", "Oracle"} <= {n["name"] for n in g["nodes"]})
    hits = client.post("/api/v1/memories/search", json={"query": "PaymentService payment provider"}).json()
    check("search returns current Razorpay first", bool(hits) and hits[0]["object"] == "Razorpay", str(hits[:1]))

    print("\n5. poisoning -> conflict -> human review")
    conflicts = client.get("/api/v1/conflicts").json()
    check("one unresolved conflict (developer says Stripe)", len(conflicts) == 1 and conflicts[0]["conflicted"]["object"] == "Stripe")
    reviews = client.get("/api/v1/reviews").json()
    check("one pending review", len(reviews) == 1)
    if reviews:
        r = client.post(f"/api/v1/reviews/{reviews[0]['id']}/decision", json={"decision": "reject", "note": "config says Razorpay"})
        check("review rejected", r.status_code == 200, r.text)
    check("no conflicts remain", client.get("/api/v1/conflicts").json() == [])
    cur = client.get(f"/api/v1/entities/{ent['id']}").json()["facts"]["payment_provider"]["current"]
    check("accepted fact unchanged after rejection", [m["object"] for m in cur] == ["Razorpay"])

    print("\n6. idempotency and tenant isolation")
    r = client.post("/api/v1/ingest", json={"text": docs[0][3], "uri": "again.md", "source_type": "official_docs",
                                            "doc_date": d_old.isoformat()})
    j = poll_job(client, r.json()["job_id"])
    check("re-ingesting identical content is a no-op", j["result"]["skipped_duplicate"] is True, str(j))
    _, key2 = bootstrap_tenant()
    other = httpx.Client(base_url=BASE, timeout=30, headers={"Authorization": f"Bearer {key2}"})
    check("other tenant cannot see the entity", other.get(f"/api/v1/entities/{ent['id']}").status_code == 404)
    check("other tenant cannot see the job", other.get(f"/api/v1/jobs/{job_ids[0]}").status_code == 404)

    print("\n7. task flow (plan, external execution, outcome, learning)")
    steps = ["run_tests", "build_image", "deploy_staging", "smoke_tests", "deploy_production", "notify_team"]
    r = client.post("/api/v1/agent/tasks", json={"task": "Deploy PaymentService v3, which changes the Oracle schema",
                                                  "allowed_steps": steps})
    check("task planned", r.status_code == 201 and bool(r.json()["plan"]), r.text)
    if r.status_code == 201:
        t = r.json()
        print(f"      plan: {' > '.join(t['plan'])}")
        check("plan uses organizational facts", "oracle" in t["memory_context"].lower() or "razorpay" in t["memory_context"].lower())
        o = client.post(f"/api/v1/agent/tasks/{t['task_id']}/outcome",
                        json={"outcome": "failure", "error": "ERR-4021: schema migration ran after application deploy"})
        check("outcome recorded as an episode", o.status_code == 200 and bool(o.json().get("episode_id")), o.text)
        check("cannot report the outcome twice", client.post(f"/api/v1/agent/tasks/{t['task_id']}/outcome",
                                                              json={"outcome": "success"}).status_code == 409)

    print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
