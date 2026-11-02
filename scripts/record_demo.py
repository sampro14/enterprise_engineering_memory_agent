"""Record the hosted demo's fixture from a running stack: real API responses, real LLM calls.

Needs two fresh tenants, one for each review decision (`make tenant NAME=...` prints a key):

    API_URL=http://localhost:8090 KEY_REJECT=mem_... KEY_APPROVE=mem_... uv run python scripts/record_demo.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parent.parent / "demo" / "fixture.json"

# Mirrors DEMO_DOCS and DEMO_QUESTIONS in src/memory_agent/ui/app.js; the replay matches on uri and question text.
DOCS = [
    {"doc_date": "2025-02-01", "source_type": "official_docs", "uri": "arch-2025.md",
     "text": "PaymentService uses Stripe as its payment provider. It is owned by the Payments Team and its primary "
             "database is PostgreSQL."},
    {"doc_date": "2026-09-15", "source_type": "architecture_repo", "uri": "adr-042.md",
     "text": "ADR-042: On 2026-09-11 the payments-svc migration completed. We moved from Stripe to Razorpay because "
             "enterprise customers required a specific settlement workflow. The primary database was migrated from "
             "PostgreSQL to Oracle on the same date. Tomorrow I will be testing the service locally."},
    {"doc_date": "2026-10-05", "source_type": "developer", "uri": "chat",
     "text": "hey, I think the payment service still uses Stripe, right?"},
]
QUESTIONS = [
    "Which payment provider does PaymentService currently use?",
    "Which payment provider did PaymentService use in March 2026?",
    "When did PaymentService switch from Stripe to Razorpay, and why?",
    "What is the primary database of PaymentService?",
]


def _ok(r: httpx.Response) -> dict | list:
    r.raise_for_status()
    return r.json()


def _ingest(c: httpx.Client, doc: dict) -> dict:
    job_id = _ok(c.post("/api/v1/ingest", json=doc))["job_id"]
    while True:
        job = _ok(c.get(f"/api/v1/jobs/{job_id}"))
        if job["status"] == "succeeded":
            return job
        if job["status"] == "failed":
            raise RuntimeError(f"{doc['uri']}: {job['error']}")
        time.sleep(0.5)


def _answers(c: httpx.Client) -> dict[str, dict]:
    return {q: _ok(c.post("/api/v1/agent/ask", json={"question": q})) for q in QUESTIONS}


def _client(key: str) -> httpx.Client:
    return httpx.Client(base_url=os.environ.get("API_URL", "http://localhost:8090"), timeout=120,
                        headers={"Authorization": f"Bearer {key}"})


def main() -> None:
    fixture: dict = {"recorded_at": time.strftime("%Y-%m-%d"), "answers": {}, "decisions": {}}
    for decision, key_var in (("reject", "KEY_REJECT"), ("approve", "KEY_APPROVE")):
        with _client(os.environ[key_var]) as c:
            jobs = {d["uri"]: _ingest(c, d) for d in DOCS}
            reviews = _ok(c.get("/api/v1/reviews?status=pending"))
            if len(reviews) != 1:
                raise RuntimeError(f"expected one pending review, got {len(reviews)}")
            if decision == "reject":  # both tenants ran the same story; the replay shows the first as "before review"
                fixture.update(jobs=jobs, reviews=reviews)
                fixture["answers"]["pending"] = _answers(c)
            fixture["decisions"][decision] = _ok(c.post(f"/api/v1/reviews/{reviews[0]['id']}/decision",
                                                        json={"decision": decision}))
            fixture["answers"][decision] = _answers(c)
            print(f"recorded {decision}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
