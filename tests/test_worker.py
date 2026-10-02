"""Worker loop: success, retry-then-fail, unknown job kinds, graceful stop, and a stuck-job takeover."""
import datetime as dt
import threading
import time

from conftest import fact
from memory_agent import jobs
from memory_agent.config import Settings
from memory_agent.llm.fake import FakeEmbedder, FakeLLM
from memory_agent.models import ExtractionResult, SourceType
from memory_agent.services.memory_service import MemoryService
from memory_agent.worker import Worker


def make(backend, handler=None, **settings):
    s = Settings(llm_provider="fake", embed_provider="fake", embed_dim=256, worker_poll_seconds=0.05, **settings)
    svc = MemoryService(s, FakeLLM(handler), FakeEmbedder(), backend)
    return svc, Worker(svc, "w-test")


def test_worker_processes_queue_in_order_and_records_results(backend):
    facts = {"D1": [fact("PaymentService", "payment_provider", "Stripe", "uses Stripe D1", valid_from="2025-01-01")],
             "D2": [fact("PaymentService", "primary_database", "Oracle", "database is Oracle D2")]}  # evidence is verbatim
    svc, w = make(backend, lambda p, sch: ExtractionResult(facts=next((f for m, f in facts.items() if m in p), [])))
    texts = {"D1": "PaymentService uses Stripe D1", "D2": "PaymentService database is Oracle D2"}
    ids = [svc.enqueue_ingest("t1", texts[m], f"{m}.md", SourceType.OFFICIAL_DOCS, "2025-01-01") for m in ("D1", "D2")]
    assert w.run_once() and w.run_once() and not w.run_once()
    jobs_done = [svc.get_job("t1", i) for i in ids]
    assert [j.status for j in jobs_done] == ["succeeded", "succeeded"] and w.done == 2
    assert jobs_done[0].result["facts"] == 1
    assert len(svc.ctx("t1").store.memories("status='active'")) == 2


def test_failing_job_is_retried_with_backoff_then_failed(backend):
    calls = []

    def boom(p, sch):
        calls.append(1)
        raise RuntimeError("provider down")

    svc, w = make(backend, boom, job_max_attempts=2)
    jid = svc.enqueue_ingest("t1", "x", "u", SourceType.DEVELOPER, None)
    assert w.run_once()
    j = svc.get_job("t1", jid)
    assert j.status == "queued" and j.attempts == 1 and "provider down" in j.error  # requeued with backoff
    assert not w.run_once()  # backing off: nothing runnable yet
    later = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=60)
    j2 = jobs.claim(backend, "w2", now=later)
    assert j2.id == jid and j2.attempts == 2
    assert jobs.fail(backend, j2, "still down") == "failed"
    assert svc.get_job("t1", jid).status == "failed" and len(calls) == 1


def test_unknown_job_kind_fails_cleanly(backend):
    svc, w = make(backend, job_max_attempts=1)
    jid = jobs.enqueue(backend.for_tenant("t1"), "bogus", {}, max_attempts=1)
    assert w.run_once()
    j = svc.get_job("t1", jid)
    assert j.status == "failed" and "unknown job kind" in j.error


def test_atomic_job_rolls_back_all_facts_when_it_fails_midway(backend):
    n = {"calls": 0}

    def handler(prompt, schema):
        n["calls"] += 1
        return ExtractionResult(facts=[fact("PaymentService", "payment_provider", "Stripe", "uses Stripe ATOM"),
                                       fact("PaymentService", "primary_database", "Oracle", "database is Oracle ATOM")])

    svc, w = make(backend, handler, job_max_attempts=1)
    real = svc.embedder.embed
    count = {"n": 0}

    def flaky_embed(texts):  # dies on the 4th embedding call, i.e. after the first fact was written
        count["n"] += 1
        if count["n"] == 4:
            raise RuntimeError("embedding service died")
        return real(texts)

    svc.embedder.embed = flaky_embed
    jid = svc.enqueue_ingest("t1", "uses Stripe ATOM database is Oracle ATOM", "atom.md", SourceType.OFFICIAL_DOCS, "2025-01-01")
    w.run_once()
    assert svc.get_job("t1", jid).status == "failed"
    store = svc.ctx("t1").store
    assert store.memories() == [] and store.list_entities() == []  # nothing half-written
    import hashlib
    h = hashlib.sha256(f"{SourceType.OFFICIAL_DOCS.value}|uses Stripe ATOM database is Oracle ATOM".encode()).hexdigest()
    assert not store.source_seen(h)  # the document was not recorded as ingested, so a retry will redo it


def test_run_forever_stops_gracefully(backend):
    svc, w = make(backend)
    stop = threading.Event()
    t = threading.Thread(target=w.run_forever, args=(stop,))
    t.start()
    svc.enqueue_ingest("t1", "x", "u", SourceType.DEVELOPER, None)
    deadline = time.time() + 5
    while w.done < 1 and time.time() < deadline:
        time.sleep(0.02)
    stop.set()
    t.join(timeout=5)
    assert not t.is_alive() and w.done == 1
