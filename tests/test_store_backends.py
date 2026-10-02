"""The same store behavior on SQLite and on Postgres+pgvector (Postgres variants need DATABASE_URL)."""
import datetime as dt
import threading

import numpy as np
import pytest

from memory_agent import jobs, tenants
from memory_agent.models import Memory, new_id
from memory_agent.store.base import BaseStore

DIM = 256


def vec(*hot: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for i in hot:
        v[i] = 1.0
    return v / np.linalg.norm(v)


def mem(store: BaseStore, subj: str, obj: str, frm="2025-01-01", **kw) -> Memory:
    m = Memory(id=new_id("mem"), tenant_id=store.tenant, subject_id=subj, relation="payment_provider", object_id=obj,
               valid_from=frm, observed_at="2025-01-02", created_at="2025-01-02", **kw)
    return store.insert_memory(m)


def test_entities_aliases_and_conflict_ignore(backend):
    e = backend.add_entity("service", "PaymentService", "2025-01-01")
    assert backend.get_entity(e.id).name == "PaymentService"
    assert backend.add_alias(e.id, "payments-svc", "paymentsservice", "t", 0.9, "2025-01-01")
    assert not backend.add_alias(e.id, "payments-svc", "paymentsservice", "t", 0.9, "2025-01-01")  # unique -> ignored
    assert backend.entity_by_alias_key("paymentsservice").id == e.id
    assert backend.aliases_of(e.id) == ["payments-svc"]
    assert [x.name for x in backend.list_entities("service")] == ["PaymentService"]


def test_bitemporal_close_keeps_history_and_evidence(backend):
    svc, stripe = backend.add_entity("service", "S", "2025-01-01"), backend.add_entity("provider", "Stripe", "2025-01-01")
    m = mem(backend, svc.id, stripe.id, confidence=0.7)
    backend.add_evidence(m.id, "developer", "doc1", "uses Stripe", 0.5, "2025-01-02")
    closed = backend.close_validity(m, "2026-09-11", "2026-09-12")
    old = backend.get_memory(m.id)
    assert old.recorded_until == "2026-09-12" and old.status == "superseded" and old.valid_until is None
    assert closed.valid_until == "2026-09-11" and closed.supersedes_id == m.id and closed.confidence == 0.7
    assert len(backend.evidence_of(closed.id)) == 1


def test_transaction_rolls_back_on_error(backend):
    with pytest.raises(RuntimeError), backend.transaction():
        backend.add_entity("service", "Ghost", "2025-01-01")
        raise RuntimeError("boom")
    assert backend.list_entities() == []
    with backend.transaction(), backend.transaction():  # re-entrant
        backend.add_entity("service", "Kept", "2025-01-01")
    assert [e.name for e in backend.list_entities()] == ["Kept"]


def test_tenant_isolation_for_rows_and_vectors(backend):
    other = backend.for_tenant("tenantB")
    a = backend.add_entity("service", "OnlyA", "2025-01-01")
    backend.put_vector("entity", a.id, vec(1))
    assert other.list_entities() == [] and other.get_entity(a.id) is None
    assert other.search_vectors("entity", vec(1), k=5) == []
    assert backend.search_vectors("entity", vec(1), k=5)[0][0] == a.id
    b = other.add_entity("service", "OnlyB", "2025-01-01")
    assert [e.name for e in backend.list_entities()] == ["OnlyA"] and [e.name for e in other.list_entities()] == ["OnlyB"]
    # a tenant cannot update another tenant's row by id
    backend.update("entities", b.id, {"name": "hijacked"})
    assert other.get_entity(b.id).name == "OnlyB"


def test_vector_search_ranking_filter_upsert_and_delete(backend):
    backend.put_vector("memory", "m1", vec(1, 2))
    backend.put_vector("memory", "m2", vec(3))
    backend.put_vector("memory", "m3", vec(1))
    hits = backend.search_vectors("memory", vec(1), k=3)
    assert [h[0] for h in hits][:2] == ["m3", "m1"] and hits[0][1] == pytest.approx(1.0, abs=1e-3)
    assert backend.search_vectors("memory", vec(1), k=3, ref_ids={"m2", "m1"})[0][0] == "m1"
    assert backend.search_vectors("memory", vec(1), ref_ids=set()) == []
    backend.put_vector("memory", "m2", vec(1))  # upsert replaces
    assert backend.search_vectors("memory", vec(1), k=1, ref_ids={"m2"})[0][1] == pytest.approx(1.0, abs=1e-3)
    assert backend.search_vectors("entity", vec(1)) == []  # kind is part of the key
    backend.delete_vectors("memory", ["m3"])
    assert "m3" not in [h[0] for h in backend.search_vectors("memory", vec(1), k=5)]


def test_purge_memory_removes_evidence_and_vectors(backend):
    s, o = backend.add_entity("service", "S", "2025-01-01"), backend.add_entity("db", "Oracle", "2025-01-01")
    m = mem(backend, s.id, o.id, content="secret text")
    backend.add_evidence(m.id, "doc", "u", "excerpt", 0.5, "2025-01-02")
    backend.put_vector("memory", m.id, vec(5))
    backend.purge_memory(m.id)
    got = backend.get_memory(m.id)
    assert got.status == "invalidated" and got.content == "" and backend.evidence_of(m.id) == []
    assert backend.search_vectors("memory", vec(5)) == []


def test_wrong_vector_dimension_is_rejected_on_postgres(backend):
    if type(backend).__name__ != "PostgresStore":
        pytest.skip("SQLite stores any dimension")
    with pytest.raises(ValueError, match="EMBED_DIM"):
        backend.put_vector("memory", "x", np.ones(DIM + 1, dtype=np.float32))


# ---------------------------------------------------------------- tenants / api keys
def test_api_keys_map_to_tenants_and_can_be_revoked(backend):
    t1 = tenants.create_tenant(backend, "Acme")
    key = tenants.create_api_key(backend, t1)
    assert key.startswith("mem_") and tenants.authenticate(backend, key) == t1
    assert tenants.authenticate(backend, key + "x") is None and tenants.authenticate(backend, "nope") is None
    stored = backend.fetch("SELECT key_hash, key_prefix FROM api_keys")[0]
    assert key not in stored["key_hash"] and stored["key_hash"] == tenants.hash_key(key)  # plaintext never stored
    assert tenants.revoke_api_key(backend, key) and tenants.authenticate(backend, key) is None
    with pytest.raises(ValueError):
        tenants.create_api_key(backend, "ten_missing")


# ---------------------------------------------------------------- job queue
def test_job_lifecycle_success(backend):
    jid = jobs.enqueue(backend, "ingest", {"text": "x"})
    assert jobs.get_job(backend, jid).status == "queued"
    j = jobs.claim(backend, "w1")
    assert j.id == jid and j.status == "running" and j.attempts == 1 and j.payload == {"text": "x"}
    assert jobs.claim(backend, "w2") is None  # already taken
    jobs.complete(backend, jid, {"memories": 2})
    done = jobs.get_job(backend, jid)
    assert done.status == "succeeded" and done.result == {"memories": 2} and done.finished_at
    assert jobs.get_job(backend.for_tenant("other"), jid) is None  # tenant-scoped reads


def test_job_retry_backoff_then_fail(backend):
    now = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    jid = jobs.enqueue(backend, "ingest", {}, max_attempts=2, now=now)
    j = jobs.claim(backend, "w", now=now)
    assert jobs.fail(backend, j, "boom", base_backoff=10, now=now) == "queued"
    assert jobs.claim(backend, "w", now=now + dt.timedelta(seconds=5)) is None  # backing off
    j2 = jobs.claim(backend, "w", now=now + dt.timedelta(seconds=11))
    assert j2.attempts == 2
    assert jobs.fail(backend, j2, "boom again", now=now + dt.timedelta(seconds=12)) == "failed"
    final = jobs.get_job(backend, jid)
    assert final.status == "failed" and final.error == "boom again"


def test_crashed_worker_job_is_reclaimed_after_visibility_timeout_then_failed(backend):
    t0 = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    jid = jobs.enqueue(backend, "ingest", {}, max_attempts=2, now=t0)
    assert jobs.claim(backend, "dead-worker", visibility_timeout=60, now=t0).attempts == 1
    assert jobs.claim(backend, "w2", visibility_timeout=60, now=t0 + dt.timedelta(seconds=30)) is None  # still leased
    again = jobs.claim(backend, "w2", visibility_timeout=60, now=t0 + dt.timedelta(seconds=90))
    assert again.id == jid and again.attempts == 2 and again.attempts >= again.max_attempts
    # worker dies again: attempts exhausted, so it is failed instead of looping forever
    assert jobs.claim(backend, "w3", visibility_timeout=60, now=t0 + dt.timedelta(seconds=200)) is None
    assert jobs.get_job(backend, jid).status == "failed"


def test_concurrent_workers_never_claim_the_same_job(backend):
    ids = {jobs.enqueue(backend, "ingest", {"n": i}) for i in range(12)}
    claimed: list[str] = []
    lock = threading.Lock()

    def worker(n: int):
        while (j := jobs.claim(backend, f"w{n}")) is not None:
            with lock:
                claimed.append(j.id)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(claimed) == sorted(ids) and len(claimed) == len(set(claimed))


def test_fifo_order_within_claims(backend):
    first = jobs.enqueue(backend, "ingest", {"n": 1})
    second = jobs.enqueue(backend, "ingest", {"n": 2})
    assert jobs.claim(backend, "w").id == first
    assert jobs.claim(backend, "w").id == second


# ---------------------------------------------------------------- Postgres only
@pytest.mark.integration
def test_schema_parity_sqlite_vs_postgres(backend):
    if type(backend).__name__ != "PostgresStore":
        pytest.skip("Postgres only")
    from memory_agent.store.sqlite import SqliteStore

    sq = SqliteStore()
    sqlite_cols = {}
    for r in sq.fetch("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        sqlite_cols[r["name"]] = {c["name"] for c in sq.fetch(f"PRAGMA table_info({r['name']})")}
    pg_cols: dict[str, set[str]] = {}
    for r in backend.fetch("SELECT table_name, column_name FROM information_schema.columns "
                           "WHERE table_schema = current_schema() AND table_name <> 'schema_migrations'"):
        pg_cols.setdefault(r["table_name"], set()).add(r["column_name"])
    pg_cols.pop("relations", None)  # the view is checked separately below
    assert set(sqlite_cols) == set(pg_cols)
    for table, cols in sqlite_cols.items():
        assert cols == pg_cols[table], table
    assert backend.fetch("SELECT count(*) AS n FROM relations")[0]["n"] == 0


@pytest.mark.integration
def test_migrations_are_idempotent_and_recorded(backend):
    import os

    from memory_agent.store.migrate import apply_migrations, migration_files
    if type(backend).__name__ != "PostgresStore":
        pytest.skip("Postgres only")
    schema = backend.fetch("SELECT current_schema() AS s")[0]["s"]
    assert apply_migrations(os.environ["DATABASE_URL"], DIM, schema) == []  # nothing pending on a second run
    versions = [r["version"] for r in backend.fetch("SELECT version FROM schema_migrations ORDER BY version")]
    assert versions == [f.stem for f in migration_files()]
