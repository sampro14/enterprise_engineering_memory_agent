import numpy as np

from memory_agent.models import Memory, new_id
from memory_agent.ontology import get_relation
from memory_agent.store.sqlite import MemoryStore


def _mem(store, subj, obj, frm="2025-01-01", **kw):
    m = Memory(id=new_id("mem"), tenant_id=store.tenant, subject_id=subj, relation="payment_provider",
               object_id=obj, valid_from=frm, observed_at="2025-01-02", created_at="2025-01-02", **kw)
    return store.insert_memory(m)


def test_ontology_cardinality():
    assert get_relation("primary_database").single_valued
    assert not get_relation("depends_on").single_valued
    assert get_relation("nope") is None


def test_close_validity_is_bitemporal():
    s = MemoryStore()
    svc = s.add_entity("service", "PaymentService", "2025-01-01")
    stripe = s.add_entity("provider", "Stripe", "2025-01-01")
    m = _mem(s, svc.id, stripe.id)
    s.add_evidence(m.id, "developer", "doc1", "uses Stripe", 0.5, "2025-01-02")
    closed = s.close_validity(m, "2026-09-11", "2026-09-12")
    old = s.get_memory(m.id)
    assert old.recorded_until == "2026-09-12" and old.status == "superseded"
    assert old.valid_until is None  # what the system believed before is preserved
    assert closed.valid_until == "2026-09-11" and closed.supersedes_id == m.id
    assert len(s.evidence_of(closed.id)) == 1


def test_tenant_isolation():
    a = MemoryStore()
    b = a.for_tenant("other")  # same physical db, different tenant view
    a.add_entity("service", "X", "2025-01-01")
    assert len(a.list_entities()) == 1
    assert b.list_entities() == []


def test_alias_unique_and_vectors():
    s = MemoryStore()
    e = s.add_entity("service", "PaymentService", "2025-01-01")
    assert s.add_alias(e.id, "payments-svc", "paymentssvc", "test", 0.9, "2025-01-01")
    assert not s.add_alias(e.id, "payments-svc", "paymentssvc", "test", 0.9, "2025-01-01")
    assert s.entity_by_alias_key("paymentssvc").id == e.id
    s.put_vector("mem", "a", np.array([1, 0], dtype=np.float32))
    s.put_vector("mem", "b", np.array([0, 1], dtype=np.float32))
    assert s.search_vectors("mem", np.array([1, 0], dtype=np.float32), k=1)[0][0] == "a"
