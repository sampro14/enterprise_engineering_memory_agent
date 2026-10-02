"""Application service: every operation the CLI, API and worker perform, in one place and tenant-scoped.

Callers pass a tenant id; the service binds a tenant-scoped store per operation, so a tenant can never see another
tenant's rows (also enforced and tested at the store level)."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from typing import Any

from .. import jobs as jobs_mod
from .. import review as review_mod
from .. import temporal
from ..confidence import compute
from ..config import Settings, get_settings
from ..context import Context
from ..entities import normalize_key
from ..graph import build_agent_graph
from ..ingest import IngestReport, ingest_text
from ..llm.base import Embedder, LLMClient
from ..models import Memory, SourceType, new_id
from ..retrieval import _graph_expand, build_context, mentioned_entities, retrieve
from ..store.base import BaseStore
from ..timeutil import utcnow_iso


class NotFoundError(LookupError):
    pass


class ConflictError(ValueError):
    pass


def _mem_dict(ctx: Context, m: Memory) -> dict[str, Any]:
    subj = ctx.store.get_entity(m.subject_id)
    obj = m.object_value or (ctx.store.get_entity(m.object_id).name if m.object_id else None)
    ev = [{"source_uri": r["source_uri"], "source_type": r["source_type"], "excerpt": r["excerpt"],
           "source_authority": r["source_authority"]} for r in ctx.store.evidence_of(m.id)]
    return {"id": m.id, "subject": subj.name if subj else m.subject_id, "subject_id": m.subject_id,
            "relation": m.relation, "object": obj, "object_id": m.object_id, "status": m.status,
            "confidence": m.confidence, "source_authority": m.source_authority, "valid_from": m.valid_from,
            "valid_until": m.valid_until, "observed_at": m.observed_at, "recorded_until": m.recorded_until,
            "last_verified": m.last_verified, "supersedes_id": m.supersedes_id, "evidence": ev}


class MemoryService:
    def __init__(self, settings: Settings, llm: LLMClient, embedder: Embedder, store: BaseStore):
        self.settings, self.llm, self.embedder, self.store = settings, llm, embedder, store

    @classmethod
    def from_settings(cls, settings: Settings | None = None, *, schema: str | None = None) -> MemoryService:
        from ..llm.factory import create_embedder, create_llm
        from ..store.factory import create_store

        s = settings or get_settings()
        return cls(s, create_llm(s), create_embedder(s), create_store(s, schema=schema))

    def ctx(self, tenant: str, clock=None) -> Context:
        c = Context(self.store.for_tenant(tenant), self.llm, self.embedder, self.settings)
        if clock:
            c.clock = clock
        return c

    # ---------------------------------------------------------------- write path
    def ingest(self, tenant: str, text: str, uri: str, source_type: SourceType = SourceType.DEVELOPER,
               doc_date: str | None = None) -> IngestReport:
        """Synchronous ingest, atomic per call: a failure anywhere rolls back every fact from this document."""
        c = self.ctx(tenant)
        with c.store.transaction():
            c.store.lock_tenant()  # serialize writers of one tenant; simple and correct (design 11)
            return ingest_text(c, text, uri, source_type, doc_date)

    def enqueue_ingest(self, tenant: str, text: str, uri: str, source_type: SourceType, doc_date: str | None) -> str:
        return jobs_mod.enqueue(self.store.for_tenant(tenant), "ingest",
                                {"text": text, "uri": uri, "source_type": source_type.value, "doc_date": doc_date},
                                max_attempts=self.settings.job_max_attempts)

    def process_job(self, job: jobs_mod.Job) -> dict[str, Any]:
        if job.kind != "ingest":
            raise ValueError(f"unknown job kind {job.kind}")
        p = job.payload
        rep = self.ingest(job.tenant_id, p["text"], p["uri"], SourceType(p["source_type"]), p.get("doc_date"))
        counts: dict[str, int] = defaultdict(int)
        for o in rep.outcomes:
            counts[str(o.decision)] += 1
        return {"skipped_duplicate": rep.skipped_duplicate, "facts": len(rep.outcomes), "decisions": dict(counts),
                "outcomes": [asdict(o) for o in rep.outcomes]}

    # ---------------------------------------------------------------- read path
    def ask(self, tenant: str, question: str) -> dict[str, Any]:
        c = self.ctx(tenant)
        st = build_agent_graph(c).invoke({"mode": "ask", "task_id": "ask", "user_query": question})
        a = st["answer"]
        ret = st.get("retrieval")
        return {"answer": a.answer, "values": a.values, "unknown": a.unknown, "citations": a.citations,
                "conflicts": list(ret.conflicts) if ret else [], "used_memory": ret is not None,
                "context": st.get("context_text", ""), "as_of": st["analysis"].as_of}

    def search(self, tenant: str, query: str, k: int = 10, as_of: str | None = None) -> list[dict[str, Any]]:
        """Hybrid retrieval without an LLM: entity mentions are matched against known aliases in the query."""
        c = self.ctx(tenant)
        ret = retrieve(c, query, mentioned_entities(c, query), None, as_of, k=k)
        out = []
        for r in ret.items[:k]:
            d = _mem_dict(c, r.memory)
            d.update(state=r.state, score=round(r.score, 4))
            out.append(d)
        return out

    def memory_context(self, tenant: str, question: str) -> dict[str, Any]:
        c = self.ctx(tenant)
        from ..answering import analyze
        a = analyze(c, question)
        ret = retrieve(c, question, a.entities, a.relations, a.as_of)
        return {"context": build_context(ret), "conflicts": ret.conflicts, "entities": a.entities, "as_of": a.as_of}

    def get_entity(self, tenant: str, entity_id: str) -> dict[str, Any]:
        c = self.ctx(tenant)
        e = c.store.get_entity(entity_id)
        if not e:
            raise NotFoundError(f"entity {entity_id} not found")
        rels = sorted({m.relation for m in c.store.memories("subject_id=?", (e.id,))})
        facts = {rel: {"current": [_mem_dict(c, m) for m in temporal.current(c, e.id, rel)],
                       "history": [_mem_dict(c, m) for m in temporal.history(c, e.id, rel)],
                       "conflicts": [_mem_dict(c, m) for m in temporal.open_conflicts(c, e.id, rel)]} for rel in rels}
        return {"id": e.id, "name": e.name, "type": e.entity_type, "aliases": c.store.aliases_of(e.id), "facts": facts}

    def find_entity(self, tenant: str, name: str) -> str | None:
        e = self.ctx(tenant).store.entity_by_alias_key(normalize_key(name))
        return e.id if e else None

    def entity_graph(self, tenant: str, entity_id: str, as_of: str | None = None, depth: int = 2) -> dict[str, Any]:
        c = self.ctx(tenant)
        if not c.store.get_entity(entity_id):
            raise NotFoundError(f"entity {entity_id} not found")
        dist = {k: v for k, v in _graph_expand(c, [entity_id]).items() if v <= depth}
        ph = ",".join("?" * len(dist))
        edges = []
        for m in c.store.memories(f"status='active' AND recorded_until IS NULL AND object_id IN ({ph}) "
                                  f"AND subject_id IN ({ph})", [*dist, *dist]):
            if as_of and not ((m.valid_from or "") <= as_of and (m.valid_until is None or as_of < m.valid_until)):
                continue
            edges.append({"memory_id": m.id, "source": m.subject_id, "relation": m.relation, "target": m.object_id,
                          "valid_from": m.valid_from, "valid_until": m.valid_until, "confidence": m.confidence})
        used = {e["source"] for e in edges} | {e["target"] for e in edges} | {entity_id}
        nodes = [{"id": i, "name": ent.name, "type": ent.entity_type, "hops": dist[i]}
                 for i in used if (ent := c.store.get_entity(i))]
        return {"root": entity_id, "as_of": as_of, "nodes": nodes, "edges": edges}

    def memory_history(self, tenant: str, memory_id: str) -> list[dict[str, Any]]:
        c = self.ctx(tenant)
        m = c.store.get_memory(memory_id)
        if not m:
            raise NotFoundError(f"memory {memory_id} not found")
        rows = c.store.memories("subject_id=? AND relation=?", (m.subject_id, m.relation))
        rows.sort(key=lambda x: (x.valid_from or "", x.created_at or "", x.id))
        return [_mem_dict(c, x) for x in rows]

    # ---------------------------------------------------------------- memory management
    def create_memory(self, tenant: str, text: str, source_type: SourceType, uri: str, doc_date: str | None):
        """A candidate memory supplied by a client goes through the same extraction and validation as any document."""
        return self.ingest(tenant, text, uri, source_type, doc_date)

    def verify_memory(self, tenant: str, memory_id: str) -> dict[str, Any]:
        """A human confirms a memory: authority rises to at least 'verified' and confidence is recomputed."""
        c = self.ctx(tenant)
        m = c.store.get_memory(memory_id)
        if not m:
            raise NotFoundError(f"memory {memory_id} not found")
        now = c.now()
        n = max(len({r["source_uri"] for r in c.store.evidence_of(m.id)}), 1)
        auth = max(m.source_authority, 0.6)
        conf = max(m.confidence, compute(c.settings.weights, auth, n, now, now))
        c.store.update("memories", m.id, {"last_verified": now, "source_authority": auth, "confidence": conf})
        return _mem_dict(c, c.store.get_memory(m.id))

    def invalidate_memory(self, tenant: str, memory_id: str) -> dict[str, Any]:
        c = self.ctx(tenant)
        if not c.store.get_memory(memory_id):
            raise NotFoundError(f"memory {memory_id} not found")
        c.store.update("memories", memory_id, {"status": "invalidated"})
        return _mem_dict(c, c.store.get_memory(memory_id))

    def delete_memory(self, tenant: str, memory_id: str) -> None:
        c = self.ctx(tenant)
        if not c.store.get_memory(memory_id):
            raise NotFoundError(f"memory {memory_id} not found")
        c.store.purge_memory(memory_id)

    def conflicts(self, tenant: str) -> list[dict[str, Any]]:
        c = self.ctx(tenant)
        out = []
        for m in c.store.memories("status='conflicted'"):
            accepted = [_mem_dict(c, x) for x in temporal.current(c, m.subject_id, m.relation)]
            out.append({"conflicted": _mem_dict(c, m), "accepted": accepted})
        return out

    # ---------------------------------------------------------------- entity merge (reversible)
    def merge_entities(self, tenant: str, kept_id: str, merged_id: str) -> dict[str, Any]:
        c = self.ctx(tenant)
        kept, merged = c.store.get_entity(kept_id), c.store.get_entity(merged_id)
        if not kept or not merged:
            raise NotFoundError("entity not found")
        if kept_id == merged_id:
            raise ConflictError("cannot merge an entity into itself")
        if kept.entity_type != merged.entity_type:
            raise ConflictError("never auto-merge entities of different types (design 14B)")
        mid = new_id("mrg")
        with c.store.transaction():
            alias_ids = [r["id"] for r in c.store.fetch(
                "SELECT id FROM entity_aliases WHERE tenant_id=? AND entity_id=?", (c.store.tenant, merged_id))]
            subj_ids = [r["id"] for r in c.store.fetch(
                "SELECT id FROM memories WHERE tenant_id=? AND subject_id=?", (c.store.tenant, merged_id))]
            obj_ids = [r["id"] for r in c.store.fetch(
                "SELECT id FROM memories WHERE tenant_id=? AND object_id=?", (c.store.tenant, merged_id))]
            c.store.execute("UPDATE entity_aliases SET entity_id=? WHERE tenant_id=? AND entity_id=?",
                            (kept_id, c.store.tenant, merged_id))
            c.store.execute("UPDATE memories SET subject_id=? WHERE tenant_id=? AND subject_id=?",
                            (kept_id, c.store.tenant, merged_id))
            c.store.execute("UPDATE memories SET object_id=? WHERE tenant_id=? AND object_id=?",
                            (kept_id, c.store.tenant, merged_id))
            c.store.insert("entity_merges", {
                "id": mid, "tenant_id": c.store.tenant, "kept_entity_id": kept_id, "merged_entity_id": merged_id,
                "merged_name": merged.name, "created_at": utcnow_iso(),
                "moved_json": json.dumps({"aliases": alias_ids, "subject": subj_ids, "object": obj_ids})})
        return {"merge_id": mid, "kept": kept_id, "merged": merged_id, "moved_aliases": len(alias_ids),
                "moved_memories": len(subj_ids) + len(obj_ids),
                "note": "re-check conflicts: the merged entity's facts now share one subject"}

    def unmerge(self, tenant: str, merge_id: str) -> dict[str, Any]:
        c = self.ctx(tenant)
        rows = c.store.fetch("SELECT * FROM entity_merges WHERE id=? AND tenant_id=?", (merge_id, c.store.tenant))
        if not rows:
            raise NotFoundError(f"merge {merge_id} not found")
        r = rows[0]
        if r["reverted_at"]:
            raise ConflictError("merge already reverted")
        moved = json.loads(r["moved_json"])
        with c.store.transaction():
            for col, ids in (("subject_id", moved["subject"]), ("object_id", moved["object"])):
                for i in ids:
                    c.store.update("memories", i, {col: r["merged_entity_id"]})
            for i in moved["aliases"]:
                c.store.update("entity_aliases", i, {"entity_id": r["merged_entity_id"]})
            c.store.update("entity_merges", merge_id, {"reverted_at": utcnow_iso()})
        return {"merge_id": merge_id, "reverted": True}

    # ---------------------------------------------------------------- reviews
    def reviews(self, tenant: str, status: str = "pending") -> list[dict[str, Any]]:
        c = self.ctx(tenant)
        rows = c.store.fetch("SELECT * FROM reviews WHERE tenant_id=? AND status=? ORDER BY created_at, id",
                             (c.store.tenant, status))
        return [{"id": r["id"], "reason": r["reason"], "status": r["status"], "memory_id": r["memory_id"],
                 "candidate": json.loads(r["candidate_json"]) if r["candidate_json"] else None,
                 "created_at": r["created_at"], "decided_at": r["decided_at"], "reviewer": r["reviewer"]} for r in rows]

    def decide_review(self, tenant: str, review_id: str, approve: bool, reviewer: str = "api",
                      note: str | None = None) -> dict[str, Any]:
        c = self.ctx(tenant)
        if c.store.get_review(review_id) is None:
            raise NotFoundError(f"review {review_id} not found")
        try:
            if approve:
                mem_id = review_mod.approve(c, review_id, reviewer, note)
            else:
                review_mod.reject(c, review_id, reviewer, note)
                mem_id = None
        except review_mod.ReviewError as e:
            raise ConflictError(str(e)) from e
        return {"review_id": review_id, "decision": "approved" if approve else "rejected", "memory_id": mem_id}

    # ---------------------------------------------------------------- tasks (external execution, two phases)
    def create_task(self, tenant: str, task: str, allowed_steps: list[str], with_facts: bool = True) -> dict[str, Any]:
        c = self.ctx(tenant)
        tid = new_id("task")
        st = build_agent_graph(c).invoke({"mode": "plan", "task_id": tid, "task": task, "allowed_steps": allowed_steps,
                                          "with_facts": with_facts})
        now = utcnow_iso()
        exps = [{k: v for k, v in e.items()} for e in st.get("experiences", [])]
        c.store.insert("tasks", {"id": tid, "tenant_id": c.store.tenant, "task": task,
                                 "allowed_steps": json.dumps(allowed_steps), "plan": json.dumps(st["plan"]),
                                 "experiences": json.dumps(exps), "context": st.get("context_text", ""),
                                 "status": "planned", "created_at": now, "updated_at": now})
        return self._task_view(c.store.fetch("SELECT * FROM tasks WHERE id=?", (tid,))[0])

    def get_task(self, tenant: str, task_id: str) -> dict[str, Any]:
        c = self.ctx(tenant)
        rows = c.store.fetch("SELECT * FROM tasks WHERE id=? AND tenant_id=?", (task_id, c.store.tenant))
        if not rows:
            raise NotFoundError(f"task {task_id} not found")
        return self._task_view(rows[0])

    def report_outcome(self, tenant: str, task_id: str, outcome: str, error: str | None) -> dict[str, Any]:
        c = self.ctx(tenant)
        rows = c.store.fetch("SELECT * FROM tasks WHERE id=? AND tenant_id=?", (task_id, c.store.tenant))
        if not rows:
            raise NotFoundError(f"task {task_id} not found")
        t = rows[0]
        if t["status"] != "planned":
            raise ConflictError(f"task {task_id} already has an outcome ({t['status']})")
        with c.store.transaction():
            c.store.lock_tenant()
            st = build_agent_graph(c).invoke({
                "mode": "outcome", "task_id": task_id, "task": t["task"], "plan": json.loads(t["plan"]),
                "outcome": outcome, "error": error, "experiences": json.loads(t["experiences"] or "[]"),
                "writeback": True, "use_experience": True})
            c.store.update("tasks", task_id, {"status": "completed" if outcome == "success" else "failed",
                                              "outcome": outcome, "error": error, "episode_id": st["episode_id"],
                                              "updated_at": utcnow_iso()})
        view = self._task_view(c.store.fetch("SELECT * FROM tasks WHERE id=?", (task_id,))[0])
        view["writeback"] = st.get("writeback_report", {})
        return view

    @staticmethod
    def _task_view(r) -> dict[str, Any]:
        return {"task_id": r["id"], "task": r["task"], "status": r["status"], "plan": json.loads(r["plan"] or "[]"),
                "lessons": [e["text"] for e in json.loads(r["experiences"] or "[]")], "memory_context": r["context"],
                "outcome": r["outcome"], "error": r["error"], "episode_id": r["episode_id"]}

    # ---------------------------------------------------------------- jobs
    def get_job(self, tenant: str, job_id: str) -> jobs_mod.Job:
        j = jobs_mod.get_job(self.store.for_tenant(tenant), job_id)
        if j is None:
            raise NotFoundError(f"job {job_id} not found")
        return j
