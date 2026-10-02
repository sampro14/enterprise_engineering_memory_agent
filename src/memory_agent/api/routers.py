"""HTTP endpoints (design 26). All handlers are sync: provider SDKs and DB drivers block, so FastAPI runs them in its
thread pool. Every route except /healthz and /readyz requires an API key; the tenant is taken from the key."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, Query, Request, Response

from .. import jobs as jobs_mod
from ..services.memory_service import MemoryService
from .deps import Principal, get_principal, get_service
from .schemas import (
    AskRequest,
    AskResponse,
    IngestAccepted,
    IngestRequest,
    IngestResult,
    JobView,
    MergeRequest,
    ReviewDecision,
    SearchRequest,
    TaskCreate,
    TaskOutcome,
)

system = APIRouter(tags=["system"])
api = APIRouter(prefix="/api/v1", dependencies=[Depends(get_principal)])


def _job_view(j: jobs_mod.Job) -> JobView:
    return JobView(id=j.id, kind=j.kind, status=j.status, attempts=j.attempts, max_attempts=j.max_attempts,  # type: ignore[arg-type]
                   error=j.error, result=j.result, created_at=j.created_at, started_at=j.started_at,
                   finished_at=j.finished_at)


# ---------------------------------------------------------------- system
@system.get("/healthz", summary="Liveness")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@system.get("/readyz", summary="Readiness: database, migrations and provider configuration")
def readyz(request: Request, response: Response, service: MemoryService = Depends(get_service)) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    try:
        service.store.fetch("SELECT 1 AS ok")
        checks["database"] = "ok"
        if service.settings.database_url:
            from ..store.migrate import migration_files

            done = {r["version"] for r in service.store.fetch("SELECT version FROM schema_migrations")}
            missing = [f.stem for f in migration_files() if f.stem not in done]
            checks["migrations"] = "ok" if not missing else f"pending: {missing}"
    except Exception as e:  # noqa: BLE001
        checks["database"] = f"error: {type(e).__name__}"
    s = service.settings
    key_for = {"gemini": s.google_api_key, "anthropic": s.anthropic_api_key, "openai": s.openai_api_key, "fake": "n/a"}
    checks["llm_provider"] = s.llm_provider if key_for.get(s.llm_provider) else f"{s.llm_provider}: API key missing"
    checks["embed_provider"] = s.embed_provider if key_for.get(s.embed_provider) else f"{s.embed_provider}: API key missing"
    ok = all(v == "ok" or v in (s.llm_provider, s.embed_provider) for v in checks.values())
    if not ok:
        response.status_code = 503
    return {"status": "ready" if ok else "not_ready", "checks": checks}


# ---------------------------------------------------------------- ingestion (async write path)
@api.post("/ingest", status_code=202, response_model=IngestAccepted, tags=["ingest"],
          summary="Queue a document for extraction, validation and consolidation")
def ingest(body: IngestRequest, request: Request, p: Principal = Depends(get_principal),
           service: MemoryService = Depends(get_service)) -> IngestAccepted:
    jid = service.enqueue_ingest(p.tenant_id, body.text, body.uri, body.source_type, body.doc_date)
    return IngestAccepted(job_id=jid, status="queued", status_url=f"/api/v1/jobs/{jid}")


@api.get("/jobs/{job_id}", response_model=JobView, tags=["ingest"])
def get_job(job_id: str, p: Principal = Depends(get_principal), service: MemoryService = Depends(get_service)) -> JobView:
    return _job_view(service.get_job(p.tenant_id, job_id))


@api.get("/jobs", response_model=list[JobView], tags=["ingest"])
def list_jobs(status: str | None = Query(default=None, pattern="^(queued|running|succeeded|failed)$"),
              limit: int = Query(default=50, ge=1, le=200), p: Principal = Depends(get_principal),
              service: MemoryService = Depends(get_service)) -> list[JobView]:
    return [_job_view(j) for j in jobs_mod.list_jobs(service.store.for_tenant(p.tenant_id), status, limit)]


# ---------------------------------------------------------------- memories
@api.post("/memories", response_model=IngestResult, tags=["memories"],
          summary="Submit a candidate memory synchronously (same extraction and validation as any document)")
def create_memory(body: IngestRequest, p: Principal = Depends(get_principal),
                  service: MemoryService = Depends(get_service)) -> IngestResult:
    rep = service.create_memory(p.tenant_id, body.text, body.source_type, body.uri, body.doc_date)
    return IngestResult(skipped_duplicate=rep.skipped_duplicate, outcomes=[asdict(o) for o in rep.outcomes])  # type: ignore[arg-type]


@api.post("/memories/search", tags=["memories"], summary="Hybrid retrieval (entities, graph, vectors) with temporal labels")
def search_memories(body: SearchRequest, p: Principal = Depends(get_principal),
                    service: MemoryService = Depends(get_service)) -> list[dict[str, Any]]:
    return service.search(p.tenant_id, body.query, body.k, body.as_of)


@api.get("/memories/{memory_id}/history", tags=["memories"], summary="All versions of this fact's subject and relation over time")
def memory_history(memory_id: str, p: Principal = Depends(get_principal),
                   service: MemoryService = Depends(get_service)) -> list[dict[str, Any]]:
    return service.memory_history(p.tenant_id, memory_id)


@api.post("/memories/{memory_id}/verify", tags=["memories"], summary="Human verification raises authority and confidence")
def verify_memory(memory_id: str, p: Principal = Depends(get_principal),
                  service: MemoryService = Depends(get_service)) -> dict[str, Any]:
    return service.verify_memory(p.tenant_id, memory_id)


@api.post("/memories/{memory_id}/invalidate", tags=["memories"], summary="Mark a memory invalid (kept for audit)")
def invalidate_memory(memory_id: str, p: Principal = Depends(get_principal),
                      service: MemoryService = Depends(get_service)) -> dict[str, Any]:
    return service.invalidate_memory(p.tenant_id, memory_id)


@api.delete("/memories/{memory_id}", status_code=204, tags=["memories"],
            summary="Invalidate and purge evidence text and vectors")
def delete_memory(memory_id: str, p: Principal = Depends(get_principal),
                  service: MemoryService = Depends(get_service)) -> Response:
    service.delete_memory(p.tenant_id, memory_id)
    return Response(status_code=204)


# ---------------------------------------------------------------- entities and graph
@api.get("/entities", tags=["entities"], summary="Look up an entity id by name or alias")
def find_entity(name: str = Query(min_length=1), p: Principal = Depends(get_principal),
                service: MemoryService = Depends(get_service)) -> dict[str, Any]:
    eid = service.find_entity(p.tenant_id, name)
    if eid is None:
        from ..services.memory_service import NotFoundError
        raise NotFoundError(f"no entity named {name!r}")
    return service.get_entity(p.tenant_id, eid)


@api.get("/entities/{entity_id}", tags=["entities"], summary="Current facts, history and conflicts of an entity")
def get_entity(entity_id: str, p: Principal = Depends(get_principal),
               service: MemoryService = Depends(get_service)) -> dict[str, Any]:
    return service.get_entity(p.tenant_id, entity_id)


@api.get("/entities/{entity_id}/graph", tags=["entities"], summary="Neighborhood graph, optionally as of a date")
def entity_graph(entity_id: str, as_of: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
                 depth: int = Query(default=2, ge=1, le=3), p: Principal = Depends(get_principal),
                 service: MemoryService = Depends(get_service)) -> dict[str, Any]:
    return service.entity_graph(p.tenant_id, entity_id, as_of, depth)


@api.post("/entities/{entity_id}/merge", tags=["entities"], summary="Merge another entity into this one (reversible)")
def merge_entity(entity_id: str, body: MergeRequest, p: Principal = Depends(get_principal),
                 service: MemoryService = Depends(get_service)) -> dict[str, Any]:
    return service.merge_entities(p.tenant_id, entity_id, body.merged_entity_id)


@api.post("/entities/merges/{merge_id}/revert", tags=["entities"], summary="Undo an entity merge")
def revert_merge(merge_id: str, p: Principal = Depends(get_principal),
                 service: MemoryService = Depends(get_service)) -> dict[str, Any]:
    return service.unmerge(p.tenant_id, merge_id)


# ---------------------------------------------------------------- conflicts and review
@api.get("/conflicts", tags=["review"], summary="Unresolved contradictions with the accepted fact they contradict")
def conflicts(p: Principal = Depends(get_principal), service: MemoryService = Depends(get_service)) -> list[dict[str, Any]]:
    return service.conflicts(p.tenant_id)


@api.get("/reviews", tags=["review"])
def reviews(status: str = Query(default="pending", pattern="^(pending|approved|rejected)$"),
            p: Principal = Depends(get_principal), service: MemoryService = Depends(get_service)) -> list[dict[str, Any]]:
    return service.reviews(p.tenant_id, status)


@api.post("/reviews/{review_id}/decision", tags=["review"])
def decide_review(review_id: str, body: ReviewDecision, p: Principal = Depends(get_principal),
                  service: MemoryService = Depends(get_service)) -> dict[str, Any]:
    return service.decide_review(p.tenant_id, review_id, body.decision == "approve", reviewer=p.tenant_id, note=body.note)


# ---------------------------------------------------------------- agent
@api.post("/agent/ask", response_model=AskResponse, tags=["agent"],
          summary="Answer a question from memory (current, historical, as-of; disputes are surfaced)")
def agent_ask(body: AskRequest, p: Principal = Depends(get_principal), service: MemoryService = Depends(get_service)) -> AskResponse:
    return AskResponse(**service.ask(p.tenant_id, body.question))


@api.post("/agent/tasks", status_code=201, tags=["agent"],
          summary="Plan a task using organizational facts and lessons; execution happens outside")
def create_task(body: TaskCreate, p: Principal = Depends(get_principal), service: MemoryService = Depends(get_service)) -> dict[str, Any]:
    return service.create_task(p.tenant_id, body.task, body.allowed_steps, body.with_facts)


@api.get("/agent/tasks/{task_id}", tags=["agent"])
def get_task(task_id: str, p: Principal = Depends(get_principal), service: MemoryService = Depends(get_service)) -> dict[str, Any]:
    return service.get_task(p.tenant_id, task_id)


@api.post("/agent/tasks/{task_id}/outcome", tags=["agent"],
          summary="Report the result: logs an episode, mines lessons, writes candidate memories through validation")
def report_outcome(task_id: str, body: TaskOutcome, p: Principal = Depends(get_principal),
                   service: MemoryService = Depends(get_service)) -> dict[str, Any]:
    return service.report_outcome(p.tenant_id, task_id, body.outcome, body.error)


@api.get("/tasks/{task_id}/memory-context", tags=["agent"], summary="The memory context a task was planned with")
def task_memory_context(task_id: str, p: Principal = Depends(get_principal),
                        service: MemoryService = Depends(get_service)) -> dict[str, Any]:
    t = service.get_task(p.tenant_id, task_id)
    return {"task_id": task_id, "memory_context": t["memory_context"], "lessons": t["lessons"]}
