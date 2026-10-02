"""Store-backed job queue for the async write path (design 11: extraction/validation/consolidation off the request path).

- `enqueue` inserts a queued job for the store's tenant.
- `claim` atomically takes the oldest runnable job of ANY tenant (FOR UPDATE SKIP LOCKED on Postgres, so concurrent
  workers never take the same job). A job stuck in `running` past the visibility timeout is reclaimed; once its attempts
  are exhausted it is failed instead.
- `fail` requeues with exponential backoff until max_attempts, then marks the job failed.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from typing import Any

from .models import new_id
from .store.base import BaseStore, Row
from .timeutil import iso, utcnow, utcnow_iso

QUEUED, RUNNING, SUCCEEDED, FAILED = "queued", "running", "succeeded", "failed"


@dataclass
class Job:
    id: str
    tenant_id: str
    kind: str
    payload: dict[str, Any]
    status: str
    attempts: int
    max_attempts: int
    error: str | None
    result: dict[str, Any] | None
    created_at: str
    started_at: str | None
    finished_at: str | None

    @classmethod
    def from_row(cls, r: Row) -> Job:
        return cls(r["id"], r["tenant_id"], r["kind"], json.loads(r["payload"]), r["status"], r["attempts"],
                   r["max_attempts"], r["error"], json.loads(r["result"]) if r["result"] else None,
                   r["created_at"], r["started_at"], r["finished_at"])


def enqueue(store: BaseStore, kind: str, payload: dict[str, Any], max_attempts: int = 3,
            now: dt.datetime | None = None) -> str:
    jid = new_id("job")
    now = iso(now) if now else utcnow_iso()
    store.insert("jobs", {"id": jid, "tenant_id": store.tenant, "kind": kind, "payload": json.dumps(payload),
                          "status": QUEUED, "attempts": 0, "max_attempts": max_attempts, "run_after": now,
                          "created_at": now})
    return jid


def get_job(store: BaseStore, job_id: str) -> Job | None:
    rows = store.fetch("SELECT * FROM jobs WHERE id=? AND tenant_id=?", (job_id, store.tenant))
    return Job.from_row(rows[0]) if rows else None


def list_jobs(store: BaseStore, status: str | None = None, limit: int = 50) -> list[Job]:
    q, p = "SELECT * FROM jobs WHERE tenant_id=?", [store.tenant]
    if status:
        q, p = q + " AND status=?", [*p, status]
    return [Job.from_row(r) for r in store.fetch(q + " ORDER BY created_at DESC LIMIT ?", [*p, limit])]


def claim(store: BaseStore, worker_id: str, visibility_timeout: float = 300.0, now: dt.datetime | None = None) -> Job | None:
    t = now or utcnow()
    now_s, stale_s = iso(t), iso(t - dt.timedelta(seconds=visibility_timeout))
    # Jobs that crashed a worker too many times are failed rather than retried forever.
    store.execute(
        "UPDATE jobs SET status='failed', error='visibility timeout exceeded after max attempts', finished_at=? "
        "WHERE status='running' AND locked_at < ? AND attempts >= max_attempts", (now_s, stale_s))
    skip = " FOR UPDATE SKIP LOCKED" if store.supports_skip_locked else ""
    rows = store.execute_returning(
        "UPDATE jobs SET status='running', locked_at=?, locked_by=?, started_at=COALESCE(started_at, ?), "
        "attempts=attempts+1 WHERE id = (SELECT id FROM jobs WHERE "
        "(status='queued' AND run_after <= ?) OR (status='running' AND locked_at < ? AND attempts < max_attempts) "
        f"ORDER BY created_at, id LIMIT 1{skip}) RETURNING *",
        (now_s, worker_id, now_s, now_s, stale_s))
    return Job.from_row(rows[0]) if rows else None


def complete(store: BaseStore, job_id: str, result: dict[str, Any] | None = None) -> None:
    store.execute("UPDATE jobs SET status='succeeded', result=?, error=NULL, finished_at=?, locked_at=NULL WHERE id=?",
                  (json.dumps(result or {}), utcnow_iso(), job_id))


def fail(store: BaseStore, job: Job, error: str, base_backoff: float = 2.0, now: dt.datetime | None = None) -> str:
    """Requeue with exponential backoff, or mark failed when attempts are exhausted. Returns the new status."""
    t = now or utcnow()
    if job.attempts >= job.max_attempts:
        store.execute("UPDATE jobs SET status='failed', error=?, finished_at=?, locked_at=NULL WHERE id=?",
                      (error[:2000], iso(t), job.id))
        return FAILED
    run_after = iso(t + dt.timedelta(seconds=base_backoff * 2 ** (job.attempts - 1)))
    store.execute("UPDATE jobs SET status='queued', error=?, run_after=?, locked_at=NULL WHERE id=?",
                  (error[:2000], run_after, job.id))
    return QUEUED
