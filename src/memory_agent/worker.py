"""Write-path worker: claims jobs from the queue and runs them (design 11). Run several for throughput; the queue
guarantees a job is held by one worker at a time, and the per-tenant lock keeps one tenant's graph updates ordered."""
from __future__ import annotations

import logging
import signal
import socket
import threading
import time
import uuid

from . import jobs
from .services.memory_service import MemoryService

log = logging.getLogger("memory_agent.worker")


class Worker:
    def __init__(self, service: MemoryService, worker_id: str | None = None):
        self.service = service
        self.worker_id = worker_id or f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
        self.done = self.failed = 0

    def run_once(self) -> bool:
        """Handle at most one job. Returns True if a job was claimed."""
        s = self.service.settings
        job = jobs.claim(self.service.store, self.worker_id, s.job_visibility_timeout)
        if job is None:
            return False
        t0 = time.monotonic()
        try:
            result = self.service.process_job(job)
        except Exception as e:
            status = jobs.fail(self.service.store, job, f"{type(e).__name__}: {e}")
            self.failed += 1
            log.exception("job %s failed (attempt %d/%d) -> %s", job.id, job.attempts, job.max_attempts, status)
        else:
            jobs.complete(self.service.store, job.id, result)
            self.done += 1
            log.info("job %s done in %.2fs (%s facts, tenant %s)", job.id, time.monotonic() - t0,
                     result.get("facts"), job.tenant_id)
        return True

    def run_forever(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        log.info("worker %s started", self.worker_id)
        while not stop.is_set():
            try:
                handled = self.run_once()
            except Exception:
                log.exception("worker loop error")
                handled = False
            if not handled:
                stop.wait(self.service.settings.worker_poll_seconds)
        log.info("worker %s stopped (done=%d failed=%d)", self.worker_id, self.done, self.failed)


def run_worker(service: MemoryService) -> None:
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())  # finish the current job, then exit
    Worker(service).run_forever(stop)
