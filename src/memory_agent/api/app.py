"""FastAPI application factory."""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .. import __version__
from ..config import Settings, get_settings
from ..llm.base import LLMError, ProviderConfigError
from ..logging_setup import setup_logging
from ..services.memory_service import ConflictError, MemoryService, NotFoundError
from .routers import api, system

log = logging.getLogger("memory_agent.api")


def _err(request: Request, status: int, error: str, detail: str) -> JSONResponse:
    rid = getattr(request.state, "request_id", None)
    return JSONResponse(status_code=status, content={"error": error, "detail": detail, "request_id": rid},
                        headers={"X-Request-ID": rid} if rid else None)


def create_app(service: MemoryService | None = None, settings: Settings | None = None) -> FastAPI:
    """`service` can be injected (tests); otherwise it is built from settings at startup (and JSON logging is set up)."""
    s = settings or (service.settings if service else get_settings())
    if service is None:
        setup_logging(s.log_level)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            app.state.service = MemoryService.from_settings(s)
        log.info("api started (llm=%s embed=%s db=%s)", s.llm_provider, s.embed_provider,
                 "postgres" if s.database_url else "sqlite")
        yield

    app = FastAPI(title="Enterprise Engineering Memory Agent", version=__version__, lifespan=lifespan,
                  description="Persistent, temporal, confidence-aware organizational memory for AI agents. "
                              "Authenticate with `Authorization: Bearer <api key>`.")
    if service is not None:
        app.state.service = service

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        t0 = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error", extra={"request_id": rid, "path": request.url.path})
            return _err(request, 500, "internal_error", "unexpected server error")
        response.headers["X-Request-ID"] = rid
        log.info("request", extra={"request_id": rid, "method": request.method, "path": request.url.path,
                                   "status": response.status_code, "duration_ms": round((time.monotonic() - t0) * 1000, 1),
                                   "tenant": getattr(request.state, "tenant", None)})
        return response

    @app.exception_handler(NotFoundError)
    async def _nf(request: Request, exc: NotFoundError):
        return _err(request, 404, "not_found", str(exc))

    @app.exception_handler(ConflictError)
    async def _cf(request: Request, exc: ConflictError):
        return _err(request, 409, "conflict", str(exc))

    @app.exception_handler(ProviderConfigError)
    async def _pc(request: Request, exc: ProviderConfigError):
        return _err(request, 503, "provider_not_configured", str(exc))

    @app.exception_handler(LLMError)
    async def _llm(request: Request, exc: LLMError):
        return _err(request, 502, "provider_error", f"{type(exc).__name__}: {exc}")

    @app.exception_handler(HTTPException)
    async def _http(request: Request, exc: HTTPException):
        rid = getattr(request.state, "request_id", None)
        headers = dict(exc.headers or {})
        if rid:
            headers["X-Request-ID"] = rid
        return JSONResponse(status_code=exc.status_code, headers=headers,
                            content={"error": "http_error", "detail": exc.detail, "request_id": rid})

    @app.exception_handler(RequestValidationError)
    async def _val(request: Request, exc: RequestValidationError):
        return _err(request, 422, "validation_error", "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()))

    app.include_router(system)
    app.include_router(api)
    return app
