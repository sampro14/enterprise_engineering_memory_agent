"""Request dependencies: the service singleton and API-key authentication (tenant comes from the key, never the body)."""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .. import tenants
from ..services.memory_service import MemoryService

bearer = HTTPBearer(auto_error=False, description="API key created with `memory-agent tenant create`")


@dataclass(frozen=True)
class Principal:
    tenant_id: str


def get_service(request: Request) -> MemoryService:
    return request.app.state.service


def get_principal(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    service: MemoryService = Depends(get_service),
) -> Principal:
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(401, "missing bearer API key", headers={"WWW-Authenticate": "Bearer"})
    tenant = tenants.authenticate(service.store, creds.credentials)
    if tenant is None:
        raise HTTPException(401, "invalid or revoked API key", headers={"WWW-Authenticate": "Bearer"})
    request.state.tenant = tenant
    return Principal(tenant)
