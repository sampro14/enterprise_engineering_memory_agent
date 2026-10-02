"""Pick the storage backend from settings: Postgres when DATABASE_URL is set, otherwise a SQLite file."""
from __future__ import annotations

import os
import threading

from ..config import Settings, get_settings
from .base import BaseStore
from .postgres import PostgresStore
from .sqlite import SqliteStore

_roots: dict[tuple, BaseStore] = {}
_lock = threading.Lock()


def create_store(settings: Settings | None = None, tenant_id: str | None = None, *, schema: str | None = None) -> BaseStore:
    """Return a store bound to `tenant_id`. Postgres pools are created once per (url, schema) and shared."""
    s = settings or get_settings()
    tenant = tenant_id or s.tenant_id
    if not s.database_url:
        key = ("sqlite", s.db_path)
        with _lock:
            if key not in _roots:
                _roots[key] = SqliteStore(s.db_path, tenant)
            return _roots[key].for_tenant(tenant)
    key = ("pg", s.database_url, schema, s.embed_dim)
    with _lock:
        if key not in _roots:
            _roots[key] = PostgresStore(s.database_url, tenant, schema=schema, embed_dim=s.embed_dim,
                                        exact=os.getenv("PGVECTOR_EXACT", "") == "1")
        return _roots[key].for_tenant(tenant)


def close_all_stores() -> None:
    with _lock:
        for st in _roots.values():
            st.close()
        _roots.clear()
