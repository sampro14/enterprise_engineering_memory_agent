"""Tenants and API keys. Keys are high-entropy random secrets, so a plain SHA-256 digest is sufficient to store them;
the plaintext is shown once at creation and never persisted."""
from __future__ import annotations

import hashlib
import secrets

from .models import new_id
from .store.base import BaseStore
from .timeutil import utcnow_iso

KEY_PREFIX = "mem_"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_tenant(store: BaseStore, name: str, tenant_id: str | None = None) -> str:
    tid = tenant_id or new_id("ten")
    store.insert("tenants", {"id": tid, "name": name, "created_at": utcnow_iso()})
    return tid


def tenant_exists(store: BaseStore, tenant_id: str) -> bool:
    return bool(store.fetch("SELECT 1 AS x FROM tenants WHERE id=?", (tenant_id,)))


def create_api_key(store: BaseStore, tenant_id: str, name: str = "default") -> str:
    """Returns the plaintext key (only chance to see it)."""
    if not tenant_exists(store, tenant_id):
        raise ValueError(f"unknown tenant {tenant_id}")
    key = KEY_PREFIX + secrets.token_urlsafe(32)
    store.insert("api_keys", {"id": new_id("key"), "tenant_id": tenant_id, "key_hash": hash_key(key),
                              "key_prefix": key[:10], "name": name, "created_at": utcnow_iso()})
    return key


def authenticate(store: BaseStore, key: str) -> str | None:
    """Return the tenant id for a valid, non-revoked key, else None."""
    if not key.startswith(KEY_PREFIX):
        return None
    rows = store.fetch("SELECT tenant_id FROM api_keys WHERE key_hash=? AND revoked_at IS NULL", (hash_key(key),))
    return rows[0]["tenant_id"] if rows else None


def revoke_api_key(store: BaseStore, key: str) -> bool:
    return store.execute("UPDATE api_keys SET revoked_at=? WHERE key_hash=? AND revoked_at IS NULL",
                         (utcnow_iso(), hash_key(key))) > 0
