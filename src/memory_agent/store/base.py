"""Backend-independent store logic. Backends implement a few primitives (insert/update/fetch/execute/delete,
transactions, vectors) and inherit every domain operation from here, so behavior cannot diverge between SQLite and
Postgres. All SQL in this package and its callers uses `?` placeholders; the Postgres backend translates them."""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any

import numpy as np

from ..models import Entity, Memory, new_id

Row = Mapping[str, Any]


class BaseStore(ABC):
    """A store bound to one tenant. `for_tenant` returns a cheap clone sharing the same connection/pool."""

    tenant: str
    supports_skip_locked: bool = False

    # ---- primitives (backend specific) --------------------------------------------------------------------------
    @abstractmethod
    def insert(self, table: str, row: dict[str, Any], ignore_conflicts: bool = False) -> bool:
        """Insert a row. With ignore_conflicts, a unique/PK conflict is skipped and False is returned."""

    @abstractmethod
    def update(self, table: str, id_: str, fields: dict[str, Any]) -> None:
        """Update by primary key, scoped to this tenant."""

    @abstractmethod
    def fetch(self, sql: str, params: tuple | list = ()) -> list[Row]:
        """Run a read query (raw: NOT tenant-scoped automatically; callers filter by tenant_id)."""

    @abstractmethod
    def execute(self, sql: str, params: tuple | list = ()) -> int:
        """Run a write statement, return affected rows."""

    @abstractmethod
    def execute_returning(self, sql: str, params: tuple | list = ()) -> list[Row]:
        """Run a write statement with a RETURNING clause and return the rows (committed)."""

    def lock_tenant(self) -> None:
        """Serialize concurrent writers for this tenant until the surrounding transaction ends (no-op by default)."""

    @abstractmethod
    def transaction(self) -> AbstractContextManager[None]:
        """Group several writes atomically (re-entrant)."""

    @abstractmethod
    def put_vector(self, kind: str, ref_id: str, vec: np.ndarray) -> None: ...

    @abstractmethod
    def search_vectors(self, kind: str, query: np.ndarray, k: int = 10,
                       ref_ids: set[str] | None = None) -> list[tuple[str, float]]:
        """Top-k by cosine similarity for this tenant: [(ref_id, similarity)] best first."""

    @abstractmethod
    def delete_vectors(self, kind: str, ref_ids: list[str]) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def for_tenant(self, tenant_id: str) -> BaseStore:
        clone = copy.copy(self)
        clone.tenant = tenant_id
        return clone

    # ---- entities ------------------------------------------------------------------------------------------------
    def add_entity(self, entity_type: str, name: str, now: str, description: str | None = None) -> Entity:
        e = Entity(id=new_id("ent"), tenant_id=self.tenant, entity_type=entity_type, name=name, description=description)
        self.insert("entities", {**e.model_dump(), "created_at": now, "updated_at": now})
        return e

    def get_entity(self, entity_id: str) -> Entity | None:
        r = self.fetch("SELECT * FROM entities WHERE id=? AND tenant_id=?", (entity_id, self.tenant))
        return Entity(**{k: r[0][k] for k in Entity.model_fields}) if r else None

    def list_entities(self, entity_type: str | None = None) -> list[Entity]:
        q, p = "SELECT * FROM entities WHERE tenant_id=?", [self.tenant]
        if entity_type:
            q, p = q + " AND entity_type=?", [*p, entity_type]
        return [Entity(**{k: r[k] for k in Entity.model_fields}) for r in self.fetch(q, p)]

    def add_alias(self, entity_id: str, alias: str, alias_key: str, source: str, confidence: float, now: str) -> bool:
        return self.insert(
            "entity_aliases",
            {"id": new_id("als"), "tenant_id": self.tenant, "entity_id": entity_id, "alias": alias,
             "alias_key": alias_key, "source": source, "confidence": confidence, "created_at": now},
            ignore_conflicts=True,
        )

    def entity_by_alias_key(self, alias_key: str) -> Entity | None:
        r = self.fetch("SELECT entity_id FROM entity_aliases WHERE tenant_id=? AND alias_key=?", (self.tenant, alias_key))
        return self.get_entity(r[0]["entity_id"]) if r else None

    def aliases_of(self, entity_id: str) -> list[str]:
        return [r["alias"] for r in self.fetch(
            "SELECT alias FROM entity_aliases WHERE tenant_id=? AND entity_id=?", (self.tenant, entity_id))]

    # ---- memories ------------------------------------------------------------------------------------------------
    def insert_memory(self, m: Memory) -> Memory:
        self.insert("memories", m.model_dump())
        return m

    def get_memory(self, memory_id: str) -> Memory | None:
        r = self.fetch("SELECT * FROM memories WHERE id=? AND tenant_id=?", (memory_id, self.tenant))
        return Memory(**dict(r[0])) if r else None

    def memories(self, where: str = "1=1", params: tuple | list = ()) -> list[Memory]:
        rows = self.fetch(f"SELECT * FROM memories WHERE tenant_id=? AND ({where})", [self.tenant, *params])
        return [Memory(**dict(r)) for r in rows]

    def close_validity(self, memory: Memory, valid_until: str, now: str, status: str = "superseded") -> Memory:
        """Bitemporal close: end the old version's system time and insert a version with valid_until set."""
        with self.transaction():
            self.update("memories", memory.id, {"recorded_until": now, "status": status})
            closed = memory.model_copy(update={
                "id": new_id("mem"), "valid_until": valid_until, "observed_at": now, "created_at": now,
                "recorded_until": None, "supersedes_id": memory.id, "status": "active",
            })
            self.insert_memory(closed)
            for ev in self.fetch("SELECT * FROM evidence WHERE memory_id=?", (memory.id,)):  # keep evidence attached
                d = dict(ev)
                d.update(id=new_id("evd"), memory_id=closed.id)
                self.insert("evidence", d)
        return closed

    def touch(self, memory_ids: list[str], now: str) -> None:
        for mid in memory_ids:
            self.update("memories", mid, {"last_accessed": now})

    def purge_memory(self, memory_id: str) -> None:
        """Hard-delete a memory's evidence text and vectors and invalidate it (design 22: deletion propagation, basic)."""
        with self.transaction():
            self.execute("DELETE FROM evidence WHERE memory_id=? AND tenant_id=?", (memory_id, self.tenant))
            self.update("memories", memory_id, {"status": "invalidated", "content": "", "object_value": None})
        self.delete_vectors("memory", [memory_id])

    # ---- evidence / sources / reviews ---------------------------------------------------------------------------
    def add_evidence(self, memory_id: str, source_type: str, uri: str, excerpt: str, authority: float, now: str) -> None:
        self.insert("evidence", {"id": new_id("evd"), "tenant_id": self.tenant, "memory_id": memory_id,
                                 "source_type": source_type, "source_uri": uri, "excerpt": excerpt,
                                 "source_authority": authority, "observed_at": now, "access_policy": None})

    def evidence_of(self, memory_id: str) -> list[Row]:
        return self.fetch("SELECT * FROM evidence WHERE memory_id=? AND tenant_id=?", (memory_id, self.tenant))

    def source_seen(self, content_hash: str) -> bool:
        return bool(self.fetch("SELECT 1 AS x FROM sources WHERE tenant_id=? AND content_hash=?", (self.tenant, content_hash)))

    def record_source(self, uri: str, content_hash: str, source_type: str, doc_date: str, now: str) -> bool:
        return self.insert("sources", {"id": new_id("src"), "tenant_id": self.tenant, "uri": uri,
                                       "content_hash": content_hash, "source_type": source_type,
                                       "doc_date": doc_date, "ingested_at": now}, ignore_conflicts=True)

    def add_review(self, reason: str, candidate_json: str, now: str, memory_id: str | None = None) -> str:
        rid = new_id("rev")
        self.insert("reviews", {"id": rid, "tenant_id": self.tenant, "memory_id": memory_id,
                                "candidate_json": candidate_json, "reason": reason, "status": "pending",
                                "created_at": now})
        return rid

    def pending_reviews(self) -> list[Row]:
        return self.fetch("SELECT * FROM reviews WHERE tenant_id=? AND status='pending' ORDER BY created_at, id",
                          (self.tenant,))

    def get_review(self, review_id: str) -> Row | None:
        r = self.fetch("SELECT * FROM reviews WHERE id=? AND tenant_id=?", (review_id, self.tenant))
        return r[0] if r else None

    # ---- episodes / experiences (link tables) --------------------------------------------------------------------
    def link_episode_entity(self, episode_id: str, entity_id: str) -> None:
        self.insert("episode_entities", {"episode_id": episode_id, "entity_id": entity_id})

    def link_experience_evidence(self, experience_id: str, episode_id: str) -> None:
        self.insert("experience_evidence", {"experience_id": experience_id, "episode_id": episode_id})
