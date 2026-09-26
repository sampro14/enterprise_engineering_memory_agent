"""SQLite store with bitemporal memory rows (design 15, 25).

Time model (PoC): all timestamps are ISO dates 'YYYY-MM-DD'.
- valid time: valid_from (inclusive), valid_until (exclusive, NULL = still true)
- system time: observed_at / created_at .. recorded_until (NULL = still believed)
A memory row is never edited in place to change what it asserts. Closing a fact inserts a new
version with valid_until set and ends the old version's system time (recorded_until).
"""
from __future__ import annotations

import os
import sqlite3
import threading
from typing import Any

import numpy as np

from ..models import Entity, Memory, new_id
from .vectors import top_k

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, entity_type TEXT NOT NULL, name TEXT NOT NULL,
  description TEXT, created_at TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS entity_aliases (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, entity_id TEXT NOT NULL, alias TEXT NOT NULL,
  alias_key TEXT NOT NULL, source TEXT, confidence REAL, created_at TEXT,
  UNIQUE (tenant_id, alias_key)
);
CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, memory_type TEXT, subject_id TEXT NOT NULL,
  relation TEXT NOT NULL, object_id TEXT, object_value TEXT, content TEXT,
  confidence REAL, importance REAL, source_authority REAL,
  valid_from TEXT, valid_until TEXT, observed_at TEXT, created_at TEXT, recorded_until TEXT,
  supersedes_id TEXT, last_verified TEXT, last_accessed TEXT, status TEXT, access_policy TEXT
);
CREATE INDEX IF NOT EXISTS ix_mem_subject ON memories (tenant_id, subject_id, relation);
CREATE VIEW IF NOT EXISTS relations AS
  SELECT id, tenant_id, subject_id AS source_entity_id, relation, object_id AS target_entity_id,
         confidence, valid_from, valid_until, recorded_until, id AS source_memory_id
  FROM memories WHERE object_id IS NOT NULL;
CREATE TABLE IF NOT EXISTS evidence (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, memory_id TEXT NOT NULL, source_type TEXT,
  source_uri TEXT, excerpt TEXT, source_authority REAL, observed_at TEXT, access_policy TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, memory_id TEXT, candidate_json TEXT,
  reason TEXT, status TEXT DEFAULT 'pending', reviewer TEXT, note TEXT,
  created_at TEXT, decided_at TEXT
);
CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, uri TEXT, content_hash TEXT NOT NULL,
  source_type TEXT, doc_date TEXT, ingested_at TEXT, UNIQUE (tenant_id, content_hash)
);
CREATE TABLE IF NOT EXISTS episodes (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, task_id TEXT, summary TEXT, outcome TEXT,
  timestamp TEXT, signature TEXT, conditions TEXT, counter_of TEXT
);
CREATE TABLE IF NOT EXISTS episode_entities (episode_id TEXT, entity_id TEXT);
CREATE TABLE IF NOT EXISTS experiences (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, status TEXT, pattern TEXT, lesson TEXT,
  evidence_count INTEGER, confidence REAL, conditions TEXT, recommended_action TEXT,
  created_at TEXT, last_verified TEXT
);
CREATE TABLE IF NOT EXISTS experience_evidence (experience_id TEXT, episode_id TEXT);
CREATE TABLE IF NOT EXISTS vectors (
  kind TEXT NOT NULL, ref_id TEXT NOT NULL, tenant_id TEXT NOT NULL, vec BLOB NOT NULL,
  PRIMARY KEY (kind, ref_id)
);
"""


class MemoryStore:
    def __init__(self, path: str = ":memory:", tenant_id: str = "default"):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.lock = threading.RLock()  # one shared connection; serialize access across threads
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.tenant = tenant_id

    # ---- generic helpers ---------------------------------------------------------------
    def insert(self, table: str, row: dict[str, Any]) -> None:
        cols = ",".join(row)
        qs = ",".join("?" * len(row))
        with self.lock:
            self.db.execute(f"INSERT INTO {table} ({cols}) VALUES ({qs})", list(row.values()))
            self.db.commit()

    def update(self, table: str, id_: str, fields: dict[str, Any]) -> None:
        sets = ",".join(f"{k}=?" for k in fields)
        with self.lock:
            self.db.execute(
                f"UPDATE {table} SET {sets} WHERE id=? AND tenant_id=?", [*fields.values(), id_, self.tenant]
            )
            self.db.commit()

    def fetch(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, params).fetchall()

    # ---- entities ----------------------------------------------------------------------
    def add_entity(self, entity_type: str, name: str, now: str, description: str | None = None) -> Entity:
        e = Entity(id=new_id("ent"), tenant_id=self.tenant, entity_type=entity_type, name=name, description=description)
        self.insert(
            "entities",
            {**e.model_dump(), "created_at": now, "updated_at": now},
        )
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
        try:
            self.insert(
                "entity_aliases",
                {"id": new_id("als"), "tenant_id": self.tenant, "entity_id": entity_id, "alias": alias,
                 "alias_key": alias_key, "source": source, "confidence": confidence, "created_at": now},
            )
            return True
        except sqlite3.IntegrityError:
            return False

    def entity_by_alias_key(self, alias_key: str) -> Entity | None:
        r = self.fetch(
            "SELECT entity_id FROM entity_aliases WHERE tenant_id=? AND alias_key=?", (self.tenant, alias_key)
        )
        return self.get_entity(r[0]["entity_id"]) if r else None

    def aliases_of(self, entity_id: str) -> list[str]:
        return [r["alias"] for r in self.fetch(
            "SELECT alias FROM entity_aliases WHERE tenant_id=? AND entity_id=?", (self.tenant, entity_id))]

    # ---- memories ----------------------------------------------------------------------
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
        self.update("memories", memory.id, {"recorded_until": now, "status": status})
        closed = memory.model_copy(update={
            "id": new_id("mem"), "valid_until": valid_until, "observed_at": now, "created_at": now,
            "recorded_until": None, "supersedes_id": memory.id, "status": "active",
        })
        self.insert_memory(closed)
        # keep evidence attached to the new version too
        for ev in self.fetch("SELECT * FROM evidence WHERE memory_id=?", (memory.id,)):
            d = dict(ev)
            d.update(id=new_id("evd"), memory_id=closed.id)
            self.insert("evidence", d)
        return closed

    def touch(self, memory_ids: list[str], now: str) -> None:
        for mid in memory_ids:
            self.update("memories", mid, {"last_accessed": now})

    # ---- evidence / sources / reviews --------------------------------------------------
    def add_evidence(self, memory_id: str, source_type: str, uri: str, excerpt: str, authority: float, now: str) -> None:
        self.insert("evidence", {"id": new_id("evd"), "tenant_id": self.tenant, "memory_id": memory_id,
                                 "source_type": source_type, "source_uri": uri, "excerpt": excerpt,
                                 "source_authority": authority, "observed_at": now, "access_policy": None})

    def evidence_of(self, memory_id: str) -> list[sqlite3.Row]:
        return self.fetch("SELECT * FROM evidence WHERE memory_id=? AND tenant_id=?", (memory_id, self.tenant))

    def source_seen(self, content_hash: str) -> bool:
        return bool(self.fetch("SELECT 1 FROM sources WHERE tenant_id=? AND content_hash=?", (self.tenant, content_hash)))

    def record_source(self, uri: str, content_hash: str, source_type: str, doc_date: str, now: str) -> None:
        self.insert("sources", {"id": new_id("src"), "tenant_id": self.tenant, "uri": uri, "content_hash": content_hash,
                                "source_type": source_type, "doc_date": doc_date, "ingested_at": now})

    def add_review(self, reason: str, candidate_json: str, now: str, memory_id: str | None = None) -> str:
        rid = new_id("rev")
        self.insert("reviews", {"id": rid, "tenant_id": self.tenant, "memory_id": memory_id,
                                "candidate_json": candidate_json, "reason": reason, "status": "pending",
                                "created_at": now})
        return rid

    def pending_reviews(self) -> list[sqlite3.Row]:
        return self.fetch("SELECT * FROM reviews WHERE tenant_id=? AND status='pending' ORDER BY created_at", (self.tenant,))

    # ---- vectors -----------------------------------------------------------------------
    def put_vector(self, kind: str, ref_id: str, vec: np.ndarray) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)",
                            (kind, ref_id, self.tenant, vec.astype(np.float32).tobytes()))
            self.db.commit()

    def search_vectors(self, kind: str, query: np.ndarray, k: int = 10,
                       ref_ids: set[str] | None = None) -> list[tuple[str, float]]:
        rows = self.fetch("SELECT ref_id, vec FROM vectors WHERE kind=? AND tenant_id=?", (kind, self.tenant))
        if ref_ids is not None:
            rows = [r for r in rows if r["ref_id"] in ref_ids]
        if not rows:
            return []
        mat = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
        return [(rows[i]["ref_id"], s) for i, s in top_k(query.astype(np.float32), mat, k)]
