"""SQLite backend: zero-infrastructure store for offline development, unit tests and the eval harness.

Time model (design 15): all timestamps are ISO dates 'YYYY-MM-DD' (valid time) or ISO datetimes (jobs).
- valid time: valid_from (inclusive), valid_until (exclusive, NULL = still true)
- system time: observed_at / created_at .. recorded_until (NULL = still believed)
A memory row is never edited in place to change what it asserts. Closing a fact inserts a new version with
valid_until set and ends the old version's system time (recorded_until).
"""
from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from .base import BaseStore, Row
from .schema import sqlite_schema
from .vectors import top_k


class SqliteStore(BaseStore):
    def __init__(self, path: str = ":memory:", tenant_id: str = "default"):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.lock = threading.RLock()  # one shared connection; serialize access across threads
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(sqlite_schema())
        self._state = {"depth": 0}  # transaction nesting, shared by tenant clones
        self.tenant = tenant_id

    def _commit(self) -> None:
        if self._state["depth"] == 0:
            self.db.commit()

    # ---- primitives ---------------------------------------------------------------------------------------------
    def insert(self, table: str, row: dict[str, Any], ignore_conflicts: bool = False) -> bool:
        cols = ",".join(row)
        qs = ",".join("?" * len(row))
        verb = "INSERT OR IGNORE" if ignore_conflicts else "INSERT"
        with self.lock:
            cur = self.db.execute(f"{verb} INTO {table} ({cols}) VALUES ({qs})", list(row.values()))
            self._commit()
            return cur.rowcount > 0

    def update(self, table: str, id_: str, fields: dict[str, Any]) -> None:
        sets = ",".join(f"{k}=?" for k in fields)
        with self.lock:
            self.db.execute(f"UPDATE {table} SET {sets} WHERE id=? AND tenant_id=?", [*fields.values(), id_, self.tenant])
            self._commit()

    def fetch(self, sql: str, params: tuple | list = ()) -> list[Row]:
        with self.lock:
            return self.db.execute(sql, params).fetchall()

    def execute(self, sql: str, params: tuple | list = ()) -> int:
        with self.lock:
            cur = self.db.execute(sql, params)
            self._commit()
            return cur.rowcount

    def execute_returning(self, sql: str, params: tuple | list = ()) -> list[Row]:
        with self.lock:
            rows = self.db.execute(sql, params).fetchall()
            self._commit()
            return rows

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self.lock:
            self._state["depth"] += 1
            try:
                yield
            except BaseException:
                self._state["depth"] -= 1
                if self._state["depth"] == 0:
                    self.db.rollback()
                raise
            else:
                self._state["depth"] -= 1
                if self._state["depth"] == 0:
                    self.db.commit()

    def put_vector(self, kind: str, ref_id: str, vec: np.ndarray) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)",
                            (kind, ref_id, self.tenant, vec.astype(np.float32).tobytes()))
            self._commit()

    def search_vectors(self, kind: str, query: np.ndarray, k: int = 10,
                       ref_ids: set[str] | None = None) -> list[tuple[str, float]]:
        rows = self.fetch("SELECT ref_id, vec FROM vectors WHERE kind=? AND tenant_id=?", (kind, self.tenant))
        if ref_ids is not None:
            rows = [r for r in rows if r["ref_id"] in ref_ids]
        if not rows:
            return []
        mat = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
        return [(rows[i]["ref_id"], s) for i, s in top_k(query.astype(np.float32), mat, k)]

    def delete_vectors(self, kind: str, ref_ids: list[str]) -> None:
        for rid in ref_ids:
            self.execute("DELETE FROM vectors WHERE kind=? AND ref_id=? AND tenant_id=?", (kind, rid, self.tenant))

    def close(self) -> None:
        with self.lock:
            self.db.close()


MemoryStore = SqliteStore  # backwards-compatible name used by the PoC code, tests and eval harness
