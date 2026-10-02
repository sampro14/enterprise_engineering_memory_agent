"""Postgres + pgvector backend (psycopg 3, connection pool).

- Vectors live in `halfvec(EMBED_DIM)` with an HNSW cosine index (halfvec supports up to 4000 dimensions).
- SQL written with `?` placeholders (shared with SQLite) is translated to psycopg's `%s`.
- Each primitive runs in its own transaction unless inside `transaction()`, which pins one connection per thread.
"""
from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .base import BaseStore, Row


def _q(sql: str) -> str:
    return sql.replace("%", "%%").replace("?", "%s")


def _vec_literal(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v.astype(np.float32).tolist()) + "]"


class _Backend:
    """State shared by all tenant clones of one PostgresStore."""

    def __init__(self, url: str, schema: str | None, embed_dim: int, pool_size: int, exact: bool):
        self.embed_dim = embed_dim
        self.local = threading.local()

        def configure(conn: psycopg.Connection) -> None:
            if schema:
                conn.execute(f'SET search_path TO "{schema}", public')
            try:
                conn.execute("SET hnsw.iterative_scan = strict_order")  # filtered ANN queries must still return k rows
            except psycopg.Error:
                conn.rollback()
            if exact:
                conn.execute("SET enable_indexscan = off")  # exact scans for evaluations that must match SQLite
            conn.commit()

        self.pool = ConnectionPool(url, min_size=1, max_size=pool_size, open=True, configure=configure,
                                   kwargs={"row_factory": dict_row}, timeout=30)

    @contextmanager
    def conn(self) -> Iterator[psycopg.Connection]:
        pinned = getattr(self.local, "conn", None)
        if pinned is not None:
            yield pinned
            return
        with self.pool.connection() as c:  # commits on success, rolls back on error
            yield c

    @contextmanager
    def pinned(self) -> Iterator[None]:
        if getattr(self.local, "conn", None) is not None:  # nested: join the outer transaction
            yield
            return
        with self.pool.connection() as c:
            self.local.conn = c
            try:
                yield
            finally:
                self.local.conn = None


class PostgresStore(BaseStore):
    supports_skip_locked = True

    def __init__(self, database_url: str, tenant_id: str = "default", *, schema: str | None = None,
                 embed_dim: int = 3072, pool_size: int = 10, exact: bool = False):
        self._b = _Backend(database_url, schema, embed_dim, pool_size, exact)
        self.tenant = tenant_id

    # ---- primitives ---------------------------------------------------------------------------------------------
    def insert(self, table: str, row: dict[str, Any], ignore_conflicts: bool = False) -> bool:
        cols = ",".join(row)
        qs = ",".join(["%s"] * len(row))
        tail = " ON CONFLICT DO NOTHING" if ignore_conflicts else ""
        with self._b.conn() as c:
            cur = c.execute(f"INSERT INTO {table} ({cols}) VALUES ({qs}){tail}", list(row.values()))
            return cur.rowcount > 0

    def update(self, table: str, id_: str, fields: dict[str, Any]) -> None:
        sets = ",".join(f"{k}=%s" for k in fields)
        with self._b.conn() as c:
            c.execute(f"UPDATE {table} SET {sets} WHERE id=%s AND tenant_id=%s", [*fields.values(), id_, self.tenant])

    def fetch(self, sql: str, params: tuple | list = ()) -> list[Row]:
        with self._b.conn() as c:
            return c.execute(_q(sql), list(params)).fetchall()

    def execute(self, sql: str, params: tuple | list = ()) -> int:
        with self._b.conn() as c:
            return c.execute(_q(sql), list(params)).rowcount

    def execute_returning(self, sql: str, params: tuple | list = ()) -> list[Row]:
        with self._b.conn() as c:
            return c.execute(_q(sql), list(params)).fetchall()

    def lock_tenant(self) -> None:
        with self._b.conn() as c:  # released automatically when the surrounding transaction ends
            c.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (self.tenant,))

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._b.pinned():
            yield

    def put_vector(self, kind: str, ref_id: str, vec: np.ndarray) -> None:
        if vec.shape[-1] != self._b.embed_dim:
            raise ValueError(f"vector has {vec.shape[-1]} dims but the store is configured for EMBED_DIM={self._b.embed_dim}")
        with self._b.conn() as c:
            c.execute(
                "INSERT INTO vectors (kind, ref_id, tenant_id, vec) VALUES (%s,%s,%s,%s::halfvec) "
                "ON CONFLICT (kind, ref_id) DO UPDATE SET vec=EXCLUDED.vec, tenant_id=EXCLUDED.tenant_id",
                (kind, ref_id, self.tenant, _vec_literal(vec)),
            )

    def search_vectors(self, kind: str, query: np.ndarray, k: int = 10,
                       ref_ids: set[str] | None = None) -> list[tuple[str, float]]:
        lit = _vec_literal(query)
        sql = "SELECT ref_id, 1 - (vec <=> %s::halfvec) AS sim FROM vectors WHERE kind=%s AND tenant_id=%s"
        params: list[Any] = [lit, kind, self.tenant]
        if ref_ids is not None:
            if not ref_ids:
                return []
            sql += " AND ref_id = ANY(%s)"
            params.append(list(ref_ids))
        sql += " ORDER BY vec <=> %s::halfvec LIMIT %s"
        params += [lit, k]
        with self._b.conn() as c:
            return [(r["ref_id"], float(r["sim"])) for r in c.execute(sql, params).fetchall()]

    def delete_vectors(self, kind: str, ref_ids: list[str]) -> None:
        if ref_ids:
            with self._b.conn() as c:
                c.execute("DELETE FROM vectors WHERE kind=%s AND tenant_id=%s AND ref_id = ANY(%s)",
                          (kind, self.tenant, list(ref_ids)))

    def close(self) -> None:
        self._b.pool.close()
