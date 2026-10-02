"""Minimal forward-only migration runner for Postgres.

Numbered SQL files live in `store/migrations/`. Each runs in its own transaction and is recorded in
`schema_migrations`, so `apply_migrations` is idempotent and safe to run from several containers at once
(guarded by an advisory lock)."""
from __future__ import annotations

import logging
from pathlib import Path

import psycopg

log = logging.getLogger(__name__)
MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_LOCK_ID = 727_272_001
HNSW_HALFVEC_MAX_DIM = 4000


def migration_files() -> list[Path]:
    return sorted(MIGRATIONS_DIR.glob("[0-9][0-9][0-9][0-9]_*.sql"))


def apply_migrations(database_url: str, embed_dim: int = 3072, schema: str | None = None) -> list[str]:
    """Apply pending migrations; returns the versions applied by this call."""
    if not 1 <= embed_dim <= HNSW_HALFVEC_MAX_DIM:
        raise ValueError(f"EMBED_DIM must be between 1 and {HNSW_HALFVEC_MAX_DIM} for halfvec + HNSW, got {embed_dim}")
    applied: list[str] = []
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_LOCK_ID,))
        try:
            # Extension lives in `public` so every tenant/test schema can resolve the vector types.
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
            if schema:
                conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
                conn.execute(f'SET search_path TO "{schema}", public')
            conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations "
                         "(version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())")
            done = {r[0] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()}
            for f in migration_files():
                version = f.stem
                if version in done:
                    continue
                sql = f.read_text().replace("{{EMBED_DIM}}", str(embed_dim))
                with conn.transaction():
                    conn.execute(sql)
                    conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
                log.info("applied migration %s", version)
                applied.append(version)
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_ID,))
    return applied
