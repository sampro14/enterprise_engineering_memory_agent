"""Tiny on-disk cache so repeated eval runs do not re-pay for identical calls."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading


class DiskCache:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v BLOB)")

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()

    def get(self, k: str) -> bytes | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return row[0] if row else None

    def set(self, k: str, v: bytes) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, v))
            self._db.commit()
