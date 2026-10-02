from __future__ import annotations

import datetime as dt


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def iso(t: dt.datetime) -> str:
    """Fixed-width UTC timestamp: sorts correctly as text in both SQLite and Postgres."""
    return t.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def utcnow_iso() -> str:
    return iso(utcnow())
