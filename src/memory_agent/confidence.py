"""Confidence calculation (design 14C). Weights are hyperparameters in config."""
from __future__ import annotations

import datetime as dt
import math

from .config import ConfidenceWeights

VERIFIED_AUTHORITY = 0.60  # authority at or above this counts as "verified" evidence


def _days(a: str, b: str) -> int:
    return abs((dt.date.fromisoformat(b[:10]) - dt.date.fromisoformat(a[:10])).days)


def compute(
    w: ConfidenceWeights,
    authority: float,
    n_sources: int,
    last_verified: str | None,
    now: str,
    contradiction: float = 0.0,
) -> float:
    src = 1 - 0.5 ** max(n_sources, 1)  # diminishing returns per independent source
    verified = 1.0 if authority >= VERIFIED_AUTHORITY else 0.0
    recency = math.exp(-_days(last_verified, now) / 365) if last_verified else 1.0
    c = (
        w.authority * authority
        + w.sources * src
        + w.verification * verified
        + w.recency * recency
        - w.contradiction * contradiction
    )
    return max(0.0, min(1.0, c))
