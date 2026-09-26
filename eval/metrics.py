"""Grading and aggregation. Programmatic (no LLM judge): compares the answer's structured `values`."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from eval.scenarios import Question


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _has(values: list[str], target: str) -> bool:
    t = norm(target)
    return any(t and (t in norm(v) or (norm(v) and norm(v) in t)) for v in values)


@dataclass
class Graded:
    qid: str
    category: str
    kind: str
    correct: bool
    stale: bool  # a forbidden (stale/poisoned/planned) value was asserted
    unknown: bool


def grade(q: Question, values: list[str], unknown: bool) -> Graded:
    if q.kind == "when":
        correct = any(norm(v) == norm(q.expected[0]) for v in values)
    else:
        correct = all(_has(values, e) for e in q.expected) and bool(values)
    stale = any(_has(values, f) for f in q.forbidden)
    if q.kind != "multi":
        correct = correct and not stale
    return Graded(q.id, q.category, q.kind, correct, stale, unknown or not values)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap_diff(a: list[bool], b: list[bool], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of mean(a) - mean(b) for paired per-question outcomes."""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]
