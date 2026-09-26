"""Brute-force in-process vector search (fine for PoC scale; pgvector in the MVP)."""
from __future__ import annotations

import numpy as np


def top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Cosine similarity for L2-normalized vectors. Returns [(row_index, score)] best first."""
    if matrix.shape[0] == 0:
        return []
    scores = matrix @ query
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]
