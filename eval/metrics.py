"""
Retrieval evaluation metrics for Spotlight: Recall@K and MRR.

All metrics are computed against held-out positive (user, item) pairs by issuing
ANN queries with the trained user embeddings and checking where each user's true
held-out item lands in the returned ranked list.

Definitions
-----------
For a held-out pair (u, i*) and the ranked list R_u of item ids returned by the
index for user u:

* Recall@K  = mean over users of  1[ i* in top-K of R_u ].
  (With exactly one held-out positive per user, Recall@K equals Hit-Rate@K.)

* MRR       = mean over users of  1 / rank(i*),  where rank is the 1-based
  position of i* in R_u, or 0 if i* is not retrieved within the cutoff.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def recall_at_k(retrieved_ids: np.ndarray, true_ids: np.ndarray, k: int) -> float:
    """
    Fraction of users whose true item appears in the top-K retrieved ids.

    Args:
        retrieved_ids: (Q, K_max) item ids ranked best-first per user. K_max must
            be >= ``k``.
        true_ids: (Q,) the single held-out positive item id per user.
        k: cutoff.

    Returns:
        Recall@K in [0, 1].
    """
    if retrieved_ids.shape[1] < k:
        raise ValueError(
            f"retrieved_ids has only {retrieved_ids.shape[1]} columns, need >= {k}"
        )
    topk = retrieved_ids[:, :k]
    hits = (topk == true_ids[:, None]).any(axis=1)
    return float(hits.mean())


def mrr(retrieved_ids: np.ndarray, true_ids: np.ndarray) -> float:
    """
    Mean Reciprocal Rank of the true item across users.

    Args:
        retrieved_ids: (Q, K_max) item ids ranked best-first per user.
        true_ids: (Q,) the single held-out positive item id per user.

    Returns:
        MRR in [0, 1]; users whose item is not retrieved contribute 0.
    """
    matches = retrieved_ids == true_ids[:, None]  # (Q, K_max)
    # First (best) matching column index per user, or -1 if no match.
    has_match = matches.any(axis=1)
    first_pos = np.argmax(matches, axis=1)  # 0 when no match -> guarded below
    ranks = np.where(has_match, first_pos + 1, 0)  # 1-based rank, 0 == miss
    reciprocal = np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0)
    return float(reciprocal.mean())


def evaluate_retrieval(
    retrieved_ids: np.ndarray,
    true_ids: np.ndarray,
    ks: Sequence[int] = (10, 50),
) -> Dict[str, float]:
    """
    Compute Recall@K (for each K) and MRR in one pass.

    Args:
        retrieved_ids: (Q, K_max) ranked item ids per user (K_max >= max(ks)).
        true_ids: (Q,) held-out positive item id per user.
        ks: cutoffs to report Recall at.

    Returns:
        dict like ``{"recall@10": ..., "recall@50": ..., "mrr": ...}``.
    """
    results: Dict[str, float] = {}
    for k in ks:
        results[f"recall@{k}"] = recall_at_k(retrieved_ids, true_ids, k)
    results["mrr"] = mrr(retrieved_ids, true_ids)
    return results
