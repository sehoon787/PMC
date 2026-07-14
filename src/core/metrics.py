"""
metrics.py -- Recall metrics for ANN evaluation.

All metrics follow the standard K-recall@K convention:
    recall_at_k:    |retrieved_K ∩ gt_K| / K
    nn_recall_at_k: fraction of queries where the true 1-NN is in top-K
"""

from __future__ import annotations

import numpy as np


def recall_at_k(retrieved_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    """Standard K-recall@K: |retrieved_K ∩ gt_K| / K.

    Parameters
    ----------
    retrieved_ids : (Q, top_k) int array of retrieved neighbor IDs.
    gt_ids        : (Q, top_k) int array of ground-truth neighbor IDs.
    k             : Number of neighbors to consider.
    """
    q = len(retrieved_ids)
    total = 0.0
    for i in range(q):
        topk = set(int(x) for x in retrieved_ids[i, :k] if x >= 0)
        gt_k = set(int(x) for x in gt_ids[i, :k] if x >= 0)
        total += len(topk & gt_k) / max(len(gt_k), 1)
    return total / q if q > 0 else 0.0


def nn_recall_at_k(retrieved_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    """1-NN recall@K: fraction of queries where the true 1-NN is in top-K."""
    Q = len(retrieved_ids)
    hits = 0
    for i in range(Q):
        true_nn = int(gt_ids[i, 0])
        topk = set(int(x) for x in retrieved_ids[i, :k] if x >= 0)
        if true_nn in topk:
            hits += 1
    return hits / Q if Q > 0 else 0.0


def recall_at_k_single_gt(retrieved_ids: np.ndarray, gt_idx: np.ndarray, k: int) -> float:
    """Recall@K where each query has exactly one ground-truth index.

    Used for t→a direction in standard protocol: each caption maps to 1 audio.

    Parameters
    ----------
    retrieved_ids : (Q, top_k) int array of retrieved neighbor IDs.
    gt_idx        : (Q,) int array; gt_idx[i] is the single correct DB index for query i.
    k             : Number of neighbors to consider.

    Returns fraction of Q queries where gt_idx[i] appears in retrieved_ids[i, :k].
    """
    Q = len(retrieved_ids)
    hits = 0
    for i in range(Q):
        topk = set(int(x) for x in retrieved_ids[i, :k] if x >= 0)
        if int(gt_idx[i]) in topk:
            hits += 1
    return hits / Q if Q > 0 else 0.0


def recall_at_k_multi_gt(retrieved_ids: np.ndarray, gt_sets: list, k: int) -> float:
    """Recall@K where each query maps to multiple valid ground-truth indices.

    Used for a→t direction: each audio maps to 5 captions; hit = any in top-K.

    Parameters
    ----------
    retrieved_ids : (Q, top_k) int array of retrieved neighbor IDs.
    gt_sets       : length-Q list of sets; gt_sets[i] = set of correct DB indices.
    k             : Number of neighbors to consider.

    Returns fraction of Q queries where at least one GT index is in retrieved_ids[i, :k].
    """
    Q = len(retrieved_ids)
    hits = 0
    for i in range(Q):
        topk = set(int(x) for x in retrieved_ids[i, :k] if x >= 0)
        if topk & gt_sets[i]:
            hits += 1
    return hits / Q if Q > 0 else 0.0


def compute_ground_truth(queries: np.ndarray, db: np.ndarray, top_k: int = 100) -> np.ndarray:
    """Brute-force ground truth via IndexFlatL2. Returns (Q, top_k) int64 array."""
    import faiss
    from src.utils import ensure_float32_c

    d = db.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(ensure_float32_c(db))
    _, indices = index.search(ensure_float32_c(queries), top_k)
    return indices


def recall_dict(
    retrieved_ids: np.ndarray,
    gt_ids: np.ndarray,
    ks: "tuple[int, ...]" = (1, 10, 100),
) -> "dict[int, float]":
    """Compute recall@k for multiple k values. Returns {k: recall}."""
    return {k: round(recall_at_k(retrieved_ids, gt_ids, k), 6) for k in ks}
