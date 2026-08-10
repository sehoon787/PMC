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


# ---------------------------------------------------------------------------
# Ranking-quality metrics (binary graded relevance over the top_k ground truth)
# ---------------------------------------------------------------------------

def average_precision_at_k(retrieved_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    """Average precision@K for ONE query, with all gt items equally relevant.

        AP@K = sum_r [ rel(r) * hits(r) / r ] / min(|gt|, K)

    where r runs over the first K retrieved ranks, rel(r) is 1 when rank r
    holds a ground-truth item, and hits(r) is the number of ground-truth items
    seen up to and including rank r.  The min(|gt|, K) normalizer is the number
    of hits a perfect ranking could achieve, so a perfect ranking scores 1.0.

    Parameters
    ----------
    retrieved_ids : (top_k,) int array of retrieved neighbor IDs for one query.
    gt_ids        : (n_gt,) int array of ground-truth IDs for the same query.
    k             : Number of retrieved ranks to consider.
    """
    gt = set(int(x) for x in gt_ids if x >= 0)
    if not gt:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        doc = int(doc_id)
        if doc >= 0 and doc in gt:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(gt), k)


def map_at_k(retrieved_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    """Mean average precision@K: average_precision_at_k averaged over queries.

    Parameters
    ----------
    retrieved_ids : (Q, top_k) int array of retrieved neighbor IDs.
    gt_ids        : (Q, n_gt) int array of ground-truth neighbor IDs.
    k             : Number of retrieved ranks to consider.
    """
    q = len(retrieved_ids)
    if q == 0:
        return 0.0
    total = sum(
        average_precision_at_k(retrieved_ids[i], gt_ids[i], k) for i in range(q)
    )
    return total / q


def ndcg_at_k(retrieved_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    """nDCG@K with binary gains (gain 1 per ground-truth hit).

        DCG  = sum_r rel(r) / log2(r + 1)          over the first K ranks
        IDCG = sum_{r=1..min(|gt|, K)} 1 / log2(r + 1)

    Averaged over queries; a perfect ranking scores 1.0.

    Parameters
    ----------
    retrieved_ids : (Q, top_k) int array of retrieved neighbor IDs.
    gt_ids        : (Q, n_gt) int array of ground-truth neighbor IDs.
    k             : Number of retrieved ranks to consider.
    """
    q = len(retrieved_ids)
    if q == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(1, k + 1) + 1.0)
    ideal_cum = np.cumsum(discounts)
    total = 0.0
    for i in range(q):
        gt = set(int(x) for x in gt_ids[i] if x >= 0)
        if not gt:
            continue
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_ids[i, :k], start=1):
            doc = int(doc_id)
            if doc >= 0 and doc in gt:
                dcg += discounts[rank - 1]
        idcg = float(ideal_cum[min(len(gt), k) - 1])
        total += dcg / idcg
    return total / q
