"""
pmc.py -- Per-Modality Centroid correction for RaBitQ cross-modal search.

Corrects the modality gap BEFORE RaBitQ index construction so that
the binary sign() encoding is fair to cross-modal queries.

Core idea:
    gap = mean(query_modality) - mean(db_modality)
    DB shift:    x' = x + alpha * gap          (move DB toward query center)
    Query shift: q' = q - (1 - alpha) * gap    (move query toward DB center)

    alpha = 0.5 => both meet at the midpoint
    alpha = 0.0 => no shift (vanilla)
    alpha = 1.0 => DB fully aligned to query center

All shifted vectors are L2-normalized before indexing / searching,
because RaBitQ operates on unit-norm vectors.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# L2 normalization (local copy to avoid circular imports)
# ---------------------------------------------------------------------------

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize each row to unit length."""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norm, 1e-12)).astype(np.float32)


# ---------------------------------------------------------------------------
# Gap computation
# ---------------------------------------------------------------------------

def compute_gap(
    db_emb: np.ndarray,
    query_emb: np.ndarray,
) -> np.ndarray:
    """
    Compute modality gap vector: mean(query) - mean(db).

    Parameters
    ----------
    db_emb:    (N, d) database modality embeddings
    query_emb: (Q, d) query modality embeddings

    Returns
    -------
    gap: (d,) float32
    """
    return (query_emb.mean(axis=0) - db_emb.mean(axis=0)).astype(np.float32)


# ---------------------------------------------------------------------------
# Shifting functions
# ---------------------------------------------------------------------------

def shift_db_vectors(
    db_emb: np.ndarray,
    gap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Shift DB vectors toward query distribution center.

    x_shifted = x + alpha * gap

    Returns L2-normalized shifted vectors.

    Parameters
    ----------
    db_emb: (N, d) float32
    gap:    (d,) float32 -- from compute_gap
    alpha:  interpolation factor (0 = no shift, 1 = fully align to query)

    Returns
    -------
    (N, d) float32, L2-normalized
    """
    shifted = db_emb + alpha * gap[np.newaxis, :]
    return _l2_normalize(shifted)


def shift_query_vectors(
    query_emb: np.ndarray,
    gap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Shift query vectors toward DB distribution center.

    q_shifted = q - (1 - alpha) * gap

    Must use same alpha as shift_db_vectors for consistency.
    Returns L2-normalized shifted vectors.

    Parameters
    ----------
    query_emb: (Q, d) float32
    gap:       (d,) float32 -- from compute_gap
    alpha:     interpolation factor (same as used for DB shift)

    Returns
    -------
    (Q, d) float32, L2-normalized
    """
    shifted = query_emb - (1.0 - alpha) * gap[np.newaxis, :]
    return _l2_normalize(shifted)


# ---------------------------------------------------------------------------
# Build PMC-corrected RaBitQ index
# ---------------------------------------------------------------------------

def _ensure_float32_c(arr: np.ndarray) -> np.ndarray:
    """Return arr as a contiguous float32 array."""
    arr = np.asarray(arr, dtype=np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def build_pmc_rabitq_index(
    db_emb: np.ndarray,
    query_emb_for_gap: np.ndarray,
    alpha: float = 0.5,
    nlist: int = 64,
    seed: int = 42,
    use_fastscan: bool = True,
) -> Tuple[Any, np.ndarray]:
    """
    Build RaBitQ index on PMC-shifted DB vectors.

    Uses faiss directly (no pareto_calibration wrapper) to avoid
    package name collisions.

    Parameters
    ----------
    db_emb:            (N, d) float32 -- original DB embeddings (L2-normalized)
    query_emb_for_gap: (Q, d) float32 -- query embeddings used ONLY to compute
                       the gap vector (not indexed)
    alpha:   interpolation factor (0.5 = meet halfway)
    nlist:   IVF coarse cells
    seed:    random seed
    use_fastscan: if True, use IVFRaBitQFastScan; else IVFRaBitQ

    Returns
    -------
    (index_wrapper, gap_vector)
        index_wrapper: SimpleRaBitQIndex with .search() and .bytes_per_vec()
        gap:           (d,) float32 gap vector (needed for query shifting)
    """
    import faiss

    gap = compute_gap(db_emb, query_emb_for_gap)
    db_shifted = _ensure_float32_c(shift_db_vectors(db_emb, gap, alpha=alpha))
    n, d = db_shifted.shape

    quantizer = faiss.IndexFlatL2(d)
    if use_fastscan:
        index = faiss.IndexIVFRaBitQFastScan(quantizer, d, nlist, 0)
    else:
        index = faiss.IndexIVFRaBitQ(quantizer, d, nlist, 0)

    index.cp.seed = seed
    index.cp.min_points_per_centroid = 1

    tag = "rabitq_fs" if use_fastscan else "rabitq"
    print(f"  [pmc/{tag}] Training (d={d}, nlist={nlist}) on {n} vectors ...")
    index.train(db_shifted)
    index.add(db_shifted)
    print(f"  [pmc/{tag}] Done. code_size={index.code_size} bytes/vec")

    wrapper = SimpleRaBitQIndex(index, d)
    return wrapper, gap


class SimpleRaBitQIndex:
    """Thin wrapper around a raw faiss IVFRaBitQ[FastScan] index.

    Provides .search(queries, top_k, nprobe) and .bytes_per_vec()
    matching the pareto_calibration index API.
    """

    def __init__(self, raw_index: Any, d: int) -> None:
        self.index = raw_index
        self.d = d

    def search(
        self,
        queries: np.ndarray,
        top_k: int = 100,
        nprobe: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray]:
        queries = _ensure_float32_c(queries)
        self.index.nprobe = nprobe
        return self.index.search(queries, top_k)

    def bytes_per_vec(self) -> int:
        return int(self.index.code_size)


# ---------------------------------------------------------------------------
# Search with PMC-shifted queries
# ---------------------------------------------------------------------------

def search_pmc(
    index: Any,
    queries: np.ndarray,
    gap: np.ndarray,
    alpha: float = 0.5,
    top_k: int = 100,
    nprobe: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search with PMC-shifted queries.

    1. Shift queries: q' = q - (1-alpha) * gap
    2. L2 normalize
    3. Search index

    Parameters
    ----------
    index:   SimpleRaBitQIndex or any index with .search(q, top_k, nprobe)
    queries: (Q, d) float32 -- original query embeddings
    gap:     (d,) float32 -- from build_pmc_rabitq_index
    alpha:   same alpha used during build
    top_k:   number of results per query
    nprobe:  IVF probe count

    Returns
    -------
    distances: (Q, top_k) float32
    indices:   (Q, top_k) int64
    """
    q_shifted = shift_query_vectors(queries, gap, alpha=alpha)
    return index.search(q_shifted, top_k=top_k, nprobe=nprobe)
