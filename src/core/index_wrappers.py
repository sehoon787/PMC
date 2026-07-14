"""
index_wrappers.py -- Index building functions and wrappers for FAISS indexes.

Extracted from duplicated code across scripts 01-03, 05, 12b.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import faiss

from src.core.pmc import SimpleRaBitQIndex
from src.utils import ensure_float32_c


def compute_nlist(n: int, max_nlist: int = 64) -> int:
    """Compute IVF cell count: max(1, min(max_nlist, n // 10))."""
    return max(1, min(max_nlist, n // 10))


# ---------------------------------------------------------------------------
# Vanilla RaBitQ
# ---------------------------------------------------------------------------

def build_vanilla_rabitq(
    db_vecs: np.ndarray, nlist: int, seed: int,
) -> SimpleRaBitQIndex:
    """Build an IVFRaBitQFastScan index. Returns SimpleRaBitQIndex wrapper."""
    db_vecs = ensure_float32_c(db_vecs)
    n, d = db_vecs.shape
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFRaBitQFastScan(quantizer, d, nlist, 0)
    index.cp.seed = seed
    index.cp.min_points_per_centroid = 1
    print(f"  [rabitq_fs] Training (d={d}, nlist={nlist}) on {n} vectors ...")
    index.train(db_vecs)
    index.add(db_vecs)
    print(f"  [rabitq_fs] Done. code_size={index.code_size} bytes/vec")
    return SimpleRaBitQIndex(index, d)


# ---------------------------------------------------------------------------
# IVFPQ
# ---------------------------------------------------------------------------

class IVFPQWrapper:
    """Wrapper for FAISS IndexIVFPQ with a consistent .search() interface."""

    def __init__(self, raw_index: Any, d: int, bpv: int) -> None:
        self.index = raw_index
        self.d = d
        self._bpv = bpv

    def search(
        self, queries: np.ndarray, top_k: int = 100, nprobe: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray]:
        queries = ensure_float32_c(queries)
        self.index.nprobe = nprobe
        return self.index.search(queries, top_k)

    def bytes_per_vec(self) -> int:
        return self._bpv


def build_ivfpq_raw(
    db_vecs: np.ndarray,
    nlist: int,
    m: int,
    nbits: int,
    seed: int,
    train_vecs: "np.ndarray | None" = None,
) -> faiss.IndexIVFPQ:
    """Build a trained+populated IVFPQ index (raw faiss object, no wrapper).

    Parameters
    ----------
    db_vecs   : (N, d) float32 array — vectors to add to the index.
    nlist     : number of IVF cells
    m         : number of PQ sub-quantizers (must divide d)
    nbits     : bits per sub-quantizer (typically 4, 6, or 8)
    seed      : random seed for k-means
    train_vecs: optional separate training array; defaults to db_vecs when None.
    """
    db_vecs = ensure_float32_c(db_vecs)
    tv = ensure_float32_c(train_vecs) if train_vecs is not None else db_vecs
    n, d = db_vecs.shape
    if d % m != 0:
        raise ValueError(f"M={m} must divide d={d}")
    quantizer = faiss.IndexFlatL2(d)
    ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    ivfpq.cp.seed = seed
    ivfpq.cp.min_points_per_centroid = 1
    print(f"  [ivfpq] Training (d={d}, nlist={nlist}, M={m}, nbits={nbits}) on {len(tv)} vectors ...")
    ivfpq.train(tv)
    ivfpq.add(db_vecs)
    print(f"  [ivfpq] Done.")
    return ivfpq


def build_ivfpq(
    db_vecs: np.ndarray, target_bpv: int, nlist: int, seed: int,
) -> Tuple[IVFPQWrapper, int]:
    """Build IVFPQ index at target bytes/vec. Returns (wrapper, actual_bpv)."""
    d = db_vecs.shape[1]
    m = target_bpv  # with nbits=8, M = target_bytes
    # m must divide d
    while m > 0 and d % m != 0:
        m -= 1
    if m <= 0:
        raise ValueError(f"No valid M for d={d}, target={target_bpv}")
    raw_index = build_ivfpq_raw(db_vecs, nlist, m, 8, seed)
    return IVFPQWrapper(raw_index, d, m), m


# ---------------------------------------------------------------------------
# OPQ + IVFPQ
# ---------------------------------------------------------------------------

class OPQIVFPQWrapper:
    """Wrapper for OPQ+IVFPQ (IndexPreTransform).

    nprobe is set on the inner IVF index extracted from the transform chain.
    """

    def __init__(self, outer_index: Any, inner_ivf: Any, bpv: int) -> None:
        self.index = outer_index
        self.inner_ivf = inner_ivf
        self._bpv = bpv

    def search(
        self, queries: np.ndarray, top_k: int = 100, nprobe: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray]:
        queries = ensure_float32_c(queries)
        if self.inner_ivf is not None:
            self.inner_ivf.nprobe = nprobe
        return self.index.search(queries, top_k)

    def bytes_per_vec(self) -> int:
        return self._bpv


# ---------------------------------------------------------------------------
# HNSW
# ---------------------------------------------------------------------------

class HNSWIndex:
    """Wrapper for faiss HNSW with .search(queries, top_k, nprobe) interface.

    nprobe maps to efSearch for API compatibility.  If ef_search is supplied
    at construction the effective efSearch at search time is
    max(nprobe, ef_search); otherwise efSearch is set to nprobe directly.
    """

    def __init__(
        self, db_vecs: np.ndarray, M: int = 32, ef_search: int | None = None,
    ) -> None:
        d = db_vecs.shape[1]
        self.index = faiss.IndexHNSWFlat(d, M)
        self._ef_search = ef_search
        if ef_search is not None:
            self.index.hnsw.efSearch = ef_search
        self.d = d
        self._M = M
        # B/vec: float32 vectors + HNSW graph (~M*2 links per node, 4 bytes each)
        self._bpv = d * 4 + M * 2 * 4
        print(f"  [hnsw M={M}] Adding {db_vecs.shape[0]} vectors (d={d}) ...")
        self.index.add(ensure_float32_c(db_vecs))
        print(f"  [hnsw M={M}] Done. B/vec~{self._bpv}")

    def search(
        self, queries: np.ndarray, top_k: int = 100, nprobe: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray]:
        queries = ensure_float32_c(queries)
        if self._ef_search is not None:
            self.index.hnsw.efSearch = max(nprobe, self._ef_search)
        else:
            self.index.hnsw.efSearch = nprobe
        return self.index.search(queries, top_k)

    def bytes_per_vec(self) -> int:
        return self._bpv


def build_hnsw(db_vecs: np.ndarray, M: int = 32) -> HNSWIndex:
    """Build an HNSW index. Returns HNSWIndex wrapper."""
    return HNSWIndex(db_vecs, M=M)


def build_opq_ivfpq_raw(
    db_vecs: np.ndarray,
    nlist: int,
    m: int,
    seed: int,
    d_out: "int | None" = None,
) -> faiss.Index:
    """Build a trained+populated OPQ+IVFPQ index (raw faiss object, no wrapper).

    Parameters
    ----------
    db_vecs : (N, d) float32 array
    nlist   : number of IVF cells
    m       : number of PQ sub-quantizers
    seed    : random seed for k-means
    d_out   : OPQ output dimension (defaults to d if None)
    """
    db_vecs = ensure_float32_c(db_vecs)
    n, d = db_vecs.shape
    if d_out is None:
        d_out = d
    factory_str = f"OPQ{m}_{d_out},IVF{nlist},PQ{m}x8"
    print(f"  [opq_ivfpq] Building with factory='{factory_str}' on {n} vectors ...")
    index = faiss.index_factory(d, factory_str)
    try:
        inner_ivf = faiss.extract_index_ivf(index)
        inner_ivf.cp.seed = seed
        inner_ivf.cp.min_points_per_centroid = 1
    except Exception as e:
        print(f"  [opq_ivfpq] WARNING: could not set seed on inner IVF: {e}")
    index.train(db_vecs)
    index.add(db_vecs)
    print(f"  [opq_ivfpq] Done.")
    return index


def build_opq_ivfpq(
    db_vecs: np.ndarray, target_bpv: int, nlist: int, seed: int,
) -> OPQIVFPQWrapper:
    """Build OPQ+IVFPQ via faiss index_factory. Returns OPQIVFPQWrapper."""
    d = db_vecs.shape[1]
    m = target_bpv
    while m > 0 and d % m != 0:
        m -= 1
    if m <= 0:
        raise ValueError(f"No valid M for d={d}, target={target_bpv}")
    raw_index = build_opq_ivfpq_raw(db_vecs, nlist, m, seed)
    try:
        inner_ivf = faiss.extract_index_ivf(raw_index)
    except Exception:
        inner_ivf = None
    return OPQIVFPQWrapper(raw_index, inner_ivf, m)
