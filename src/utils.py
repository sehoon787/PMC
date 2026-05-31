"""
utils.py -- Shared utility functions for v4 RaBitQ cross-modal experiments.

Extracted from duplicated helpers across scripts 01-13.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np

from .fixtures.synthetic import synthetic_dataset
from .io.bigann import read_fbin


def ensure_float32_c(arr: np.ndarray) -> np.ndarray:
    """Return *arr* as a contiguous float32 array (no-op if already correct)."""
    arr = np.asarray(arr, dtype=np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a 2-D array."""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norm, 1e-12)).astype(np.float32)

def compute_modality_means(features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute per-modality mean vectors."""
    return {mod: emb.mean(axis=0).astype(np.float32) for mod, emb in features_dict.items()}


def apply_meanshift(
    query_emb: np.ndarray,
    db_means: Dict[str, np.ndarray],
    query_modality: str,
    db_modality: str,
) -> np.ndarray:
    """Standard mean-shift: q' = q - mean(query_mod) + mean(db_mod)."""
    q_mean = db_means[query_modality]
    db_mean = db_means[db_modality]
    return (query_emb - q_mean + db_mean).astype(np.float32)

def timed_search(
    index,
    queries: np.ndarray,
    top_k: int,
    nprobe: int,
    n_warmup: int = 1,
    n_timed: int = 5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run warmup + timed searches. Returns (distances, indices, qps).

    Single-thread timing, median of n_timed runs -> QPS.
    """
    queries = ensure_float32_c(queries)

    # warmup
    for _ in range(n_warmup):
        index.search(queries, top_k=top_k, nprobe=nprobe)

    # timed runs
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        dists, ids = index.search(queries, top_k=top_k, nprobe=nprobe)
        times.append(time.perf_counter() - t0)

    median_sec = float(np.median(times))
    qps = len(queries) / median_sec if median_sec > 0 else 0.0
    return dists, ids, qps


def measure_qps(
    index,
    queries: np.ndarray,
    top_k: int,
    nprobe: int,
    n_warmup: int = 1,
    n_timed: int = 5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Time index search and return (distances, indices, qps).

    Sets faiss.omp_set_num_threads(1) for reproducible single-thread timing.
    Runs n_warmup discarded passes then n_timed timed passes; returns the
    median QPS.
    """
    import faiss

    faiss.omp_set_num_threads(1)
    queries = ensure_float32_c(queries)
    n_queries = len(queries)

    for _ in range(n_warmup):
        index.search(queries, top_k=top_k, nprobe=nprobe)

    times = []
    last_D = None
    last_I = None
    for _ in range(n_timed):
        t0 = time.perf_counter()
        D, I = index.search(queries, top_k=top_k, nprobe=nprobe)
        times.append(time.perf_counter() - t0)
        last_D = D
        last_I = I

    median_sec = float(np.median(times))
    qps = n_queries / median_sec if median_sec > 0 else 0.0
    return last_D, last_I, qps
