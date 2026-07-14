"""Shared helpers for baseline comparison scripts."""

from __future__ import annotations

from typing import Any

import faiss
import numpy as np

from src.core.metrics import recall_at_k
from src.utils import ensure_float32_c
from src.utils import measure_qps


class FlatIndex:
    """Wrapper for exact IndexFlatL2 search."""

    def __init__(self, db_vecs: np.ndarray) -> None:
        d = db_vecs.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(ensure_float32_c(db_vecs))
        self._bpv = d * 4

    def search(
        self,
        queries: np.ndarray,
        top_k: int = 100,
        nprobe: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        del nprobe
        return self.index.search(ensure_float32_c(queries), top_k)

    def bytes_per_vec(self) -> int:
        return self._bpv


def make_baseline_record(
    *,
    method: str,
    alpha: float,
    nprobe: int,
    direction: str,
    retrieved: np.ndarray,
    gt: np.ndarray,
    qps: float,
    bpv: int,
    n_vectors: int,
    d: int,
    seed: int,
) -> dict:
    """Create a baseline CSV row from retrieved ids and timing."""
    r1 = recall_at_k(retrieved, gt, k=1)
    r10 = recall_at_k(retrieved, gt, k=10)
    r100 = recall_at_k(retrieved, gt, k=100)
    print(
        f"    np={nprobe:>4}  R@1={r1:.4f}  R@10={r10:.4f}  R@100={r100:.4f}  QPS={qps:.0f}"
    )
    return {
        "method": method,
        "alpha": alpha,
        "nprobe": nprobe,
        "direction": direction,
        "r1": round(r1, 4),
        "r10": round(r10, 4),
        "r100": round(r100, 4),
        "qps": round(qps, 1),
        "bytes_per_vec": bpv,
        "n_vectors": n_vectors,
        "d": d,
        "seed": seed,
    }


def run_baseline_nprobe_sweep(
    *,
    index: Any,
    queries: np.ndarray,
    gt: np.ndarray,
    method: str,
    alpha: float,
    direction: str,
    bpv: int,
    n_vectors: int,
    d: int,
    seed: int,
    nprobe_values: list[int],
    top_k: int,
    n_warmup: int,
    n_timed: int,
) -> list[dict]:
    """Evaluate one baseline index across nprobe/efSearch values."""
    records: list[dict] = []
    for nprobe in nprobe_values:
        _, retrieved, qps = measure_qps(
            index,
            queries,
            top_k,
            nprobe,
            n_warmup=n_warmup,
            n_timed=n_timed,
        )
        records.append(
            make_baseline_record(
                method=method,
                alpha=alpha,
                nprobe=nprobe,
                direction=direction,
                retrieved=retrieved,
                gt=gt,
                qps=qps,
                bpv=bpv,
                n_vectors=n_vectors,
                d=d,
                seed=seed,
            )
        )
    return records


def run_baseline_single(
    *,
    index: Any,
    queries: np.ndarray,
    gt: np.ndarray,
    method: str,
    alpha: float,
    direction: str,
    bpv: int,
    n_vectors: int,
    d: int,
    seed: int,
    nprobe: int,
    top_k: int,
    n_warmup: int,
    n_timed: int,
) -> dict:
    """Evaluate one baseline index at a single nprobe value."""
    _, retrieved, qps = measure_qps(
        index,
        queries,
        top_k,
        nprobe,
        n_warmup=n_warmup,
        n_timed=n_timed,
    )
    return make_baseline_record(
        method=method,
        alpha=alpha,
        nprobe=nprobe,
        direction=direction,
        retrieved=retrieved,
        gt=gt,
        qps=qps,
        bpv=bpv,
        n_vectors=n_vectors,
        d=d,
        seed=seed,
    )
