"""Shared PQ/OPQ experiment helpers."""

from __future__ import annotations

import numpy as np

from src.core.index_wrappers import build_ivfpq, build_opq_ivfpq
from src.core.metrics import compute_ground_truth, recall_at_k
from src.core.pmc import compute_gap, shift_db_vectors
from src.utils import ensure_float32_c


def run_pq_direction(
    *,
    db_emb: np.ndarray,
    query_emb: np.ndarray,
    direction: str,
    nlist: int,
    nprobe: int,
    top_k: int,
    seed: int,
    include_shape_fields: bool,
) -> list[dict]:
    """Run IVFPQ/OPQ with and without PMC for one retrieval direction."""
    n_vectors, d = db_emb.shape
    gt = compute_ground_truth(query_emb, db_emb, top_k=top_k)
    gap = compute_gap(db_emb, query_emb)

    print(f"\n{'='*70}")
    print(f"Direction: {direction}  N={n_vectors}, d={d}  gap_norm={np.linalg.norm(gap):.4f}")
    print("=" * 70)

    records: list[dict] = []
    for method_name, use_opq, use_pmc, label in [
        ("ivfpq", False, False, "IVFPQ"),
        ("ivfpq_pmc", False, True, "IVFPQ+PMC"),
        ("opq_ivfpq", True, False, "OPQ"),
        ("opq_ivfpq_pmc", True, True, "OPQ+PMC"),
    ]:
        db_use = shift_db_vectors(db_emb, gap, alpha=1.0) if use_pmc else db_emb
        print(f"  [{label}] Building ...")
        try:
            if use_opq:
                index = build_opq_ivfpq(db_use, target_bpv=64, nlist=nlist, seed=seed)
            else:
                index, _ = build_ivfpq(db_use, target_bpv=64, nlist=nlist, seed=seed)

            _, ids = index.search(ensure_float32_c(query_emb), top_k, nprobe)
            r1 = recall_at_k(ids, gt, k=1)
            r10 = recall_at_k(ids, gt, k=10)
            r100 = recall_at_k(ids, gt, k=100)
            print(f"    {label:<11} R@1={r1:.4f}  R@10={r10:.4f}  R@100={r100:.4f}")

            record = {
                "method": method_name,
                "nprobe": nprobe,
                "direction": direction,
                "r1": round(r1, 4),
                "r10": round(r10, 4),
                "r100": round(r100, 4),
                "seed": seed,
            }
            if include_shape_fields:
                record["n_vectors"] = n_vectors
                record["d"] = d
            records.append(record)
        except Exception as exc:
            print(f"    {label} SKIP: {exc}")

    return records
