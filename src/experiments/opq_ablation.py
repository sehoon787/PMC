"""OPQ rotation ablation: frozen-R vs joint-retrain under PMC shift.

Conditions tested per (dataset, direction):
  1. OPQ (vanilla)       — rotation + PQ + IVF all on original data
  2. OPQ+PMC (joint)     — rotation + PQ + IVF all retrained on shifted data
  3. OPQ-frozenR+PMC     — rotation from (1), PQ + IVF retrained on shifted data

This isolates whether the OPQ rotation itself is the source of regression
when PMC is applied, or whether the issue is structural.
"""

from __future__ import annotations

import numpy as np
import faiss

from src.core.metrics import compute_ground_truth, recall_at_k
from src.core.pmc import compute_gap, shift_db_vectors
from src.utils import ensure_float32_c


def _determine_m(d: int, target_bpv: int = 64) -> int:
    """Find largest m <= target_bpv that divides d."""
    m = target_bpv
    while m > 0 and d % m != 0:
        m -= 1
    if m <= 0:
        raise ValueError(f"No valid M for d={d}, target={target_bpv}")
    return m


def _extract_opq_rotation(trained_index: faiss.Index) -> np.ndarray:
    """Extract the OPQ rotation matrix A from a trained IndexPreTransform.

    Returns A of shape (d_out, d_in) such that y = x @ A.T.
    """
    vt = faiss.downcast_VectorTransform(trained_index.chain.at(0))
    d_out, d_in = vt.d_out, vt.d_in
    A = faiss.vector_to_array(vt.A).reshape(d_out, d_in).copy()
    return A


def _build_opq_full(db: np.ndarray, nlist: int, m: int, seed: int) -> faiss.Index:
    """Train+populate a full OPQ+IVFPQ index."""
    db = ensure_float32_c(db)
    d = db.shape[1]
    factory_str = f"OPQ{m}_{d},IVF{nlist},PQ{m}x8"
    index = faiss.index_factory(d, factory_str)
    try:
        inner_ivf = faiss.extract_index_ivf(index)
        inner_ivf.cp.seed = seed
        inner_ivf.cp.min_points_per_centroid = 1
    except Exception:
        pass
    index.train(db)
    index.add(db)
    return index


def _build_ivfpq_plain(
    db: np.ndarray, nlist: int, m: int, seed: int,
) -> faiss.IndexIVFPQ:
    """Train+populate a plain IVFPQ (no OPQ rotation)."""
    db = ensure_float32_c(db)
    n, d = db.shape
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    index.cp.seed = seed
    index.cp.min_points_per_centroid = 1
    index.train(db)
    index.add(db)
    return index


def _search(index: faiss.Index, queries: np.ndarray, top_k: int, nprobe: int):
    """Search with nprobe setting."""
    queries = ensure_float32_c(queries)
    try:
        inner_ivf = faiss.extract_index_ivf(index)
        inner_ivf.nprobe = nprobe
    except Exception:
        if hasattr(index, "nprobe"):
            index.nprobe = nprobe
    return index.search(queries, top_k)


def _rotation_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius cosine similarity between two rotation matrices."""
    num = np.sum(A * B)
    denom = np.linalg.norm(A, "fro") * np.linalg.norm(B, "fro")
    return float(num / max(denom, 1e-12))


def run_opq_ablation_direction(
    *,
    db_emb: np.ndarray,
    query_emb: np.ndarray,
    direction: str,
    nlist: int = 64,
    nprobe: int = 16,
    top_k: int = 100,
    seed: int = 42,
) -> list[dict]:
    """Run the 3-condition OPQ ablation for one retrieval direction.

    Returns a list of dicts, one per condition, with recall metrics and
    rotation analysis fields.
    """
    n, d = db_emb.shape
    m = _determine_m(d)
    gt = compute_ground_truth(query_emb, db_emb, top_k=top_k)
    gap = compute_gap(db_emb, query_emb)
    gap_norm = float(np.linalg.norm(gap))

    db_shifted = shift_db_vectors(db_emb, gap, alpha=1.0)

    print(f"\n{'='*70}")
    print(f"OPQ Ablation — {direction}  N={n} d={d} m={m}  gap_norm={gap_norm:.4f}")
    print("=" * 70)

    records: list[dict] = []

    # --- Condition 1: OPQ vanilla ---
    print("  [1/3] OPQ (vanilla) ...")
    opq_vanilla = _build_opq_full(db_emb, nlist, m, seed)
    R_original = _extract_opq_rotation(opq_vanilla)
    _, ids = _search(opq_vanilla, query_emb, top_k, nprobe)
    r1 = recall_at_k(ids, gt, k=1)
    r10 = recall_at_k(ids, gt, k=10)
    r100 = recall_at_k(ids, gt, k=100)
    print(f"        R@1={r1:.4f}  R@10={r10:.4f}  R@100={r100:.4f}")
    records.append({
        "condition": "opq_vanilla",
        "direction": direction,
        "r1": round(r1, 4), "r10": round(r10, 4), "r100": round(r100, 4),
        "n_vectors": n, "d": d, "m": m,
        "gap_norm": round(gap_norm, 4),
        "rotation_sim_to_vanilla": 1.0,
        "nprobe": nprobe, "seed": seed,
    })

    # --- Condition 2: OPQ+PMC (joint retrain) ---
    print("  [2/3] OPQ+PMC (joint retrain) ...")
    opq_joint = _build_opq_full(db_shifted, nlist, m, seed)
    R_joint = _extract_opq_rotation(opq_joint)
    _, ids = _search(opq_joint, query_emb, top_k, nprobe)
    r1 = recall_at_k(ids, gt, k=1)
    r10 = recall_at_k(ids, gt, k=10)
    r100 = recall_at_k(ids, gt, k=100)
    rot_sim_joint = _rotation_similarity(R_original, R_joint)
    print(f"        R@1={r1:.4f}  R@10={r10:.4f}  R@100={r100:.4f}"
          f"  rot_sim={rot_sim_joint:.4f}")
    records.append({
        "condition": "opq_pmc_joint",
        "direction": direction,
        "r1": round(r1, 4), "r10": round(r10, 4), "r100": round(r100, 4),
        "n_vectors": n, "d": d, "m": m,
        "gap_norm": round(gap_norm, 4),
        "rotation_sim_to_vanilla": round(rot_sim_joint, 4),
        "nprobe": nprobe, "seed": seed,
    })

    # --- Condition 3: OPQ-frozenR + PMC ---
    print("  [3/3] OPQ-frozenR+PMC (rotation from vanilla, PQ retrained on shifted) ...")
    # Apply the vanilla rotation to shifted data
    db_shifted_rotated = ensure_float32_c(db_shifted @ R_original.T)
    query_rotated = ensure_float32_c(query_emb @ R_original.T)

    # Build plain IVFPQ on the rotated+shifted data
    ivfpq_frozen = _build_ivfpq_plain(db_shifted_rotated, nlist, m, seed)
    _, ids = _search(ivfpq_frozen, query_rotated, top_k, nprobe)
    r1 = recall_at_k(ids, gt, k=1)
    r10 = recall_at_k(ids, gt, k=10)
    r100 = recall_at_k(ids, gt, k=100)
    print(f"        R@1={r1:.4f}  R@10={r10:.4f}  R@100={r100:.4f}")
    records.append({
        "condition": "opq_frozenR_pmc",
        "direction": direction,
        "r1": round(r1, 4), "r10": round(r10, 4), "r100": round(r100, 4),
        "n_vectors": n, "d": d, "m": m,
        "gap_norm": round(gap_norm, 4),
        "rotation_sim_to_vanilla": 1.0,
        "nprobe": nprobe, "seed": seed,
    })

    # Summary
    print(f"\n  Summary ({direction}):")
    for r in records:
        print(f"    {r['condition']:<22} R@100={r['r100']:.4f}")

    return records
