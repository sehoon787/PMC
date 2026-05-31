"""Reusable experiment sweep helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.index_wrappers import build_ivfpq, build_vanilla_rabitq as build_rabitq
from src.core.metrics import compute_ground_truth, nn_recall_at_k, recall_at_k
from src.core.pmc import compute_gap, shift_db_vectors, shift_query_vectors
from src.utils import l2_normalize, measure_qps


def run_recall_qps_sweep(
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
    """Evaluate one index/query pair across nprobe values with recall and QPS."""
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

        r1 = recall_at_k(retrieved, gt, k=1)
        r10 = recall_at_k(retrieved, gt, k=10)
        r100 = recall_at_k(retrieved, gt, k=100)
        nn_r10 = nn_recall_at_k(retrieved, gt, k=10)
        nn_r100 = nn_recall_at_k(retrieved, gt, k=100)

        print(
            f"    np={nprobe:>2}  R@1={r1:.4f}  R@10={r10:.4f}  R@100={r100:.4f}  "
            f"1NN@10={nn_r10:.4f}  1NN@100={nn_r100:.4f}  QPS={qps:.0f}"
        )
        records.append(
            {
                "method": method,
                "alpha": alpha,
                "nprobe": nprobe,
                "direction": direction,
                "r1": round(r1, 4),
                "r10": round(r10, 4),
                "r100": round(r100, 4),
                "nn_r10": round(nn_r10, 4),
                "nn_r100": round(nn_r100, 4),
                "qps": round(qps, 1),
                "bytes_per_vec": bpv,
                "n_vectors": n_vectors,
                "d": d,
                "seed": seed,
            }
        )
    return records


def run_pmc_qps_pareto_direction(
    *,
    db_emb: np.ndarray,
    query_emb: np.ndarray,
    direction: str,
    nlist: int,
    seed: int,
    nprobe_values: list[int],
    top_k: int,
    n_warmup: int,
    n_timed: int,
) -> list[dict]:
    """Run RaBitQ/PMC/meanshift/IVFPQ QPS Pareto evaluation for one direction."""
    n_vectors, d = db_emb.shape
    n_queries = query_emb.shape[0]
    records: list[dict] = []

    print(f"\n{'='*70}")
    print(f"Direction: {direction}  N_db={n_vectors}, N_query={n_queries}, d={d}")
    print("=" * 70)

    print("[GT] Computing ground truth on original vectors ...")
    gt = compute_ground_truth(query_emb, db_emb, top_k=top_k)
    print(f"[GT] Done. gt shape={gt.shape}")

    q_mean = query_emb.mean(axis=0).astype(np.float32)
    db_mean = db_emb.mean(axis=0).astype(np.float32)

    gap = compute_gap(db_emb, query_emb)
    gap_norm = float(np.linalg.norm(gap))
    print(f"[PMC] gap norm = {gap_norm:.6f}")

    print(f"\n[vanilla_rabitq] Building index ...")
    vanilla_idx = build_rabitq(db_emb, nlist=nlist, seed=seed)
    bpv_rabitq = vanilla_idx.bytes_per_vec()

    print("  Sweeping nprobe:")
    records.extend(
        run_recall_qps_sweep(
            index=vanilla_idx,
            queries=query_emb,
            gt=gt,
            method="vanilla_rabitq",
            alpha=0.0,
            direction=direction,
            bpv=bpv_rabitq,
            n_vectors=n_vectors,
            d=d,
            seed=seed,
            nprobe_values=nprobe_values,
            top_k=top_k,
            n_warmup=n_warmup,
            n_timed=n_timed,
        )
    )

    print(f"\n[vanilla_rabitq_meanshift] Applying meanshift to queries ...")
    q_ms = l2_normalize((query_emb - q_mean + db_mean).astype(np.float32))

    print("  Sweeping nprobe (reusing vanilla index):")
    records.extend(
        run_recall_qps_sweep(
            index=vanilla_idx,
            queries=q_ms,
            gt=gt,
            method="vanilla_rabitq_meanshift",
            alpha=0.0,
            direction=direction,
            bpv=bpv_rabitq,
            n_vectors=n_vectors,
            d=d,
            seed=seed,
            nprobe_values=nprobe_values,
            top_k=top_k,
            n_warmup=n_warmup,
            n_timed=n_timed,
        )
    )

    alpha = 1.0
    print(f"\n[pmc_{alpha:.2f}] Building PMC-shifted index ...")
    db_shifted = shift_db_vectors(db_emb, gap, alpha=alpha)
    pmc_idx = build_rabitq(db_shifted, nlist=nlist, seed=seed)
    pmc_bpv = pmc_idx.bytes_per_vec()

    q_shifted = shift_query_vectors(query_emb, gap, alpha=alpha)
    print("  Sweeping nprobe:")
    records.extend(
        run_recall_qps_sweep(
            index=pmc_idx,
            queries=q_shifted,
            gt=gt,
            method=f"pmc_{alpha:.2f}",
            alpha=alpha,
            direction=direction,
            bpv=pmc_bpv,
            n_vectors=n_vectors,
            d=d,
            seed=seed,
            nprobe_values=nprobe_values,
            top_k=top_k,
            n_warmup=n_warmup,
            n_timed=n_timed,
        )
    )

    print(f"\n[ivfpq_meanshift_64B] Building IVFPQ index ...")
    try:
        ivfpq_idx, ivfpq_bpv = build_ivfpq(db_emb, target_bpv=64, nlist=nlist, seed=seed)
        q_ivfpq = l2_normalize((query_emb - q_mean + db_mean).astype(np.float32))

        print("  Sweeping nprobe:")
        records.extend(
            run_recall_qps_sweep(
                index=ivfpq_idx,
                queries=q_ivfpq,
                gt=gt,
                method="ivfpq_meanshift_64B",
                alpha=0.0,
                direction=direction,
                bpv=ivfpq_bpv,
                n_vectors=n_vectors,
                d=d,
                seed=seed,
                nprobe_values=nprobe_values,
                top_k=top_k,
                n_warmup=n_warmup,
                n_timed=n_timed,
            )
        )
    except Exception as exc:
        print(f"  [ivfpq_meanshift_64B] SKIP: {exc}")

    return records
