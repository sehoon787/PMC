"""Shared paired-modality recall evaluation helpers."""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.core.index_wrappers import build_vanilla_rabitq as build_rabitq_index
from src.core.metrics import compute_ground_truth, nn_recall_at_k, recall_at_k
from src.core.pmc import compute_gap, shift_db_vectors, shift_query_vectors
from src.utils import measure_qps


def recall_at_k_old(retrieved_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    """Old compatibility metric: |retrieved_K intersect gt_ALL| / len(gt_ALL)."""
    n_queries = len(retrieved_ids)
    total = 0.0
    for i in range(n_queries):
        topk = set(int(x) for x in retrieved_ids[i, :k] if x >= 0)
        gt = set(int(x) for x in gt_ids[i] if x >= 0)
        if not gt:
            continue
        total += len(topk & gt) / len(gt)
    return total / n_queries if n_queries > 0 else 0.0


def _corrected_metrics(
    *,
    index,
    queries: np.ndarray,
    gt: np.ndarray,
    nprobe: int,
    top_k: int,
) -> dict[str, float]:
    """Run search and compute script-03-compatible metric columns."""
    _, retrieved = index.search(queries, top_k=top_k, nprobe=nprobe)
    return {
        "r1": round(recall_at_k_old(retrieved, gt, k=1), 4),
        "r10": round(recall_at_k_old(retrieved, gt, k=10), 4),
        "r100": round(recall_at_k_old(retrieved, gt, k=100), 4),
        "recall_at_10_standard": round(recall_at_k(retrieved, gt, k=10), 4),
        "nn_recall_at_10": round(nn_recall_at_k(retrieved, gt, k=10), 4),
        "nn_recall_at_100": round(nn_recall_at_k(retrieved, gt, k=100), 4),
    }


def run_corrected_pmc_direction(
    *,
    db_emb: np.ndarray,
    query_emb: np.ndarray,
    direction: str,
    nlist: int,
    seed: int,
    nprobe: int,
    top_k: int,
    pmc_alphas: list[float],
) -> list[dict]:
    """Evaluate vanilla and PMC alphas with script-03 corrected metrics."""
    n_vectors, d = db_emb.shape
    n_query = query_emb.shape[0]
    records: list[dict] = []

    print(f"\n{'='*70}")
    print(f"Direction: {direction}  N_db={n_vectors}, N_query={n_query}, d={d}")
    print("=" * 70)

    print("[GT] Computing ground truth on original vectors ...")
    gt = compute_ground_truth(query_emb, db_emb, top_k=top_k)
    print(f"[GT] Done. gt shape={gt.shape}")

    print(f"\n[vanilla_rabitq] Building ... (nprobe={nprobe})")
    vanilla_idx = build_rabitq_index(db_emb, nlist=nlist, seed=seed)
    bytes_per_vec = vanilla_idx.bytes_per_vec()
    metrics = _corrected_metrics(
        index=vanilla_idx,
        queries=query_emb,
        gt=gt,
        nprobe=nprobe,
        top_k=top_k,
    )
    print(
        f"  vanilla_rabitq  R@10_old={metrics['r10']:.4f}  "
        f"R@10_std={metrics['recall_at_10_standard']:.4f}  "
        f"1NN@10={metrics['nn_recall_at_10']:.4f}  "
        f"1NN@100={metrics['nn_recall_at_100']:.4f}"
    )
    records.append({
        "method": "vanilla_rabitq",
        "alpha": 0.0,
        "nprobe": nprobe,
        "direction": direction,
        **metrics,
        "bytes_per_vec": bytes_per_vec,
        "n_vectors": n_vectors,
        "d": d,
        "seed": seed,
    })

    gap = compute_gap(db_emb, query_emb)
    gap_norm = float(np.linalg.norm(gap))
    print(f"\n[PMC] gap norm = {gap_norm:.6f}")

    for alpha in pmc_alphas:
        print(f"\n[PMC alpha={alpha:.2f}] Building ...")
        db_shifted = shift_db_vectors(db_emb, gap, alpha=alpha)
        pmc_idx = build_rabitq_index(db_shifted, nlist=nlist, seed=seed)
        q_shifted = shift_query_vectors(query_emb, gap, alpha=alpha)
        metrics = _corrected_metrics(
            index=pmc_idx,
            queries=q_shifted,
            gt=gt,
            nprobe=nprobe,
            top_k=top_k,
        )
        delta = metrics["recall_at_10_standard"] - records[0]["recall_at_10_standard"]
        print(
            f"  pmc_{alpha:.2f}  R@10_old={metrics['r10']:.4f}  "
            f"R@10_std={metrics['recall_at_10_standard']:.4f}  "
            f"1NN@10={metrics['nn_recall_at_10']:.4f}  "
            f"1NN@100={metrics['nn_recall_at_100']:.4f}  "
            f"delta_R@10_std={delta:+.4f}"
        )
        records.append({
            "method": f"pmc_{alpha:.2f}",
            "alpha": alpha,
            "nprobe": nprobe,
            "direction": direction,
            **metrics,
            "bytes_per_vec": pmc_idx.bytes_per_vec(),
            "n_vectors": n_vectors,
            "d": d,
            "seed": seed,
        })

    return records


def run_three_method_direction(
    *,
    db_emb: np.ndarray,
    query_emb: np.ndarray,
    direction: str,
    nlist: int,
    seed: int,
    nprobe: int,
    top_k: int,
    pmc_alpha: float = 1.0,
    include_qps: bool = False,
    metric_hook: Callable[[str, str, float, float], None] | None = None,
) -> list[dict]:
    """Evaluate vanilla, PMC, and meanshift for one direction."""
    n_vectors, d = db_emb.shape
    n_query = query_emb.shape[0]
    records: list[dict] = []

    print(f"\n{'='*70}")
    print(f"Direction: {direction}  N_db={n_vectors}  N_query={n_query}  d={d}")
    print("=" * 70)

    print("[GT] Computing brute-force ground truth ...")
    gt = compute_ground_truth(query_emb, db_emb, top_k=top_k)
    print(f"[GT] Done.  gt.shape={gt.shape}")

    gap = compute_gap(db_emb, query_emb)
    gap_norm = float(np.linalg.norm(gap))
    print(f"[PMC] gap norm = {gap_norm:.6f}")

    print("\n[vanilla] Building ...")
    vanilla_idx = build_rabitq_index(db_emb, nlist=nlist, seed=seed)
    bytes_per_vec = vanilla_idx.bytes_per_vec()
    _, vanilla_retrieved = vanilla_idx.search(query_emb, top_k=top_k, nprobe=nprobe)
    vanilla_r1 = recall_at_k(vanilla_retrieved, gt, k=1)
    vanilla_r10 = recall_at_k(vanilla_retrieved, gt, k=10)
    vanilla_r100 = recall_at_k(vanilla_retrieved, gt, k=100)
    vanilla_qps = None
    if include_qps:
        _, _, vanilla_qps = measure_qps(vanilla_idx, query_emb, top_k=top_k, nprobe=nprobe)
        print(
            f"  vanilla  R@1={vanilla_r1:.4f}  R@10={vanilla_r10:.4f}  "
            f"R@100={vanilla_r100:.4f}  QPS={vanilla_qps:.1f}"
        )
    else:
        print(f"  vanilla_rabitq  R@1={vanilla_r1:.4f}  R@10={vanilla_r10:.4f}  R@100={vanilla_r100:.4f}")
    if metric_hook is not None:
        metric_hook("vanilla_rabitq", direction, vanilla_r10, vanilla_r100)

    vanilla_record = {
        "method": "vanilla_rabitq",
        "alpha": 0.0,
        "nprobe": nprobe,
        "direction": direction,
        "r1": round(vanilla_r1, 4),
        "r10": round(vanilla_r10, 4),
        "r100": round(vanilla_r100, 4),
        "bytes_per_vec": bytes_per_vec,
        "n_vectors": n_vectors,
        "d": d,
        "seed": seed,
    }
    if include_qps:
        vanilla_record["qps"] = round(float(vanilla_qps), 2)
    records.append(vanilla_record)

    print(f"\n[pmc_{pmc_alpha:.2f}] Building ...")
    db_shifted = shift_db_vectors(db_emb, gap, alpha=pmc_alpha)
    pmc_idx = build_rabitq_index(db_shifted, nlist=nlist, seed=seed)
    q_shifted = shift_query_vectors(query_emb, gap, alpha=pmc_alpha)
    _, pmc_retrieved = pmc_idx.search(q_shifted, top_k=top_k, nprobe=nprobe)
    pmc_r1 = recall_at_k(pmc_retrieved, gt, k=1)
    pmc_r10 = recall_at_k(pmc_retrieved, gt, k=10)
    pmc_r100 = recall_at_k(pmc_retrieved, gt, k=100)
    pmc_qps = None
    if include_qps:
        _, _, pmc_qps = measure_qps(pmc_idx, q_shifted, top_k=top_k, nprobe=nprobe)
        print(
            f"  pmc_{pmc_alpha:.2f}  R@1={pmc_r1:.4f}  R@10={pmc_r10:.4f}  "
            f"R@100={pmc_r100:.4f}  QPS={pmc_qps:.1f}  delta_R@10={pmc_r10 - vanilla_r10:+.4f}"
        )
    else:
        print(
            f"  pmc_{pmc_alpha:.2f}  R@1={pmc_r1:.4f}  R@10={pmc_r10:.4f}  "
            f"R@100={pmc_r100:.4f}  delta_R@10={pmc_r10 - vanilla_r10:+.4f}"
        )
    if metric_hook is not None:
        metric_hook(f"pmc_{pmc_alpha:.2f}", direction, pmc_r10, pmc_r100)

    pmc_record = {
        "method": f"pmc_{pmc_alpha:.2f}",
        "alpha": pmc_alpha,
        "nprobe": nprobe,
        "direction": direction,
        "r1": round(pmc_r1, 4),
        "r10": round(pmc_r10, 4),
        "r100": round(pmc_r100, 4),
        "bytes_per_vec": pmc_idx.bytes_per_vec(),
        "n_vectors": n_vectors,
        "d": d,
        "seed": seed,
    }
    if include_qps:
        pmc_record["qps"] = round(float(pmc_qps), 2)
    records.append(pmc_record)

    print("\n[vanilla_meanshift] Shifting queries by -gap and searching vanilla index ...")
    q_ms = shift_query_vectors(query_emb, gap, alpha=0.0)
    _, ms_retrieved = vanilla_idx.search(q_ms, top_k=top_k, nprobe=nprobe)
    ms_r1 = recall_at_k(ms_retrieved, gt, k=1)
    ms_r10 = recall_at_k(ms_retrieved, gt, k=10)
    ms_r100 = recall_at_k(ms_retrieved, gt, k=100)
    ms_qps = None
    if include_qps:
        _, _, ms_qps = measure_qps(vanilla_idx, q_ms, top_k=top_k, nprobe=nprobe)
        print(
            f"  vanilla_meanshift  R@1={ms_r1:.4f}  R@10={ms_r10:.4f}  "
            f"R@100={ms_r100:.4f}  QPS={ms_qps:.1f}  delta_R@10={ms_r10 - vanilla_r10:+.4f}"
        )
    else:
        print(
            f"  vanilla_meanshift  R@1={ms_r1:.4f}  R@10={ms_r10:.4f}  "
            f"R@100={ms_r100:.4f}  delta_R@10={ms_r10 - vanilla_r10:+.4f}"
        )

    ms_record = {
        "method": "vanilla_meanshift",
        "alpha": 0.0,
        "nprobe": nprobe,
        "direction": direction,
        "r1": round(ms_r1, 4),
        "r10": round(ms_r10, 4),
        "r100": round(ms_r100, 4),
        "bytes_per_vec": bytes_per_vec,
        "n_vectors": n_vectors,
        "d": d,
        "seed": seed,
    }
    if include_qps:
        ms_record["qps"] = round(float(ms_qps), 2)
    records.append(ms_record)

    return records
