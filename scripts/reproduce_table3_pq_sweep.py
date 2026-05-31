"""
14_pq_alpha_sweep.py -- PMC alpha sweep for IVFPQ and OPQ+IVFPQ.

Tests alpha in {0, 0.25, 0.5, 0.75, 1.0} for PQ-based quantizers
to verify alpha=1 optimality (matching RaBitQ alpha sweep).

Output: results/pmc_pq_alpha_sweep_clip_mscoco_seed42.csv
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_V4_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "src").is_dir() and (parent / "config").is_dir()
)
if str(_V4_ROOT) not in sys.path:
    sys.path.insert(0, str(_V4_ROOT))

import numpy as np
import faiss

from src.core.pmc import compute_gap, shift_db_vectors, shift_query_vectors
from src.utils import ensure_float32_c
from src.runtime.config import CFG as V4_CFG
from src.core.metrics import recall_at_k, compute_ground_truth
from src.core.index_wrappers import build_ivfpq, build_opq_ivfpq, compute_nlist
from src.features.loader import load_npy

SEED = 42
NLIST = 64
NPROBE = 16
TOP_K = 100
ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.00]
DIRECTIONS = ["text->image", "image->text"]

FIELDNAMES = [
    "method", "alpha", "nprobe", "direction",
    "r1", "r10", "r100", "seed",
]


def main() -> None:
    np.random.seed(SEED)
    faiss.omp_set_num_threads(1)

    feat_dir = V4_CFG.features_dir
    img_path = feat_dir / f"mscoco_karpathy_val5k_clip_image_seed{SEED}.npy"
    txt_path = feat_dir / f"mscoco_karpathy_val5k_clip_text_seed{SEED}.npy"

    print(f"[pq_alpha_sweep] Loading features ...")
    image_emb = load_npy(img_path)
    text_emb = load_npy(txt_path)
    print(f"  image: {image_emb.shape}  text: {text_emb.shape}")

    modality_map = {"image": image_emb, "text": text_emb}
    all_records: List[Dict] = []

    for direction in DIRECTIONS:
        q_mod, db_mod = direction.split("->")
        db_emb = modality_map[db_mod]
        query_emb = modality_map[q_mod]
        N, d = db_emb.shape

        print(f"\n{'='*70}")
        print(f"Direction: {direction}  N={N}, d={d}")
        print("=" * 70)

        gt = compute_ground_truth(query_emb, db_emb, top_k=TOP_K)
        gap = compute_gap(db_emb, query_emb)
        print(f"  gap norm = {np.linalg.norm(gap):.6f}")

        nlist = compute_nlist(N)

        for alpha in ALPHAS:
            # Shift vectors
            if alpha == 0.0:
                db_use = db_emb
                q_use = query_emb
            else:
                db_use = shift_db_vectors(db_emb, gap, alpha=alpha)
                q_use = shift_query_vectors(query_emb, gap, alpha=alpha)

            # --- IVFPQ ---
            print(f"  [ivfpq alpha={alpha:.2f}] Building ...")
            try:
                ivfpq_idx, _ = build_ivfpq(db_use, target_bpv=64, nlist=nlist, seed=SEED)
                _, ids = ivfpq_idx.search(ensure_float32_c(q_use), TOP_K, NPROBE)
                r1 = recall_at_k(ids, gt, k=1)
                r10 = recall_at_k(ids, gt, k=10)
                r100 = recall_at_k(ids, gt, k=100)
                print(f"    IVFPQ   R@1={r1:.4f}  R@10={r10:.4f}  R@100={r100:.4f}")
                all_records.append({
                    "method": "ivfpq", "alpha": alpha, "nprobe": NPROBE,
                    "direction": direction, "r1": round(r1, 4),
                    "r10": round(r10, 4), "r100": round(r100, 4), "seed": SEED,
                })
            except Exception as e:
                print(f"    IVFPQ SKIP: {e}")

            # --- OPQ+IVFPQ ---
            print(f"  [opq alpha={alpha:.2f}] Building ...")
            try:
                opq_idx = build_opq_ivfpq(db_use, target_bpv=64, nlist=nlist, seed=SEED)
                _, ids = opq_idx.search(ensure_float32_c(q_use), TOP_K, NPROBE)
                r1 = recall_at_k(ids, gt, k=1)
                r10 = recall_at_k(ids, gt, k=10)
                r100 = recall_at_k(ids, gt, k=100)
                print(f"    OPQ     R@1={r1:.4f}  R@10={r10:.4f}  R@100={r100:.4f}")
                all_records.append({
                    "method": "opq_ivfpq", "alpha": alpha, "nprobe": NPROBE,
                    "direction": direction, "r1": round(r1, 4),
                    "r10": round(r10, 4), "r100": round(r100, 4), "seed": SEED,
                })
            except Exception as e:
                print(f"    OPQ SKIP: {e}")

    # Write CSV
    out_path = V4_CFG.results_dir / f"pmc_pq_alpha_sweep_clip_mscoco_seed{SEED}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(all_records)
    print(f"\n[pq_alpha_sweep] CSV -> {out_path}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: PQ/OPQ Alpha Sweep (nprobe=16)")
    print("=" * 80)
    print(f"{'Method':<12} {'Alpha':>5} {'Dir':<14} {'R@1':>6} {'R@10':>7} {'R@100':>7}")
    print("-" * 60)
    for r in all_records:
        print(f"{r['method']:<12} {r['alpha']:>5.2f} {r['direction']:<14} "
              f"{r['r1']:>6.4f} {r['r10']:>7.4f} {r['r100']:>7.4f}")


if __name__ == "__main__":
    main()
