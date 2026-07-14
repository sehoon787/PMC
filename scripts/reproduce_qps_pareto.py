"""
05_pmc_qps_pareto.py -- QPS + recall Pareto curve data for PMC-RaBitQ.

Sweeps nprobe = [1, 2, 4, 8, 16, 32, 64] for each method to generate
recall-vs-QPS Pareto curves needed for the CIKM paper.

Methods:
    vanilla_rabitq          (no correction)
    vanilla_rabitq_meanshift (negative control)
    pmc_1.00                (best PMC)
    ivfpq_meanshift_64B     (IVFPQ reference)

Metrics (all correct K-recall@K):
    R@1, R@10, R@100, 1NN_R@10, 1NN_R@100, QPS

Usage:
    python scripts/scale/05_pmc_qps_pareto.py

Output:
    results/pmc_qps_pareto_clip_mscoco_seed42.csv
    results/pmc_qps_pareto_imagebind_mscoco_seed42.csv
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_V4_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "src").is_dir() and (parent / "config").is_dir()
)
if str(_V4_ROOT) not in sys.path:
    sys.path.insert(0, str(_V4_ROOT))

import numpy as np
import faiss

from src.runtime.config import CFG as V4_CFG
from src.experiments.sweeps import run_pmc_qps_pareto_direction
from src.utils import l2_normalize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
TOP_K = 100
NLIST = 64
NPROBE_VALUES = [1, 2, 4, 8, 16, 32, 64]
BACKBONES = ["clip", "imagebind"]
DIRECTIONS = ["text->image", "image->text"]

# QPS timing
WARMUP_RUNS = 1
TIMING_RUNS = 5

FIELDNAMES = [
    "method", "alpha", "nprobe", "direction",
    "r1", "r10", "r100", "nn_r10", "nn_r100",
    "qps", "bytes_per_vec", "n_vectors", "d", "seed",
]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(records: List[Dict], backbone: str) -> None:
    print(f"\n{'='*120}")
    print(f"SUMMARY TABLE -- {backbone.upper()} MSCOCO val5k  (QPS Pareto)")
    print("=" * 120)

    header = (
        f"{'Method':<28} {'np':>4} {'Direction':<14} "
        f"{'R@1':>6} {'R@10':>6} {'R@100':>6} "
        f"{'1NN@10':>7} {'1NN@100':>8} {'QPS':>8} {'B/v':>4}"
    )
    print(header)
    print("-" * len(header))

    directions = list(dict.fromkeys(r["direction"] for r in records))
    for direction in directions:
        dir_recs = [r for r in records if r["direction"] == direction]
        methods = list(dict.fromkeys(r["method"] for r in dir_recs))
        for method in methods:
            m_recs = [r for r in dir_recs if r["method"] == method]
            for r in m_recs:
                print(
                    f"{r['method']:<28} {r['nprobe']:>4} {r['direction']:<14} "
                    f"{r['r1']:>6.4f} {r['r10']:>6.4f} {r['r100']:>6.4f} "
                    f"{r['nn_r10']:>7.4f} {r['nn_r100']:>8.4f} "
                    f"{r['qps']:>8.0f} {r['bytes_per_vec']:>4}"
                )
            print()


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[qps_pareto] CSV written -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    seed = SEED
    np.random.seed(seed)
    feat_dir = V4_CFG.features_dir

    # Single-thread for reproducible QPS
    faiss.omp_set_num_threads(1)
    print("[qps_pareto] faiss.omp_set_num_threads(1) for reproducible QPS")

    t_global = time.perf_counter()

    for backbone in BACKBONES:
        print(f"\n{'#'*70}")
        print(f"# BACKBONE: {backbone.upper()}")
        print(f"{'#'*70}")

        # Load features
        img_path = feat_dir / f"mscoco_karpathy_val5k_{backbone}_image_seed{seed}.npy"
        txt_path = feat_dir / f"mscoco_karpathy_val5k_{backbone}_text_seed{seed}.npy"

        if not img_path.exists() or not txt_path.exists():
            print(f"[qps_pareto] SKIP {backbone}: features not found")
            print(f"  image: {img_path}")
            print(f"  text:  {txt_path}")
            continue

        print(f"\n[qps_pareto] Loading {backbone} MSCOCO val5k features ...")
        image_emb = l2_normalize(np.load(str(img_path)).astype(np.float32))
        text_emb = l2_normalize(np.load(str(txt_path)).astype(np.float32))
        print(f"  image: {image_emb.shape}  text: {text_emb.shape}")

        N = image_emb.shape[0]
        nlist = min(NLIST, N // 10)
        nlist = max(nlist, 1)
        print(f"  nlist={nlist}")

        backbone_records: List[Dict] = []

        for direction in DIRECTIONS:
            q_mod, db_mod = direction.split("->")
            modality_map = {"image": image_emb, "text": text_emb}
            db_emb = modality_map[db_mod]
            query_emb = modality_map[q_mod]

            records = run_pmc_qps_pareto_direction(
                db_emb=db_emb,
                query_emb=query_emb,
                direction=direction,
                nlist=nlist,
                seed=seed,
                nprobe_values=NPROBE_VALUES,
                top_k=TOP_K,
                n_warmup=WARMUP_RUNS,
                n_timed=TIMING_RUNS,
            )
            backbone_records.extend(records)

        # Write CSV per backbone
        out_path = V4_CFG.results_dir / f"pmc_qps_pareto_{backbone}_mscoco_seed{seed}.csv"
        write_csv(backbone_records, out_path)
        print_summary(backbone_records, backbone)
        print(f"\n[qps_pareto] {backbone}: {len(backbone_records)} records -> {out_path}")

    dt_total = time.perf_counter() - t_global
    print(f"\n[qps_pareto] ALL DONE in {dt_total:.1f}s ({dt_total/60:.1f} min)")


if __name__ == "__main__":
    main()
