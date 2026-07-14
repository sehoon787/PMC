"""
12_clotho_r1_eval.py -- R@1/R@10/R@100 for Clotho v2 (default: all splits; --eval-only for eval split).

Methods evaluated:
  - vanilla_rabitq:    plain IndexIVFRaBitQFastScan (no shift)
  - pmc_1.00:          PMC with alpha=1.0 (DB + query shifted by gap)
  - vanilla_meanshift: vanilla index but queries shifted by -gap at search time

Output: results/<device>/pmc_clotho_r1_seed42.csv
"""

from __future__ import annotations

import csv
import os
import sys
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

from src.runtime.config import CFG
from src.experiments import run_three_method_direction
from src.core.index_wrappers import compute_nlist
from src.utils import l2_normalize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
TOP_K = 100
NPROBE = 16
PMC_ALPHA = 1.0

FEATURES_DIR = CFG.features_dir
RESULTS_DIR = CFG.results_dir

CLOTHO_EVAL_ONLY = "--eval-only" in sys.argv
_clotho_split = "clotho_eval" if CLOTHO_EVAL_ONLY else "clotho_all"
_clotho_label = "eval" if CLOTHO_EVAL_ONLY else "all"

AUDIO_PATH = FEATURES_DIR / f"{_clotho_split}_imagebind_audio_seed{SEED}.npy"
TEXT_PATH  = FEATURES_DIR / f"{_clotho_split}_imagebind_text_seed{SEED}.npy"

OUT_CSV = RESULTS_DIR / f"pmc_clotho_r1_seed{SEED}.csv"

FIELDNAMES = [
    "method", "alpha", "nprobe", "direction",
    "r1", "r10", "r100",
    "bytes_per_vec", "n_vectors", "d", "seed",
]


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(records: List[Dict]) -> None:
    print(f"\n{'='*90}")
    print(f"SUMMARY  (Clotho v2 {_clotho_label}, ImageBind, K-recall@K)")
    print("=" * 90)
    hdr = f"{'Method':<22} {'Direction':<16} {'R@1':>7} {'R@10':>7} {'R@100':>7} {'B/v':>4}"
    print(hdr)
    print("-" * len(hdr))
    for r in records:
        print(
            f"{r['method']:<22} {r['direction']:<16} "
            f"{r['r1']:>7.4f} {r['r10']:>7.4f} {r['r100']:>7.4f} "
            f"{r['bytes_per_vec']:>4}"
        )
    print()

    # Delta table
    print(f"{'='*90}")
    print("PMC DELTA vs vanilla_rabitq")
    print("=" * 90)
    directions = list(dict.fromkeys(r["direction"] for r in records))
    for direction in directions:
        dir_recs = [r for r in records if r["direction"] == direction]
        van = next((r for r in dir_recs if r["method"] == "vanilla_rabitq"), None)
        if van is None:
            continue
        print(f"\n  {direction}  vanilla: R@1={van['r1']:.4f}  R@10={van['r10']:.4f}  R@100={van['r100']:.4f}")
        for r in dir_recs:
            if r["method"] == "vanilla_rabitq":
                continue
            d1   = r["r1"]   - van["r1"]
            d10  = r["r10"]  - van["r10"]
            d100 = r["r100"] - van["r100"]
            print(
                f"    {r['method']:<20} "
                f"R@1={r['r1']:.4f} (Δ={d1:+.4f})  "
                f"R@10={r['r10']:.4f} (Δ={d10:+.4f})  "
                f"R@100={r['r100']:.4f} (Δ={d100:+.4f})"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    faiss.omp_set_num_threads(1)

    print(f"[clotho_r1_eval] seed={SEED}  nprobe={NPROBE}  top_k={TOP_K}  split={_clotho_label}")
    print(f"[clotho_r1_eval] audio: {AUDIO_PATH}")
    print(f"[clotho_r1_eval] text:  {TEXT_PATH}")

    for p in (AUDIO_PATH, TEXT_PATH):
        if not p.exists():
            print(f"[clotho_r1_eval] ERROR: not found: {p}")
            sys.exit(1)

    # Load and L2-normalize
    audio_raw = np.load(str(AUDIO_PATH)).astype(np.float32)
    text_raw  = np.load(str(TEXT_PATH)).astype(np.float32)
    audio_emb = l2_normalize(audio_raw)
    text_emb  = l2_normalize(text_raw)
    print(f"  audio: {audio_emb.shape}  text: {text_emb.shape}")

    N = audio_emb.shape[0]
    nlist = compute_nlist(N)
    print(f"  nlist={nlist} (N={N})")

    all_records: List[Dict] = []

    # text -> audio
    all_records.extend(
        run_three_method_direction(
            db_emb=audio_emb,
            query_emb=text_emb,
            direction="text->audio",
            nlist=nlist,
            seed=SEED,
            nprobe=NPROBE,
            top_k=TOP_K,
            pmc_alpha=PMC_ALPHA,
            include_qps=False,
        )
    )

    # audio -> text
    all_records.extend(
        run_three_method_direction(
            db_emb=text_emb,
            query_emb=audio_emb,
            direction="audio->text",
            nlist=nlist,
            seed=SEED,
            nprobe=NPROBE,
            top_k=TOP_K,
            pmc_alpha=PMC_ALPHA,
            include_qps=False,
        )
    )

    # Write CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n[clotho_r1_eval] CSV written -> {OUT_CSV}")

    print_summary(all_records)
    print(f"[clotho_r1_eval] Done. {len(all_records)} records.")


if __name__ == "__main__":
    main()
