"""
11_audiocaps_r1_reeval.py -- Correct R@1/R@10/R@100 for AudioCaps (ImageBind).

The previous script (03_pmc_r10_reeval.py) computed R@1 with the OLD/buggy formula:
    recall_at_1 = |retrieved_1 ∩ gt_100| / 100  ≈ 0.01  (meaningless)

This script uses the CORRECT K-recall@K formula for all metrics:
    R@K = |retrieved_K ∩ gt_K| / K

Methods evaluated:
  - vanilla_rabitq:    plain IndexIVFRaBitQFastScan (no shift)
  - pmc_1.00:          PMC with alpha=1.0 (DB + query shifted by gap)
  - vanilla_meanshift: vanilla index but queries shifted by -gap at search time

Cross-validation: R@10 and R@100 should match pmc_reeval_imagebind_audiocaps_seed42.csv
(recall_at_10_standard and recall_at_100 columns).

Output: results/pmc_audiocaps_r1_seed42.csv
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

AUDIO_PATH = FEATURES_DIR / f"audiocaps_test_imagebind_audio_seed{SEED}.npy"
TEXT_PATH  = FEATURES_DIR / f"audiocaps_test_imagebind_text_seed{SEED}.npy"

OUT_CSV = RESULTS_DIR / f"pmc_audiocaps_r1_seed{SEED}.csv"

FIELDNAMES = [
    "method", "alpha", "nprobe", "direction",
    "r1", "r10", "r100",
    "bytes_per_vec", "n_vectors", "d", "seed",
]

# Expected values from pmc_reeval_imagebind_audiocaps_seed42.csv for cross-validation
# recall_at_10_standard and recall_at_100 (which is also correct since k=100)
_EXPECTED = {
    # (method, direction): (r10, r100)
    ("vanilla_rabitq", "text->audio"): (0.5934, 0.7335),
    ("pmc_1.00",       "text->audio"): (0.6201, 0.7482),
    ("vanilla_rabitq", "audio->text"): (0.4749, 0.6364),
    ("pmc_1.00",       "audio->text"): (0.5766, 0.7115),
}
_TOLERANCE = 0.02  # allow ±2% difference (index is stochastic)


def cross_validate(method: str, direction: str, r10: float, r100: float) -> None:
    """Warn if computed metrics deviate from previously confirmed values."""
    key = (method, direction)
    if key not in _EXPECTED:
        return
    exp_r10, exp_r100 = _EXPECTED[key]
    ok_r10  = abs(r10  - exp_r10)  <= _TOLERANCE
    ok_r100 = abs(r100 - exp_r100) <= _TOLERANCE
    status = "OK" if (ok_r10 and ok_r100) else "MISMATCH"
    print(
        f"  [xval {status}] {method} {direction}: "
        f"R@10={r10:.4f} (exp={exp_r10:.4f}, diff={r10-exp_r10:+.4f})  "
        f"R@100={r100:.4f} (exp={exp_r100:.4f}, diff={r100-exp_r100:+.4f})"
    )


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(records: List[Dict]) -> None:
    print(f"\n{'='*90}")
    print("SUMMARY  (AudioCaps, ImageBind, corrected K-recall@K)")
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
            d1  = r["r1"]   - van["r1"]
            d10 = r["r10"]  - van["r10"]
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
    print(f"[audiocaps_r1_reeval] seed={SEED}  nprobe={NPROBE}  top_k={TOP_K}")
    print(f"[audiocaps_r1_reeval] audio: {AUDIO_PATH}")
    print(f"[audiocaps_r1_reeval] text:  {TEXT_PATH}")

    for p in (AUDIO_PATH, TEXT_PATH):
        if not p.exists():
            print(f"[audiocaps_r1_reeval] ERROR: not found: {p}")
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
            metric_hook=cross_validate,
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
            metric_hook=cross_validate,
        )
    )

    # Write CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n[audiocaps_r1_reeval] CSV written -> {OUT_CSV}")

    print_summary(all_records)
    print(f"[audiocaps_r1_reeval] Done. {len(all_records)} records.")


if __name__ == "__main__":
    main()
