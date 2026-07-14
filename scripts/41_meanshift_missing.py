"""
41_meanshift_missing.py -- Meanshift-only eval for MSCOCO CL-L and AudioCaps.

Fills in the missing meanshift rows for:
  - MSCOCO_CLIP-L (CLIP-L/14, d=768): seeds 42, 123, 456, both directions
  - AudioCaps_std (ImageBind, d=1024): seeds 42, 123, 456, both directions

Vanilla and PMC rows for these conditions already exist in multiseed_rabitq_summary.csv.
This script builds the vanilla index (required to search with shifted queries) but
records ONLY the meanshift method.

Output:
  - Appends meanshift rows to results/multiseed_rabitq_summary.csv
  - Writes standalone results/meanshift_missing_results.csv for verification

Protocol is identical to 40_multiseed_rabitq.py.
"""

from __future__ import annotations

import csv
import json
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
import pandas as pd
import faiss

from src.runtime.config import CFG
from src.core.index_wrappers import build_vanilla_rabitq, compute_nlist
from src.core.metrics import (
    recall_at_k, compute_ground_truth,
    recall_at_k_single_gt, recall_at_k_multi_gt,
)
from src.utils import l2_normalize

# ---------------------------------------------------------------------------
# Constants (identical to 40_multiseed_rabitq.py)
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 456]
TOP_K = 100
NPROBE = 16

FEATURES_DIR = CFG.features_dir
RESULTS_DIR = CFG.results_dir
SUMMARY_CSV = RESULTS_DIR / "multiseed_rabitq_summary.csv"
STANDALONE_CSV = RESULTS_DIR / "meanshift_missing_results.csv"

FIELDNAMES = [
    "condition", "backbone", "dataset", "direction", "seed",
    "method", "r10", "r100", "delta_r10", "delta_r100",
    "gt_protocol",
]


# ---------------------------------------------------------------------------
# Symmetric meanshift eval (MSCOCO protocol)
# ---------------------------------------------------------------------------

def eval_meanshift_symmetric(
    db_emb: np.ndarray,
    query_emb: np.ndarray,
    direction: str,
    condition: str,
    backbone: str,
    dataset: str,
    seed: int,
) -> Dict:
    """Build vanilla index and evaluate meanshift on symmetric paired data."""
    n_db = db_emb.shape[0]
    nlist = compute_nlist(n_db)

    gt = compute_ground_truth(query_emb, db_emb, top_k=TOP_K)

    vanilla_idx = build_vanilla_rabitq(db_emb, nlist=nlist, seed=seed)

    # Vanilla recall (needed only to compute delta fields)
    _, van_ret = vanilla_idx.search(query_emb, top_k=TOP_K, nprobe=NPROBE)
    van_r10 = recall_at_k(van_ret, gt, k=10)
    van_r100 = recall_at_k(van_ret, gt, k=100)

    # Meanshift: shift queries toward DB centroid
    q_mean = query_emb.mean(axis=0)
    db_mean = db_emb.mean(axis=0)
    q_ms = l2_normalize((query_emb - q_mean + db_mean).astype(np.float32))
    _, ms_ret = vanilla_idx.search(q_ms, top_k=TOP_K, nprobe=NPROBE)
    ms_r10 = recall_at_k(ms_ret, gt, k=10)
    ms_r100 = recall_at_k(ms_ret, gt, k=100)

    print(
        f"  [{condition}] seed={seed} {direction}: "
        f"vanilla R@10={van_r10:.4f} R@100={van_r100:.4f}  "
        f"meanshift R@10={ms_r10:.4f} R@100={ms_r100:.4f}  "
        f"Δ10={ms_r10 - van_r10:+.4f} Δ100={ms_r100 - van_r100:+.4f}"
    )

    return {
        "condition": condition, "backbone": backbone, "dataset": dataset,
        "direction": direction, "seed": seed, "method": "meanshift",
        "r10": round(ms_r10, 6), "r100": round(ms_r100, 6),
        "delta_r10": round(ms_r10 - van_r10, 6),
        "delta_r100": round(ms_r100 - van_r100, 6),
        "gt_protocol": "original_exact_ip",
    }


# ---------------------------------------------------------------------------
# AudioCaps standard meanshift eval (asymmetric protocol)
# ---------------------------------------------------------------------------

def build_audiocaps_clip_mapping() -> tuple:
    """Reconstruct clip<->caption mapping from test.csv and feature sidecar."""
    audio_path = FEATURES_DIR / "audiocaps_test_imagebind_audio_seed42.npy"
    sidecar = audio_path.with_suffix(".json")
    with open(sidecar, "r") as f:
        valid_aids = json.load(f)
    valid_set = set(valid_aids)

    df = pd.read_csv(str(CFG.audiocaps_metadata_csv))
    clip_groups = df.groupby(["youtube_id", "start_time"])
    clips = []
    for (yt, st), group in clip_groups:
        aids_valid = sorted(
            int(r["audiocap_id"]) for _, r in group.iterrows()
            if int(r["audiocap_id"]) in valid_set
        )
        if aids_valid:
            clips.append(((yt, st), aids_valid))
    clips.sort(key=lambda x: x[1][0])
    return clips, valid_aids


def eval_meanshift_audiocaps(seed: int) -> List[Dict]:
    """Build vanilla index and evaluate meanshift on AudioCaps asymmetric protocol."""
    clips, valid_aids = build_audiocaps_clip_mapping()
    n_clips = len(clips)
    aid_to_pos = {aid: pos for pos, aid in enumerate(valid_aids)}

    clip_audio_positions = []
    text_positions_ordered = []
    offsets = [0]
    for _, aids in clips:
        clip_audio_positions.append(aid_to_pos[aids[0]])
        for a in aids:
            text_positions_ordered.append(aid_to_pos[a])
        offsets.append(offsets[-1] + len(aids))

    audio_all = l2_normalize(
        np.load(str(FEATURES_DIR / "audiocaps_test_imagebind_audio_seed42.npy")).astype(np.float32)
    )
    text_all = l2_normalize(
        np.load(str(FEATURES_DIR / "audiocaps_test_imagebind_text_seed42.npy")).astype(np.float32)
    )
    audio_std = audio_all[clip_audio_positions]
    text_std = text_all[text_positions_ordered]

    # GT construction (identical to 40_multiseed_rabitq.py)
    gt_t2a = np.zeros(len(text_positions_ordered), dtype=np.int64)
    for j in range(n_clips):
        gt_t2a[offsets[j]:offsets[j + 1]] = j
    gt_a2t = [set(range(offsets[j], offsets[j + 1])) for j in range(n_clips)]

    records: List[Dict] = []

    for direction, db_emb, query_emb, gt, recall_fn in [
        ("text->audio", audio_std, text_std, gt_t2a, recall_at_k_single_gt),
        ("audio->text", text_std, audio_std, gt_a2t, recall_at_k_multi_gt),
    ]:
        n_db = db_emb.shape[0]
        nlist = compute_nlist(n_db)

        vanilla_idx = build_vanilla_rabitq(db_emb, nlist=nlist, seed=seed)

        # Vanilla recall (for delta computation)
        _, van_ret = vanilla_idx.search(query_emb, top_k=TOP_K, nprobe=NPROBE)
        van_r10 = recall_fn(van_ret, gt, 10)
        van_r100 = recall_fn(van_ret, gt, 100)

        # Meanshift: shift queries toward DB centroid
        q_mean = query_emb.mean(axis=0)
        db_mean = db_emb.mean(axis=0)
        q_ms = l2_normalize((query_emb - q_mean + db_mean).astype(np.float32))
        _, ms_ret = vanilla_idx.search(q_ms, top_k=TOP_K, nprobe=NPROBE)
        ms_r10 = recall_fn(ms_ret, gt, 10)
        ms_r100 = recall_fn(ms_ret, gt, 100)

        print(
            f"  [AudioCaps_std] seed={seed} {direction}: "
            f"vanilla R@10={van_r10:.4f} R@100={van_r100:.4f}  "
            f"meanshift R@10={ms_r10:.4f} R@100={ms_r100:.4f}  "
            f"Δ10={ms_r10 - van_r10:+.4f} Δ100={ms_r100 - van_r100:+.4f}"
        )

        records.append({
            "condition": "AudioCaps_std", "backbone": "ImageBind",
            "dataset": "AudioCaps", "direction": direction,
            "seed": seed, "method": "meanshift",
            "r10": round(ms_r10, 6), "r100": round(ms_r100, 6),
            "delta_r10": round(ms_r10 - van_r10, 6),
            "delta_r100": round(ms_r100 - van_r100, 6),
            "gt_protocol": "original_exact_ip",
        })

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    faiss.omp_set_num_threads(1)
    t0 = time.time()
    all_records: List[Dict] = []

    # -----------------------------------------------------------------------
    # MSCOCO CLIP-L/14 (symmetric)
    # -----------------------------------------------------------------------
    db_file = "mscoco_karpathy_val5k_clip-l_image_seed42.npy"
    query_file = "mscoco_karpathy_val5k_clip-l_text_seed42.npy"
    db_path = FEATURES_DIR / db_file
    query_path = FEATURES_DIR / query_file

    if not db_path.exists() or not query_path.exists():
        print(f"\n[SKIP] MSCOCO_CLIP-L: features not found ({db_file})")
    else:
        image_emb = l2_normalize(np.load(str(db_path)).astype(np.float32))
        text_emb = l2_normalize(np.load(str(query_path)).astype(np.float32))

        print(f"\n{'='*70}")
        print("Condition: MSCOCO_CLIP-L  (CLIP-L/14 / MSCOCO)")
        print(f"  DB: {db_file}  shape={image_emb.shape}")
        print(f"  Query: {query_file}  shape={text_emb.shape}")
        print("=" * 70)

        for direction, db_emb, query_emb in [
            ("text->image", image_emb, text_emb),
            ("image->text", text_emb, image_emb),
        ]:
            for seed in SEEDS:
                rec = eval_meanshift_symmetric(
                    db_emb=db_emb,
                    query_emb=query_emb,
                    direction=direction,
                    condition="MSCOCO_CLIP-L",
                    backbone="CLIP-L/14",
                    dataset="MSCOCO",
                    seed=seed,
                )
                all_records.append(rec)

    # -----------------------------------------------------------------------
    # AudioCaps standard (asymmetric)
    # -----------------------------------------------------------------------
    audio_path = FEATURES_DIR / "audiocaps_test_imagebind_audio_seed42.npy"
    if not audio_path.exists():
        print("\n[SKIP] AudioCaps: features not found (audiocaps_test_imagebind_audio_seed42.npy)")
    else:
        print(f"\n{'='*70}")
        print("Condition: AudioCaps_std (ImageBind / AudioCaps standard protocol)")
        print("=" * 70)
        for seed in SEEDS:
            recs = eval_meanshift_audiocaps(seed)
            all_records.extend(recs)

    if not all_records:
        print("\n[WARNING] No records produced — nothing to write.")
        return

    # -----------------------------------------------------------------------
    # Write standalone CSV (always overwrite for clean verification)
    # -----------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STANDALONE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n[meanshift_missing] Standalone CSV -> {STANDALONE_CSV}")

    # -----------------------------------------------------------------------
    # Append to existing summary CSV
    # -----------------------------------------------------------------------
    summary_exists = SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not summary_exists:
            writer.writeheader()
        writer.writerows(all_records)
    print(f"[meanshift_missing] Appended {len(all_records)} rows -> {SUMMARY_CSV}")

    elapsed = time.time() - t0
    print(f"\n[meanshift_missing] Total time: {elapsed:.1f}s  Records: {len(all_records)}")


if __name__ == "__main__":
    main()
