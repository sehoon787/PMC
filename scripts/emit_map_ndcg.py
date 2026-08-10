"""emit_map_ndcg.py -- mAP/nDCG at the Table-2 operating point (exploratory).

FAISS-bound builder: reads feature caches from data/features/ and builds indexes,
so it cannot run from a bare clone. Named emit_* per the convention in
docs/PAPER_RESULT_PROVENANCE.md (reproduce_* read committed CSVs only).

Answers the reviewer question "what do rank-aware metrics say?" without
redefining relevance.  Relevance is the exact inner-product top-100 over the
same corpus on the ORIGINAL (unshifted) embeddings -- the identical ground
truth Table 2 uses for R@100.  mAP/nDCG therefore measure how faithfully a
compressed index reproduces exact search ranking (approximation fidelity),
not caption<->image semantics.

Methods (per dataset/encoder/direction cell):
  - vanilla_rabitq            plain IndexIVFRaBitQFastScan, unshifted queries
  - vanilla_rabitq_meanshift  same index, queries shifted by -gap (alpha=0)
  - pmc_1.00                  PMC with alpha=1.0 (DB shifted, queries shifted)

Index config matches Table 2: IVF-RaBitQFastScan, nlist = round(sqrt(N_db))
(the paper's "sqrtN" setting), seed 42, single FAISS thread, and the matched
budget nprobe = round_half_up(nlist / 4) snapped to the swept nprobe grid.

AudioCaps caveat: Table 2's AudioCaps row is the only one evaluated under the
standard caption<->clip protocol (single/multi semantic ground truth) on the
deduplicated 672-clip / 3346-caption subsets.  This script keeps those exact
subsets and index settings, but scores them with exact-IP top-100 ground truth
like every other row, so the frame stays uniform.  The paper's semantic R@100
is recomputed alongside purely as a harness cross-check and printed to stdout.

NOTE: exploratory run -- not a camera-ready artifact.

Output: results/map_ndcg_seed42.csv
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from decimal import Decimal, ROUND_HALF_UP
from math import sqrt
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from src.core.index_wrappers import build_vanilla_rabitq
from src.core.metrics import (
    compute_ground_truth,
    map_at_k,
    ndcg_at_k,
    recall_at_k,
    recall_at_k_multi_gt,
    recall_at_k_single_gt,
)
from src.core.pmc import build_pmc_rabitq_index, search_pmc, shift_query_vectors
from src.utils import l2_normalize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
TOP_K = 100
PMC_ALPHA = 1.0
KS = (10, 100)
NPROBE_BASE = [1, 2, 4, 8, 16, 32, 64, 128]
# FastScan allocates per-call scratch that degrades superlinearly (and faults
# outright on 31k-query batches), so queries are searched in fixed batches.
# Search is per-query independent, so batching leaves results unchanged.
QUERY_BATCH = 2048

FEATURES_DIR = CFG.features_dir
RESULTS_DIR = CFG.results_dir
OUT_CSV = RESULTS_DIR / f"map_ndcg_seed{SEED}.csv"
TAB2_CSV = RESULTS_DIR / "tab2_main_reproduced.csv"

FIELDNAMES = [
    "dataset", "backbone", "direction", "nlist", "nprobe", "method",
    "map@10", "map@100", "ndcg@10", "ndcg@100",
    "r10", "r100", "n_db", "d", "seed",
]

# (dataset, backbone, db_modality, query_modality, db_file, query_file) per direction pair.
IMAGE_CONDITIONS = [
    ("MSCOCO", "CLIP-B/32", "image", "text",
     "mscoco_karpathy_val5k_clip_image_seed42.npy",
     "mscoco_karpathy_val5k_clip_text_seed42.npy"),
    ("MSCOCO", "CLIP-L/14", "image", "text",
     "mscoco_karpathy_val5k_clip-l_image_seed42.npy",
     "mscoco_karpathy_val5k_clip-l_text_seed42.npy"),
    ("MSCOCO", "ImageBind", "image", "text",
     "mscoco_karpathy_val5k_imagebind_image_seed42.npy",
     "mscoco_karpathy_val5k_imagebind_text_seed42.npy"),
    ("Flickr30K-full", "CLIP-L/14", "image", "text",
     "flickr30k_full_clip-l_image_seed42.npy",
     "flickr30k_full_clip-l_text_seed42.npy"),
    ("Clotho-all", "ImageBind", "audio", "text",
     "clotho_all_imagebind_audio_seed42.npy",
     "clotho_all_imagebind_text_seed42.npy"),
]

AUDIOCAPS_AUDIO_FILE = "audiocaps_test_imagebind_audio_seed42.npy"
AUDIOCAPS_TEXT_FILE = "audiocaps_test_imagebind_text_seed42.npy"

# Table 2 row identity -> (tab2 Dataset, tab2 Enc); "q" columns are the
# text->X direction, "db" columns are the X->text direction.
TAB2_KEY = {
    ("MSCOCO", "CLIP-B/32"): ("MSCOCO", "CLIP"),
    ("MSCOCO", "CLIP-L/14"): ("MSCOCO", "CL-L"),
    ("MSCOCO", "ImageBind"): ("MSCOCO", "IB"),
    ("Flickr30K-full", "CLIP-L/14"): ("Flickr30K", "CL-L"),
    ("Clotho-all", "ImageBind"): ("Clotho", "IB"),
    ("AudioCaps", "ImageBind"): ("AudioCaps", "IB"),
}
TAB2_METHOD_COL = {
    "vanilla_rabitq": "van",
    "vanilla_rabitq_meanshift": "ms",
    "pmc_1.00": "pmc",
}


# ---------------------------------------------------------------------------
# Index configuration (Table 2 "sqrtN" + rho-quarter nprobe)
# ---------------------------------------------------------------------------

def sqrt_nlist(n: int) -> int:
    """Table 2's sqrtN cell count: max(4, round(sqrt(N_db)))."""
    return max(4, int(round(sqrt(n))))


def compute_nprobe_grid(nlist: int) -> List[int]:
    """Sorted deduped nprobe values clamped to [1, nlist] (the swept grid)."""
    raw = NPROBE_BASE + [nlist]
    return sorted(set(max(1, min(nlist, v)) for v in raw))


def rho_quarter_nprobe(nlist: int) -> int:
    """Matched-budget nprobe: round_half_up(nlist / 4) snapped to the grid.

    Ties break toward the larger swept value, matching reproduce_tab2_main.py.
    """
    rho = int(Decimal(str(nlist / 4)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    grid = compute_nprobe_grid(nlist)
    return min(grid, key=lambda np_: (abs(np_ - rho), -np_))


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def batched_search(search_fn, queries: np.ndarray) -> np.ndarray:
    """Run search_fn over QUERY_BATCH-sized query slices; stack the neighbor IDs.

    search_fn takes a query slice and returns its (Q, top_k) ID array.
    """
    parts = [
        search_fn(queries[start:start + QUERY_BATCH])
        for start in range(0, len(queries), QUERY_BATCH)
    ]
    return np.vstack(parts)


def require(path: Path) -> Path:
    """Fail loudly on a missing feature cache instead of skipping a row."""
    if not path.exists():
        print(f"[map_ndcg] ERROR: feature cache not found: {path}")
        sys.exit(1)
    return path


def load_embeddings(filename: str) -> np.ndarray:
    """Load an .npy feature cache and L2-normalize it."""
    path = require(FEATURES_DIR / filename)
    return l2_normalize(np.load(str(path)).astype(np.float32))


# ---------------------------------------------------------------------------
# One (dataset, backbone, direction) cell: three methods
# ---------------------------------------------------------------------------

def run_cell(
    dataset: str,
    backbone: str,
    direction: str,
    db_emb: np.ndarray,
    query_emb: np.ndarray,
) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """Evaluate vanilla / meanshift / PMC on one direction.

    Returns (csv_records, {method: retrieved_ids}) so callers can run extra
    protocol-specific checks on the same retrievals.
    """
    n_db, d = db_emb.shape
    nlist = sqrt_nlist(n_db)
    nprobe = rho_quarter_nprobe(nlist)

    print(f"\n{'=' * 78}")
    print(f"{dataset} / {backbone} / {direction}  N_db={n_db} d={d} "
          f"nlist={nlist} nprobe={nprobe} Q={query_emb.shape[0]}")
    print("=" * 78, flush=True)

    t0 = time.time()
    gt = compute_ground_truth(query_emb, db_emb, top_k=TOP_K)
    print(f"  exact-IP top-{TOP_K} ground truth: {time.time() - t0:.1f}s", flush=True)

    vanilla_idx = build_vanilla_rabitq(db_emb, nlist=nlist, seed=SEED)
    pmc_idx, gap = build_pmc_rabitq_index(
        db_emb, query_emb, alpha=PMC_ALPHA, nlist=nlist, seed=SEED,
    )
    q_meanshift = shift_query_vectors(query_emb, gap, alpha=0.0)

    retrievals: Dict[str, np.ndarray] = {
        "vanilla_rabitq": batched_search(
            lambda q: vanilla_idx.search(q, top_k=TOP_K, nprobe=nprobe)[1], query_emb),
        "vanilla_rabitq_meanshift": batched_search(
            lambda q: vanilla_idx.search(q, top_k=TOP_K, nprobe=nprobe)[1], q_meanshift),
        "pmc_1.00": batched_search(
            lambda q: search_pmc(
                pmc_idx, q, gap, alpha=PMC_ALPHA, top_k=TOP_K, nprobe=nprobe)[1],
            query_emb),
    }

    records: List[Dict] = []
    for method, retrieved in retrievals.items():
        t1 = time.time()
        row = {
            "dataset": dataset,
            "backbone": backbone,
            "direction": direction,
            "nlist": nlist,
            "nprobe": nprobe,
            "method": method,
            "n_db": n_db,
            "d": d,
            "seed": SEED,
        }
        for k in KS:
            row[f"map@{k}"] = round(map_at_k(retrieved, gt, k), 6)
            row[f"ndcg@{k}"] = round(ndcg_at_k(retrieved, gt, k), 6)
        row["r10"] = round(recall_at_k(retrieved, gt, 10), 6)
        row["r100"] = round(recall_at_k(retrieved, gt, 100), 6)
        records.append(row)
        print(
            f"  {method:<26} mAP@10={row['map@10']:.4f} mAP@100={row['map@100']:.4f}  "
            f"nDCG@10={row['ndcg@10']:.4f} nDCG@100={row['ndcg@100']:.4f}  "
            f"R@10={row['r10']:.4f} R@100={row['r100']:.4f}  ({time.time() - t1:.1f}s)",
            flush=True,
        )
    return records, retrievals


# ---------------------------------------------------------------------------
# AudioCaps: rebuild Table 2's deduplicated clip/caption subsets
# ---------------------------------------------------------------------------

def audiocaps_metadata_csv() -> Path:
    """Locate AudioCaps test.csv (CFG path first, then the features tree)."""
    candidates = [
        Path(str(CFG.audiocaps_metadata_csv)),
        FEATURES_DIR.parent / "raw" / "audiocaps" / "test.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    print("[map_ndcg] ERROR: AudioCaps test.csv not found. Tried:")
    for path in candidates:
        print(f"    {path}")
    sys.exit(1)


def build_audiocaps_subsets() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[set]]:
    """Return (audio_std, text_std, gt_t2a, gt_a2t) as used by Table 2.

    audio_std holds one embedding per unique clip; text_std holds the captions
    of those clips in clip order.  gt_t2a/gt_a2t are the paper's SEMANTIC
    ground truth, kept only for the harness cross-check.
    """
    import pandas as pd

    audio_path = require(FEATURES_DIR / AUDIOCAPS_AUDIO_FILE)
    with open(audio_path.with_suffix(".json"), "r") as handle:
        valid_aids = json.load(handle)
    valid_set = set(valid_aids)

    df = pd.read_csv(str(audiocaps_metadata_csv()))
    clips = []
    for (yt, st), group in df.groupby(["youtube_id", "start_time"]):
        aids_valid = sorted(
            int(r["audiocap_id"]) for _, r in group.iterrows()
            if int(r["audiocap_id"]) in valid_set
        )
        if aids_valid:
            clips.append(((yt, st), aids_valid))
    clips.sort(key=lambda x: x[1][0])

    aid_to_pos = {aid: pos for pos, aid in enumerate(valid_aids)}
    clip_audio_positions: List[int] = []
    text_positions_ordered: List[int] = []
    offsets = [0]
    for _, aids in clips:
        clip_audio_positions.append(aid_to_pos[aids[0]])
        text_positions_ordered.extend(aid_to_pos[a] for a in aids)
        offsets.append(offsets[-1] + len(aids))

    audio_all = load_embeddings(AUDIOCAPS_AUDIO_FILE)
    text_all = load_embeddings(AUDIOCAPS_TEXT_FILE)
    audio_std = audio_all[clip_audio_positions]
    text_std = text_all[text_positions_ordered]

    n_clips = len(clips)
    gt_t2a = np.zeros(len(text_positions_ordered), dtype=np.int64)
    for j in range(n_clips):
        gt_t2a[offsets[j]:offsets[j + 1]] = j
    gt_a2t = [set(range(offsets[j], offsets[j + 1])) for j in range(n_clips)]
    return audio_std, text_std, gt_t2a, gt_a2t


def audiocaps_semantic_check(
    direction: str,
    retrievals: Dict[str, np.ndarray],
    gt_semantic,
    single_gt: bool,
) -> None:
    """Print Table 2's semantic-protocol R@100 for the same retrievals."""
    recall_fn = recall_at_k_single_gt if single_gt else recall_at_k_multi_gt
    print(f"  [semantic-protocol cross-check] AudioCaps {direction} (Table 2 definition)")
    for method, retrieved in retrievals.items():
        r100 = recall_fn(retrieved, gt_semantic, 100)
        print(f"    {method:<26} semantic R@100={r100:.4f}")


# ---------------------------------------------------------------------------
# Cross-check against Table 2's committed R@100 values
# ---------------------------------------------------------------------------

def load_tab2() -> Dict[Tuple[str, str], Dict[str, str]]:
    """Index tab2_main_reproduced.csv rows by (Dataset, Enc)."""
    if not TAB2_CSV.exists():
        print(f"[map_ndcg] WARNING: {TAB2_CSV} missing -- skipping cross-check.")
        return {}
    with TAB2_CSV.open(newline="", encoding="utf-8") as handle:
        return {(r["Dataset"], r["Enc"]): r for r in csv.DictReader(handle)}


def parse_tab2_recall(cell: str) -> float:
    """Parse Table 2's leading-dot recall strings (".58" -> 0.58)."""
    return float("0" + cell) if cell.startswith(".") else float(cell)


def print_cross_check(records: Sequence[Dict]) -> None:
    """Compare this run's r100 against Table 2's committed no-rerank values."""
    tab2 = load_tab2()
    if not tab2:
        return
    print(f"\n{'=' * 78}")
    print("CROSS-CHECK: r100 (this run) vs Table 2 'No reranking' R@100")
    print("=" * 78)
    header = (f"{'Dataset/Enc':<26} {'Direction':<14} {'Method':<26} "
              f"{'this':>6} {'tab2':>6} {'diff':>7}")
    print(header)
    print("-" * len(header))
    for row in records:
        key = TAB2_KEY.get((row["dataset"], row["backbone"]))
        if key is None or key not in tab2:
            continue
        prefix = "q" if row["direction"].startswith("text->") else "db"
        col = f"{prefix}_r100_{TAB2_METHOD_COL[row['method']]}"
        cell = tab2[key].get(col, "")
        if not cell:
            continue
        expected = parse_tab2_recall(cell)
        diff = row["r100"] - expected
        flag = "" if abs(diff) <= 0.01 else ("  <-- AudioCaps: different GT (see note)"
                                             if row["dataset"] == "AudioCaps" else "  <-- MISMATCH")
        print(f"{row['dataset'] + '/' + row['backbone']:<26} {row['direction']:<14} "
              f"{row['method']:<26} {row['r100']:>6.3f} {expected:>6.3f} {diff:>+7.3f}{flag}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_markdown_summary(records: Sequence[Dict]) -> None:
    """Markdown table of every cell/method plus a per-metric PMC win tally."""
    print(f"\n{'=' * 78}")
    print("SUMMARY (markdown)")
    print("=" * 78)
    cols = ["dataset", "backbone", "direction", "nprobe", "method",
            "map@10", "map@100", "ndcg@10", "ndcg@100", "r10", "r100"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for row in records:
        cells = [
            str(row[c]) if not isinstance(row[c], float) else f"{row[c]:.4f}"
            for c in cols
        ]
        print("| " + " | ".join(cells) + " |")

    print("\nPMC best-in-cell tally (out of the evaluated cells):")
    cells = sorted({(r["dataset"], r["backbone"], r["direction"]) for r in records})
    for metric in ["map@10", "map@100", "ndcg@10", "ndcg@100", "r10", "r100"]:
        wins = 0
        for cell in cells:
            group = [r for r in records
                     if (r["dataset"], r["backbone"], r["direction"]) == cell]
            best = max(group, key=lambda r: r[metric])
            if best["method"] == "pmc_1.00":
                wins += 1
        print(f"  {metric:<9} PMC best in {wins}/{len(cells)} cells")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    faiss.omp_set_num_threads(1)
    t_start = time.time()
    print(f"[map_ndcg] seed={SEED} top_k={TOP_K} alpha={PMC_ALPHA} ks={KS}")
    print(f"[map_ndcg] features: {FEATURES_DIR}")

    records: List[Dict] = []

    for dataset, backbone, db_mod, query_mod, db_file, query_file in IMAGE_CONDITIONS:
        db_emb = load_embeddings(db_file)
        query_emb = load_embeddings(query_file)
        forward, _ = run_cell(
            dataset, backbone, f"{query_mod}->{db_mod}", db_emb, query_emb)
        reverse, _ = run_cell(
            dataset, backbone, f"{db_mod}->{query_mod}", query_emb, db_emb)
        records.extend(forward)
        records.extend(reverse)

    audio_std, text_std, gt_t2a, gt_a2t = build_audiocaps_subsets()
    ac_forward, ac_forward_ret = run_cell(
        "AudioCaps", "ImageBind", "text->audio", audio_std, text_std)
    audiocaps_semantic_check("text->audio", ac_forward_ret, gt_t2a, single_gt=True)
    ac_reverse, ac_reverse_ret = run_cell(
        "AudioCaps", "ImageBind", "audio->text", text_std, audio_std)
    audiocaps_semantic_check("audio->text", ac_reverse_ret, gt_a2t, single_gt=False)
    records.extend(ac_forward)
    records.extend(ac_reverse)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[map_ndcg] CSV written -> {OUT_CSV}")

    print_cross_check(records)
    print_markdown_summary(records)
    print(f"\n[map_ndcg] Done. {len(records)} records in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
