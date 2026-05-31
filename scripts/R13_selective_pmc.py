"""
R13_selective_pmc.py -- Selective PMC: shift only the top-P% gap dimensions.

MOTIVATION:
  The modality gap g = mean(queries) - mean(db) is concentrated in a small
  number of dimensions (top 10% carry ~83% of gap energy). Selective PMC
  applies the gap correction only to those high-energy dimensions, zeroing
  the shift in the remaining low-energy dimensions.

  Hypothesis: most of the recall improvement from full PMC comes from the
  top-P% dimensions; shifting low-energy dimensions may add noise.

METHOD:
  gap = mean(queries) - mean(db)
  mask = top P% dimensions by |gap_i|  (1 for selected, 0 for rest)
  g_sel = gap * mask

  DB shift (alpha=1):   x' = normalize(x + 1.0 * g_sel)
  Query: no shift       q' = q   (alpha=1 => 1-alpha=0 => no query shift)

  Build IVFRaBitQFastScan on x'. Evaluate recall against ORIGINAL GT
  (brute-force IP on unshifted db vs original queries).

P_VALUES = [5, 10, 20, 50, 100]  (100 = full PMC, 0 = vanilla baseline)

DATASETS:
  MSCOCO val5k -- CLIP-ViT-B/32 (d=512)   directions: t2i, i2t
  AudioCaps test -- ImageBind (d=1024)     directions: t2a, a2t

OUTPUT:
  results/selective_pmc_rabitq.csv

USAGE:
  python3 scripts/research/R13_selective_pmc.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Suppress OpenMP duplicate library warnings (macOS + faiss)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Locate project root: first ancestor that has both src/ and data/ directories
_ROOT = next(
    p for p in Path(__file__).resolve().parents
    if (p / "src").is_dir() and (p / "data").is_dir()
)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import faiss

from src.core.pmc import compute_gap
from src.core.metrics import recall_at_k
from src.utils import ensure_float32_c, l2_normalize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
NPROBE = 16
TOP_K = 100
ALPHA = 1.0  # alpha=1 => DB fully shifted, query unchanged

# Percentage of dimensions (ranked by |g_i|) to shift
P_VALUES = [5, 10, 20, 50, 100]

DATA_DIR = _ROOT / "data" / "features"
OUT_CSV = _ROOT / "results" / "selective_pmc_rabitq.csv"

FIELDNAMES = [
    "dataset", "backbone", "direction",
    "top_p_percent", "r10", "r100",
    "gt_protocol", "gap_norm", "gap_energy_captured",
]


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def compute_nlist(n: int) -> int:
    """IVF cell count: sqrt heuristic for small datasets, 64 otherwise."""
    if n < 1000:
        return max(4, int(np.sqrt(n)))
    return 64


def build_rabitq(db: np.ndarray, nlist: int = 64, nprobe: int = 16, seed: int = 42) -> faiss.Index:
    """Build IVFRaBitQFastScan index on db. Returns raw faiss index."""
    db = ensure_float32_c(db)
    n, d = db.shape
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFRaBitQFastScan(quantizer, d, nlist)
    index.cp.seed = seed
    index.cp.min_points_per_centroid = 1
    index.nprobe = nprobe
    print(f"    [rabitq] Training (d={d}, nlist={nlist}) on {n} vectors ...")
    index.train(db)
    index.add(db)
    print(f"    [rabitq] Done. code_size={index.code_size} bytes/vec")
    return index


# ---------------------------------------------------------------------------
# Ground truth: brute-force IP on ORIGINAL (unshifted) db vs queries
# ---------------------------------------------------------------------------

def compute_gt_ip(queries: np.ndarray, db: np.ndarray, top_k: int = 100) -> np.ndarray:
    """Brute-force inner product ground truth on original (unshifted) vectors.

    Uses IndexFlatIP since CLIP/ImageBind embeddings are L2-normalized and
    cosine similarity == IP on unit vectors.

    Returns (Q, top_k) int64 array of nearest-neighbor indices.
    """
    d = db.shape[1]
    q_f32 = ensure_float32_c(l2_normalize(queries))
    db_f32 = ensure_float32_c(l2_normalize(db))
    gt_index = faiss.IndexFlatIP(d)
    gt_index.add(db_f32)
    _, indices = gt_index.search(q_f32, top_k)
    return indices


# ---------------------------------------------------------------------------
# Selective PMC
# ---------------------------------------------------------------------------

def selective_gap(gap: np.ndarray, top_p_percent: float) -> np.ndarray:
    """Return gap vector zeroed in all but the top top_p_percent dimensions.

    Dimensions are ranked by absolute value |gap_i|.
    top_p_percent=100 => full gap (standard PMC).
    top_p_percent=0   => zero vector (no shift, vanilla).
    """
    d = len(gap)
    n_sel = max(1, int(round(d * top_p_percent / 100.0)))
    top_dims = np.argsort(np.abs(gap))[::-1][:n_sel]
    mask = np.zeros(d, dtype=np.float32)
    mask[top_dims] = 1.0
    return gap * mask


def gap_energy_captured(gap: np.ndarray, gap_sel: np.ndarray) -> float:
    """Fraction of gap L2 energy captured by the selected dimensions."""
    total_energy = float(np.dot(gap, gap))
    if total_energy < 1e-12:
        return 0.0
    return float(np.dot(gap_sel, gap_sel)) / total_energy


# ---------------------------------------------------------------------------
# Evaluation for one direction
# ---------------------------------------------------------------------------

def run_direction(
    db_emb: np.ndarray,
    query_emb: np.ndarray,
    dataset: str,
    backbone: str,
    direction: str,
) -> List[Dict]:
    """Run vanilla + selective PMC experiments for one dataset direction.

    Returns a list of result dicts (one per P value, plus vanilla baseline).
    """
    n, d = db_emb.shape
    nlist = compute_nlist(n)

    print(f"\n  [{dataset} | {direction}] n={n}, d={d}, nlist={nlist}")

    # Ground truth: IP on original (unshifted, L2-normalized) vectors
    print("    Computing brute-force GT (original vectors, IP) ...")
    gt = compute_gt_ip(query_emb, db_emb, top_k=TOP_K)

    # Gap vector
    gap = compute_gap(db_emb, query_emb)
    gap_norm = float(np.linalg.norm(gap))
    print(f"    Gap norm: {gap_norm:.4f}")

    # Normalized queries (unchanged for alpha=1)
    queries_norm = ensure_float32_c(l2_normalize(query_emb))

    rows: List[Dict] = []

    # --- Vanilla baseline (no shift) ---
    print("    Running: vanilla (P=0)")
    db_norm = ensure_float32_c(l2_normalize(db_emb))
    idx_vanilla = build_rabitq(db_norm, nlist=nlist, nprobe=NPROBE, seed=SEED)
    _, ids_vanilla = idx_vanilla.search(queries_norm, TOP_K)
    r10_v = recall_at_k(ids_vanilla, gt, 10)
    r100_v = recall_at_k(ids_vanilla, gt, 100)
    rows.append({
        "dataset": dataset,
        "backbone": backbone,
        "direction": direction,
        "top_p_percent": 0,
        "r10": round(r10_v, 6),
        "r100": round(r100_v, 6),
        "gt_protocol": "original_ip",
        "gap_norm": round(gap_norm, 6),
        "gap_energy_captured": 0.0,
    })
    print(f"    Vanilla: R@10={r10_v:.4f}, R@100={r100_v:.4f}")

    # --- Selective PMC for each P ---
    for p in P_VALUES:
        print(f"    Running: selective PMC P={p}%")
        gap_sel = selective_gap(gap, p)
        energy_frac = gap_energy_captured(gap, gap_sel)

        # Shift DB: x' = normalize(x + 1.0 * gap_sel)
        db_shifted = ensure_float32_c(l2_normalize(db_emb + ALPHA * gap_sel[np.newaxis, :]))

        idx = build_rabitq(db_shifted, nlist=nlist, nprobe=NPROBE, seed=SEED)
        # Query: alpha=1 => no query shift; use L2-normalized original queries
        _, ids = idx.search(queries_norm, TOP_K)
        r10 = recall_at_k(ids, gt, 10)
        r100 = recall_at_k(ids, gt, 100)
        rows.append({
            "dataset": dataset,
            "backbone": backbone,
            "direction": direction,
            "top_p_percent": p,
            "r10": round(r10, 6),
            "r100": round(r100, 6),
            "gt_protocol": "original_ip",
            "gap_norm": round(gap_norm, 6),
            "gap_energy_captured": round(energy_frac, 6),
        })
        print(f"    P={p}%: R@10={r10:.4f}, R@100={r100:.4f}  (energy={energy_frac:.3f})")

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(SEED)

    out_path = OUT_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []

    # ------------------------------------------------------------------
    # MSCOCO val5k -- CLIP-ViT-B/32 (d=512)
    # ------------------------------------------------------------------
    print("\n=== MSCOCO val5k (CLIP-ViT-B/32, d=512) ===")
    mscoco_image = np.load(DATA_DIR / "mscoco_karpathy_val5k_clip_image_seed42.npy")
    mscoco_text = np.load(DATA_DIR / "mscoco_karpathy_val5k_clip_text_seed42.npy")
    print(f"  image: {mscoco_image.shape}, text: {mscoco_text.shape}")

    # text -> image (query=text, db=image)
    rows = run_direction(
        db_emb=mscoco_image,
        query_emb=mscoco_text,
        dataset="mscoco",
        backbone="clip-b32",
        direction="t2i",
    )
    all_rows.extend(rows)

    # image -> text (query=image, db=text)
    rows = run_direction(
        db_emb=mscoco_text,
        query_emb=mscoco_image,
        dataset="mscoco",
        backbone="clip-b32",
        direction="i2t",
    )
    all_rows.extend(rows)

    # ------------------------------------------------------------------
    # AudioCaps test -- ImageBind (d=1024)
    # ------------------------------------------------------------------
    # AudioCaps: asymmetric 884 unique clips / 4415 captions
    # t→a: DB=audio_single(884), Q=text_full(4415)
    # a→t: DB=text_full(4415), Q=audio_single(884)
    print("\n=== AudioCaps test (ImageBind, d=1024) ===")
    ac_audio_single = np.load(DATA_DIR / "audiocaps_test_imagebind_audio_single_seed42.npy")
    ac_text = np.load(DATA_DIR / "audiocaps_test_imagebind_text_seed42.npy")
    print(f"  audio(single): {ac_audio_single.shape}, text: {ac_text.shape}")

    # text -> audio (query=text, db=audio_single)
    rows = run_direction(
        db_emb=ac_audio_single,
        query_emb=ac_text,
        dataset="audiocaps",
        backbone="imagebind",
        direction="t2a",
    )
    all_rows.extend(rows)

    # audio -> text (query=audio_single, db=text)
    rows = run_direction(
        db_emb=ac_text,
        query_emb=ac_audio_single,
        dataset="audiocaps",
        backbone="imagebind",
        direction="a2t",
    )
    all_rows.extend(rows)

    # ------------------------------------------------------------------
    # Flickr30K test1k -- CLIP-ViT-B/32 (d=512)
    # ------------------------------------------------------------------
    print("\n=== Flickr30K test1k (CLIP-ViT-B/32, d=512) ===")
    flickr_image = np.load(DATA_DIR / "flickr30k_test1k_clip_image_seed42.npy")
    flickr_text = np.load(DATA_DIR / "flickr30k_test1k_clip_text_seed42.npy")
    print(f"  image: {flickr_image.shape}, text: {flickr_text.shape}")

    # text -> image
    rows = run_direction(
        db_emb=flickr_image,
        query_emb=flickr_text,
        dataset="flickr30k",
        backbone="clip-b32",
        direction="t2i",
    )
    all_rows.extend(rows)

    # image -> text
    rows = run_direction(
        db_emb=flickr_text,
        query_emb=flickr_image,
        dataset="flickr30k",
        backbone="clip-b32",
        direction="i2t",
    )
    all_rows.extend(rows)

    # ------------------------------------------------------------------
    # Clotho eval -- ImageBind (d=1024)
    # ------------------------------------------------------------------
    print("\n=== Clotho eval (ImageBind, d=1024) ===")
    clotho_audio = np.load(DATA_DIR / "clotho_eval_imagebind_audio_seed42.npy")
    clotho_text = np.load(DATA_DIR / "clotho_eval_imagebind_text_seed42.npy")
    print(f"  audio: {clotho_audio.shape}, text: {clotho_text.shape}")

    # text -> audio
    rows = run_direction(
        db_emb=clotho_audio,
        query_emb=clotho_text,
        dataset="clotho",
        backbone="imagebind",
        direction="t2a",
    )
    all_rows.extend(rows)

    # audio -> text
    rows = run_direction(
        db_emb=clotho_text,
        query_emb=clotho_audio,
        dataset="clotho",
        backbone="imagebind",
        direction="a2t",
    )
    all_rows.extend(rows)

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {out_path}")

    # Summary table
    print("\n--- Summary (R@100) ---")
    print(f"{'dataset':<12} {'direction':<6} {'P%':>5} {'R@10':>8} {'R@100':>8} {'energy':>8}")
    for row in all_rows:
        print(
            f"{row['dataset']:<12} {row['direction']:<6} {row['top_p_percent']:>5} "
            f"{row['r10']:>8.4f} {row['r100']:>8.4f} {row['gap_energy_captured']:>8.3f}"
        )


if __name__ == "__main__":
    main()
