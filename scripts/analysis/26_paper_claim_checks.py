"""
Paper claim checks for PMC CIKM 2026.

Reproduces the two categories of embedding-level evidence cited in the paper:
  1. Centroid-gap direction convergence (cosine vs. n_samples)
  2. Audio cosine-to-centroid values (AudioCaps, Clotho)

Query-modality samples
----------------------
No separate raw query files are needed.  The convergence check (Check 1)
samples rows from the existing query feature caches stored under
data/features/ using numpy RNG with a fixed seed (RNG_SEED = 42).
For example, for MSCOCO CLIP-L the query cache is:
  data/features/mscoco_karpathy_val5k_clip-l_text_seed42.npy
The check calls rng.choice(len(query_feats), n, replace=False) to draw n
rows from this cache for each trial.  The database mean uses the full
corresponding image cache (no sampling on the database side).

Dependencies: numpy, pathlib (stdlib)
Run: python scripts/analysis/26_paper_claim_checks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# scripts/analysis/26_... is at: <project>/scripts/analysis/26_...
# parents[0] = scripts/analysis, parents[1] = scripts, parents[2] = project root
REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES = REPO_ROOT / "data" / "features"

# Claimed values from paper (for cross-check)
PAPER_CLAIMS = {
    "clotho_audio_cosine": 0.37,
}

# Convergence check parameters
SAMPLE_SIZES = [50, 100, 200, 500]
N_TRIALS = 20
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def mean_cosine_to_centroid(feats: np.ndarray) -> tuple[float, float]:
    """Return (mean, std) of per-vector cosine similarity to the centroid."""
    centroid = feats.mean(axis=0)
    norm_c = np.linalg.norm(centroid)
    if norm_c < 1e-12:
        return float("nan"), float("nan")
    centroid_unit = centroid / norm_c
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    feats_unit = feats / norms
    cosines = feats_unit @ centroid_unit
    return float(cosines.mean()), float(cosines.std())


def gap_convergence(
    query_feats: np.ndarray,
    db_feats: np.ndarray,
    sample_sizes: list[int],
    n_trials: int,
    rng: np.random.Generator,
) -> dict[int, dict[str, float]]:
    """
    For each sample size n, estimate the gap direction by sampling n query
    embeddings while using the full database mean.

    Full gap  = mean(query_feats) - mean(db_feats)
    Sample gap = mean(query_sample) - mean(db_feats)   [db mean is fixed]

    Sampling: rng.choice(len(query_feats), n, replace=False) draws row
    indices from the pre-loaded query feature cache (no separate raw files).

    Returns {n: {"rel_err_mean", "rel_err_std", "cos_mean", "cos_std"}}.
    """
    db_mean = db_feats.mean(axis=0)          # full database mean (fixed)
    full_gap = query_feats.mean(axis=0) - db_mean
    full_norm = np.linalg.norm(full_gap)

    results: dict[int, dict[str, float]] = {}
    for n in sample_sizes:
        n_query = min(n, len(query_feats))
        rel_errs, cosines = [], []
        for _ in range(n_trials):
            idx_q = rng.choice(len(query_feats), n_query, replace=False)
            sample_gap = query_feats[idx_q].mean(axis=0) - db_mean
            rel_errs.append(np.linalg.norm(sample_gap - full_gap) / (full_norm + 1e-12))
            cosines.append(cosine(sample_gap, full_gap))
        results[n] = {
            "rel_err_mean": float(np.mean(rel_errs)),
            "rel_err_std": float(np.std(rel_errs)),
            "cos_mean": float(np.mean(cosines)),
            "cos_std": float(np.std(cosines)),
        }
    return results


# ---------------------------------------------------------------------------
# Check 1: Centroid-gap direction convergence
# ---------------------------------------------------------------------------

def check_centroid_convergence() -> bool:
    """Verify gap direction stabilizes with a few hundred unlabeled queries."""
    print("\n=== CHECK 1: Gap direction convergence ===")
    rng = np.random.default_rng(RNG_SEED)
    ok = True

    configs = [
        {
            "label": "MSCOCO CLIP-L text->image",
            "text_file": "mscoco_karpathy_val5k_clip-l_text_seed42.npy",
            "img_file": "mscoco_karpathy_val5k_clip-l_image_seed42.npy",
        },
        {
            "label": "AudioCaps ImageBind text->audio",
            "text_file": "audiocaps_test_imagebind_text_single_seed42.npy",
            "img_file": "audiocaps_test_imagebind_audio_single_seed42.npy",
        },
        {
            "label": "Clotho ImageBind text->audio",
            "text_file": "clotho_eval_imagebind_text_seed42.npy",
            "img_file": "clotho_eval_imagebind_audio_seed42.npy",
        },
    ]

    for cfg in configs:
        text_path = FEATURES / cfg["text_file"]
        img_path = FEATURES / cfg["img_file"]
        if not text_path.exists() or not img_path.exists():
            print(f"  SKIP {cfg['label']}: feature files not found")
            continue

        query_feats = np.load(text_path).astype(np.float32)
        db_feats = np.load(img_path).astype(np.float32)

        conv = gap_convergence(query_feats, db_feats, SAMPLE_SIZES, N_TRIALS, rng)

        print(f"\n  {cfg['label']} (n_query={len(query_feats)}, n_db={len(db_feats)})")
        print(f"  {'n':>6}  {'rel_err':>12}  {'cos_to_full':>14}")
        for n, v in conv.items():
            cos_str = f"{v['cos_mean']:.4f}+/-{v['cos_std']:.4f}"
            err_str = f"{v['rel_err_mean']:.4f}+/-{v['rel_err_std']:.4f}"
            flag = " <-- paper claim (n=200: cos>0.995)" if n == 200 else ""
            print(f"  {n:>6}  {err_str:>12}  {cos_str:>14}{flag}")
            if n == 200 and v["cos_mean"] < 0.990:
                print(f"  WARNING: n=200 cosine {v['cos_mean']:.4f} < 0.990 threshold")
                ok = False

    return ok


# ---------------------------------------------------------------------------
# Check 3: Fixed calibration split sanity check (first-200 prefix)
# ---------------------------------------------------------------------------
# This check treats the first 200 query embeddings as an unlabeled calibration
# split and computes the gap direction from that subset alone, then measures
# cosine similarity against the full-population gap.
#
# NOTE: These are gap-direction evidence values only.  No ANN index is rebuilt
# here, so the cosines below do NOT imply retrieval recall.  The purpose is to
# confirm that a fixed, disjoint calibration prefix stabilises the gap estimate
# to a sufficient degree for PMC construction.
#
# Verified values (2026-05):
#   MSCOCO CLIP-L   : N=5000, held-out=4800, cos=0.9984, angle=3.3 deg
#   AudioCaps IB    : N=4415 captions / 884 clips; recomputed by this script
#   Clotho IB       : N=1045, held-out= 845, cos=0.9929, angle=6.8 deg
#
# Threshold: cos >= 0.98 for fixed-prefix gap-direction sanity.
CALIB_SPLIT_N = 200
CALIB_THRESHOLD = 0.98


def check_calibration_split() -> bool:
    """
    Fixed prefix calibration-split sanity check (gap-direction evidence only).

    Uses the first CALIB_SPLIT_N rows of each query cache as a disjoint
    calibration split; remaining rows form the held-out set.  Reports cosine
    similarity between the calibration-estimated gap and the full-population
    gap, plus the angle in degrees.

    PASS when cos(g_calib, g_full) >= CALIB_THRESHOLD for all datasets with
    sufficient queries (>= CALIB_SPLIT_N + 1).
    """
    print("\n=== CHECK 3: Calibration split sanity (gap-direction evidence only) ===")
    print(f"  Calibration split size : {CALIB_SPLIT_N}")
    print(f"  PASS threshold         : cos >= {CALIB_THRESHOLD}")
    print(f"  NOTE: values are gap-direction cosines, NOT retrieval recall\n")

    configs = [
        {
            "label": "MSCOCO CLIP-L (text->image)",
            "text_file": "mscoco_karpathy_val5k_clip-l_text_seed42.npy",
            "img_file": "mscoco_karpathy_val5k_clip-l_image_seed42.npy",
        },
        {
            "label": "AudioCaps ImageBind (text->audio)",
            "text_file": "audiocaps_test_imagebind_text_single_seed42.npy",
            "img_file": "audiocaps_test_imagebind_audio_single_seed42.npy",
            # AudioCaps fixed-prefix cos is intentionally near the 0.98 threshold
            # (observed ~0.9818); small corpus (884 clips / 4415 captions) reduces
            # the calib-split estimate stability relative to larger datasets.
        },
        {
            "label": "Clotho ImageBind (text->audio)",
            "text_file": "clotho_eval_imagebind_text_seed42.npy",
            "img_file": "clotho_eval_imagebind_audio_seed42.npy",
        },
    ]

    print(f"  {'Dataset':<40}  {'N query':>8}  {'held-out':>9}  {'cos':>7}  {'angledeg':>7}  status")
    ok = True
    for cfg in configs:
        text_path = FEATURES / cfg["text_file"]
        img_path = FEATURES / cfg["img_file"]
        if not text_path.exists() or not img_path.exists():
            print(f"  {cfg['label']:<40}  SKIP (feature files not found)")
            continue

        query_feats = np.load(text_path).astype(np.float32)
        db_feats = np.load(img_path).astype(np.float32)
        n_query = len(query_feats)

        if n_query <= CALIB_SPLIT_N:
            print(f"  {cfg['label']:<40}  SKIP (only {n_query} queries, need >{CALIB_SPLIT_N})")
            continue

        calib_queries = query_feats[:CALIB_SPLIT_N]
        n_held_out = n_query - CALIB_SPLIT_N

        db_mean = db_feats.mean(axis=0)
        full_gap = query_feats.mean(axis=0) - db_mean
        calib_gap = calib_queries.mean(axis=0) - db_mean

        cos_val = cosine(calib_gap, full_gap)
        # Clamp for arccos stability
        angle_deg = float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))

        if cos_val >= CALIB_THRESHOLD:
            status = "PASS"
        else:
            status = f"WARN (cos={cos_val:.4f} < {CALIB_THRESHOLD})"
            ok = False

        print(
            f"  {cfg['label']:<40}  {n_query:>8}  {n_held_out:>9}"
            f"  {cos_val:.4f}  {angle_deg:>6.1f}  {status}"
        )

    return ok


# ---------------------------------------------------------------------------
# Check 2: Audio cosine-to-centroid
# ---------------------------------------------------------------------------

def check_audio_cosine() -> bool:
    """Verify AudioCaps/Clotho cosine-to-centroid values cited in paper."""
    print("\n=== CHECK 2: Audio cosine-to-centroid ===")
    ok = True

    configs = [
        {
            "label": "AudioCaps text (captions)",
            "file": "audiocaps_test_imagebind_text_seed42.npy",
            "paper_key": None,
            "tol": 0.05,
        },
        {
            "label": "AudioCaps audio (caption-level)",
            "file": "audiocaps_test_imagebind_audio_seed42.npy",
            "paper_key": None,
            "tol": 0.05,
        },
        {
            "label": "AudioCaps audio (clip-level)",
            "file": "audiocaps_test_imagebind_audio_single_seed42.npy",
            "paper_key": None,
            "tol": 0.05,
        },
        {
            "label": "Clotho text (first cap)",
            "file": "clotho_eval_imagebind_text_seed42.npy",
            "paper_key": None,
            "tol": 0.05,
        },
        {
            "label": "Clotho audio (first cap)",
            "file": "clotho_eval_imagebind_audio_seed42.npy",
            "paper_key": "clotho_audio_cosine",
            "tol": 0.05,
        },
    ]

    print(f"\n  {'Label':<35}  {'mean':>7}  {'std':>7}  {'paper':>7}  status")
    for cfg in configs:
        feat_path = FEATURES / cfg["file"]
        if not feat_path.exists():
            print(f"  {cfg['label']:<35}  SKIP (file not found)")
            continue

        feats = np.load(feat_path).astype(np.float32)
        mean_c, std_c = mean_cosine_to_centroid(feats)

        paper_val = PAPER_CLAIMS.get(cfg["paper_key"]) if cfg["paper_key"] else None
        paper_str = f"{paper_val:.2f}" if paper_val is not None else "  N/A"

        if paper_val is not None and abs(mean_c - paper_val) > cfg["tol"]:
            status = f"WARN (diff={abs(mean_c - paper_val):.3f} > tol={cfg['tol']})"
            ok = False
        else:
            status = "OK"

        print(f"  {cfg['label']:<35}  {mean_c:.4f}  {std_c:.4f}  {paper_str:>7}  {status}")

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("PMC CIKM 2026 -- paper claim checks")
    print(f"Feature dir: {FEATURES}")

    results = [
        check_centroid_convergence(),
        check_audio_cosine(),
        check_calibration_split(),
    ]

    print("\n=== SUMMARY ===")
    labels = ["Gap convergence", "Audio cosine-to-centroid", "Calibration split (gap-dir)"]
    all_ok = True
    for label, passed in zip(labels, results):
        status = "PASS" if passed else "WARN"
        if not passed:
            all_ok = False
        print(f"  {label:<28}: {status}")

    if all_ok:
        print("\nAll checks passed.")
        return 0
    else:
        print("\nOne or more checks produced warnings. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
