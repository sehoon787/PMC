"""
27_verify_unverifiable_claims.py -- Compute and record previously-unverifiable
paper claims for the PMC CIKM 2026 short paper.

Claims verified:
  1. Gap norm ||g|| for each available dataset/backbone
  2. Gap energy concentration (top 10% dims > 83%, top 5% dims > 87%)
  3. Same-modality oracle R@100 -- both exact-IP and vanilla RaBitQ
  4. "33 B/vec lower bound" -- recorded as THEORETICAL

Output:
  results/unverifiable_claims_verification.csv
  Formatted summary printed to stdout.

Run:
  python scripts/analysis/27_verify_unverifiable_claims.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "src").is_dir() and (parent / "config").is_dir()
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import faiss

faiss.omp_set_num_threads(1)

from src.runtime.config import CFG

FEATURES_DIR = CFG.features_dir
RESULTS_DIR = CFG.results_dir
OUTPUT_CSV = RESULTS_DIR / "unverifiable_claims_verification.csv"

TOP_K = 100
NLIST = 64
NPROBE = 16
SEED = 42

# Paper-stated values for comparison (None = paper does not give an exact figure)
# gap_norm: paper mentions the gap exists but no exact norm table is cited
# energy_top10: "top 10% of dimensions capture >83% of gap energy"
# energy_top5:  "top 5% of dimensions capture >87% of gap energy"
# oracle_r100:  "same-modality retrieval achieves R@100=0.71 on MSCOCO"
PAPER_GAP_NORM = None          # no exact table in paper
PAPER_ENERGY_TOP10 = 0.83      # lower bound claimed in paper
PAPER_ENERGY_TOP5 = 0.87       # lower bound claimed in paper
PAPER_ORACLE_R100 = 0.71       # paper intro figure for vanilla RaBitQ same-modal

# ---------------------------------------------------------------------------
# Dataset configurations -- (label, db_file, query_file, backbone)
# ---------------------------------------------------------------------------
DATASETS = [
    {
        "label": "MSCOCO CLIP-L/14",
        "dataset": "mscoco_coco5k",
        "backbone": "clip-l",
        "direction": "text->image",
        "db_file": "mscoco_karpathy_val5k_clip-l_image_seed42.npy",
        "query_file": "mscoco_karpathy_val5k_clip-l_text_seed42.npy",
    },
    {
        "label": "AudioCaps ImageBind",
        "dataset": "audiocaps_test",
        "backbone": "imagebind",
        "direction": "text->audio",
        "db_file": "audiocaps_test_imagebind_audio_seed42.npy",
        "query_file": "audiocaps_test_imagebind_text_seed42.npy",
    },
    {
        "label": "Clotho ImageBind",
        "dataset": "clotho_eval",
        "backbone": "imagebind",
        "direction": "text->audio",
        "db_file": "clotho_eval_imagebind_audio_seed42.npy",
        "query_file": "clotho_eval_imagebind_text_5cap_seed42.npy",
    },
]

# For same-modality oracle: use MSCOCO CLIP-L image as both DB and queries
ORACLE_DB_FILE = "mscoco_karpathy_val5k_clip-l_image_seed42.npy"
ORACLE_BACKBONE = "clip-l"
ORACLE_DATASET = "mscoco_coco5k"

# CSV column names
FIELDNAMES = [
    "claim_type",
    "dataset",
    "backbone",
    "direction",
    "computed_value",
    "paper_value",
    "status",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_features(fname: str) -> "np.ndarray | None":
    """Load float32 feature array from FEATURES_DIR. Returns None if missing."""
    path = FEATURES_DIR / fname
    if not path.exists():
        print(f"  [SKIP] Feature file not found: {path}")
        return None
    arr = np.load(path).astype(np.float32)
    print(f"  Loaded {fname}: shape={arr.shape}")
    return arr


def compute_gap(db: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Compute gap vector g = mean(query) - mean(db) on L2-normalised vectors."""
    return query.mean(axis=0) - db.mean(axis=0)


def compute_gap_energy_fractions(g: np.ndarray) -> "tuple[float, float]":
    """
    Returns (top10_frac, top5_frac) -- fraction of total squared gap energy
    captured by the top 10% and top 5% of dimensions (by energy), respectively.
    """
    d = len(g)
    g_sq = g ** 2
    total = g_sq.sum()
    if total < 1e-30:
        return float("nan"), float("nan")
    sorted_desc = np.sort(g_sq)[::-1]
    n10 = max(1, int(0.10 * d))
    n05 = max(1, int(0.05 * d))
    top10 = float(sorted_desc[:n10].sum() / total)
    top05 = float(sorted_desc[:n05].sum() / total)
    return top10, top05


def ensure_float32_c(arr: np.ndarray) -> np.ndarray:
    """Return C-contiguous float32 copy if needed."""
    if arr.dtype != np.float32 or not arr.flags["C_CONTIGUOUS"]:
        return np.ascontiguousarray(arr, dtype=np.float32)
    return arr


def build_flat_ip_index(db: np.ndarray) -> faiss.IndexFlatIP:
    """Build exact inner-product index (for same-modality oracle)."""
    d = db.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(ensure_float32_c(db))
    return idx


def build_rabitq_index(db: np.ndarray, nlist: int, seed: int) -> faiss.IndexIVFRaBitQFastScan:
    """Build IVFRaBitQFastScan index. Returns raw faiss index."""
    db = ensure_float32_c(db)
    d = db.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFRaBitQFastScan(quantizer, d, nlist, 0)
    index.cp.seed = seed
    index.cp.min_points_per_centroid = 1
    print(f"  [rabitq_fs] Training (d={d}, nlist={nlist}) ...")
    index.train(db)
    index.add(db)
    print(f"  [rabitq_fs] Done. code_size={index.code_size} bytes/vec")
    return index


def compute_ground_truth_ip(queries: np.ndarray, db: np.ndarray, top_k: int) -> np.ndarray:
    """Brute-force inner-product ground truth. Returns (Q, top_k) int64 array."""
    d = db.shape[1]
    flat = faiss.IndexFlatIP(d)
    flat.add(ensure_float32_c(db))
    _, indices = flat.search(ensure_float32_c(queries), top_k)
    return indices


def recall_at_k(retrieved: np.ndarray, gt: np.ndarray, k: int) -> float:
    """K-recall@K: average over queries of |ret_k inter gt_k| / |gt_k|."""
    q = len(retrieved)
    total = 0.0
    for i in range(q):
        ret_k = set(int(x) for x in retrieved[i, :k] if x >= 0)
        gt_k = set(int(x) for x in gt[i, :k] if x >= 0)
        total += len(ret_k & gt_k) / max(len(gt_k), 1)
    return total / q if q > 0 else 0.0


def pass_status(computed: float, paper: "float | None", mode: str = "eq", tol: float = 0.02) -> str:
    """
    Determine PASS / MISMATCH status.
    mode='eq'  : |computed - paper| <= tol
    mode='ge'  : computed >= paper (lower bound claim)
    mode='skip': not applicable
    """
    if paper is None:
        return "NO_PAPER_VALUE"
    if mode == "eq":
        if abs(computed - paper) <= tol:
            return "PASS"
        return f"MISMATCH (diff={computed - paper:+.4f})"
    if mode == "ge":
        if computed >= paper:
            return "PASS"
        return f"MISMATCH (computed={computed:.4f} < paper={paper:.4f})"
    return "SKIP"


# ---------------------------------------------------------------------------
# Claim 1 + 2: Gap norm and energy concentration
# ---------------------------------------------------------------------------

def check_gap_claims(records: list) -> None:
    """Compute gap norm and energy concentration for each available dataset."""
    print("\n=== CLAIMS 1+2: Gap norm and energy concentration ===")

    for ds in DATASETS:
        db = load_features(ds["db_file"])
        queries = load_features(ds["query_file"])

        if db is None or queries is None:
            records.append({
                "claim_type": "gap_norm",
                "dataset": ds["dataset"],
                "backbone": ds["backbone"],
                "direction": ds["direction"],
                "computed_value": "N/A",
                "paper_value": str(PAPER_GAP_NORM),
                "status": "SKIP",
            })
            records.append({
                "claim_type": "gap_energy_top10pct",
                "dataset": ds["dataset"],
                "backbone": ds["backbone"],
                "direction": ds["direction"],
                "computed_value": "N/A",
                "paper_value": str(PAPER_ENERGY_TOP10),
                "status": "SKIP",
            })
            records.append({
                "claim_type": "gap_energy_top5pct",
                "dataset": ds["dataset"],
                "backbone": ds["backbone"],
                "direction": ds["direction"],
                "computed_value": "N/A",
                "paper_value": str(PAPER_ENERGY_TOP5),
                "status": "SKIP",
            })
            continue

        g = compute_gap(db, queries)
        gap_norm = float(np.linalg.norm(g))
        top10_frac, top5_frac = compute_gap_energy_fractions(g)
        d = len(g)

        print(f"\n  {ds['label']}  (d={d}, n_db={len(db)}, n_query={len(queries)})")
        print(f"    gap_norm     = {gap_norm:.6f}")
        print(f"    top10% dims  = {top10_frac:.4f}  (paper claim: >{PAPER_ENERGY_TOP10})")
        print(f"    top 5% dims  = {top5_frac:.4f}  (paper claim: >{PAPER_ENERGY_TOP5})")

        # Claim 1: gap norm (no exact paper value, just record)
        records.append({
            "claim_type": "gap_norm",
            "dataset": ds["dataset"],
            "backbone": ds["backbone"],
            "direction": ds["direction"],
            "computed_value": f"{gap_norm:.6f}",
            "paper_value": "N/A",
            "status": "NO_PAPER_VALUE",
        })

        # Claim 2a: top 10% energy
        status_10 = pass_status(top10_frac, PAPER_ENERGY_TOP10, mode="ge")
        records.append({
            "claim_type": "gap_energy_top10pct",
            "dataset": ds["dataset"],
            "backbone": ds["backbone"],
            "direction": ds["direction"],
            "computed_value": f"{top10_frac:.6f}",
            "paper_value": str(PAPER_ENERGY_TOP10),
            "status": status_10,
        })

        # Claim 2b: top 5% energy
        status_5 = pass_status(top5_frac, PAPER_ENERGY_TOP5, mode="ge")
        records.append({
            "claim_type": "gap_energy_top5pct",
            "dataset": ds["dataset"],
            "backbone": ds["backbone"],
            "direction": ds["direction"],
            "computed_value": f"{top5_frac:.6f}",
            "paper_value": str(PAPER_ENERGY_TOP5),
            "status": status_5,
        })


# ---------------------------------------------------------------------------
# Claim 3: Same-modality oracle R@100
# ---------------------------------------------------------------------------

def check_oracle_recall(records: list) -> None:
    """
    Compute R@100 for same-modality (image-to-image) retrieval using MSCOCO
    CLIP-L/14 features.

    (a) Exact IndexFlatIP: upper bound -- should approach 1.0 since every
        vector is its own nearest neighbour.
    (b) Vanilla IVFRaBitQFastScan: the approximate binary index applied to
        same-modality vectors where there is no cross-modal gap.
    """
    print("\n=== CLAIM 3: Same-modality oracle R@100 ===")

    db = load_features(ORACLE_DB_FILE)
    if db is None:
        for suffix in ("exact_flatip", "vanilla_rabitq"):
            records.append({
                "claim_type": f"same_modal_oracle_r100_{suffix}",
                "dataset": ORACLE_DATASET,
                "backbone": ORACLE_BACKBONE,
                "direction": "image->image",
                "computed_value": "N/A",
                "paper_value": str(PAPER_ORACLE_R100),
                "status": "SKIP",
            })
        return

    queries = db   # same-modality: use image features as queries
    print(f"  n_vectors={len(db)}, d={db.shape[1]}")

    # Ground truth: exact IP neighbours of each vector in the full DB
    # The correct answer for query i is itself (index i), which sits at rank 1.
    # R@100 should therefore be 1.0 exactly for the exact index.
    print("  [oracle_exact] Computing exact IP ground truth ...")
    gt = compute_ground_truth_ip(queries, db, top_k=TOP_K)

    # (a) Exact flat IP
    print("  [oracle_exact] Searching FlatIP ...")
    flat_idx = build_flat_ip_index(db)
    _, flat_ids = flat_idx.search(ensure_float32_c(queries), TOP_K)
    r100_exact = recall_at_k(flat_ids, gt, k=TOP_K)
    print(f"    FlatIP same-modal R@100 = {r100_exact:.4f}  (expected ~1.0)")
    records.append({
        "claim_type": "same_modal_oracle_r100_exact_flatip",
        "dataset": ORACLE_DATASET,
        "backbone": ORACLE_BACKBONE,
        "direction": "image->image",
        "computed_value": f"{r100_exact:.6f}",
        "paper_value": "1.0 (trivial)",
        "status": "PASS" if r100_exact > 0.99 else "MISMATCH",
    })

    # (b) Vanilla IVFRaBitQFastScan same-modality
    print(f"  [oracle_rabitq] Building IVFRaBitQFastScan (nlist={NLIST}) ...")
    rabitq_idx = build_rabitq_index(db, nlist=NLIST, seed=SEED)
    rabitq_idx.nprobe = NPROBE
    _, rb_ids = rabitq_idx.search(ensure_float32_c(queries), TOP_K)
    r100_rb = recall_at_k(rb_ids, gt, k=TOP_K)
    paper_rb = PAPER_ORACLE_R100
    status_rb = pass_status(r100_rb, paper_rb, mode="eq", tol=0.05)
    print(f"    RaBitQ same-modal R@100 = {r100_rb:.4f}  (paper: ~{paper_rb})")
    records.append({
        "claim_type": "same_modal_oracle_r100_vanilla_rabitq",
        "dataset": ORACLE_DATASET,
        "backbone": ORACLE_BACKBONE,
        "direction": "image->image",
        "computed_value": f"{r100_rb:.6f}",
        "paper_value": str(paper_rb),
        "status": status_rb,
    })


# ---------------------------------------------------------------------------
# Claim 4: 33 B/vec lower bound (theoretical)
# ---------------------------------------------------------------------------

def record_theoretical_claims(records: list) -> None:
    """Record claims that are theoretical and cannot be computed empirically."""
    print("\n=== CLAIM 4: Theoretical 33 B/vec lower bound ===")
    print("  This is a derived lower bound from Shannon entropy arguments in the")
    print("  RaBitQ paper (Gao et al., SIGMOD 2024).  Not empirically computed.")

    records.append({
        "claim_type": "rabitq_lower_bound_33bpv",
        "dataset": "all",
        "backbone": "all",
        "direction": "N/A",
        "computed_value": "N/A",
        "paper_value": "33 B/vec",
        "status": "THEORETICAL",
    })


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_csv(records: list, path: Path) -> None:
    """Write records to CSV. Creates parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved {len(records)} rows to {path}")


def print_summary_table(records: list) -> None:
    """Print a formatted summary of all records."""
    print("\n" + "=" * 90)
    print("SUMMARY: Unverifiable Claims Verification")
    print("=" * 90)
    hdr = f"{'claim_type':<38} {'dataset':<16} {'backbone':<10} {'computed':<12} {'paper':<14} {'status'}"
    print(hdr)
    print("-" * 90)
    for r in records:
        cval = str(r["computed_value"])[:11]
        pval = str(r["paper_value"])[:13]
        print(
            f"  {r['claim_type']:<36} {r['dataset']:<16} {r['backbone']:<10}"
            f" {cval:<12} {pval:<14} {r['status']}"
        )
    print("=" * 90)

    pass_count = sum(1 for r in records if r["status"] == "PASS")
    skip_count = sum(1 for r in records if r["status"] in ("SKIP", "THEORETICAL"))
    mismatch_count = sum(1 for r in records if "MISMATCH" in r["status"])
    no_val_count = sum(1 for r in records if r["status"] == "NO_PAPER_VALUE")
    print(
        f"\nTotal: {len(records)} rows  |"
        f"  PASS={pass_count}  MISMATCH={mismatch_count}"
        f"  SKIP/THEORETICAL={skip_count}  NO_PAPER_VALUE={no_val_count}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("PMC CIKM 2026 -- unverifiable claims verification")
    print(f"Features dir : {FEATURES_DIR}")
    print(f"Output CSV   : {OUTPUT_CSV}")

    records: list[dict[str, Any]] = []

    check_gap_claims(records)
    check_oracle_recall(records)
    record_theoretical_claims(records)

    save_csv(records, OUTPUT_CSV)
    print_summary_table(records)

    mismatch = [r for r in records if "MISMATCH" in r["status"]]
    if mismatch:
        print(f"\nWARNING: {len(mismatch)} claim(s) do not match paper values.")
        return 1
    print("\nAll verifiable claims match paper values (within tolerance).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
