"""
R13: Sign-bit variant experiments with CONSISTENT original GT.

Re-runs round7 sign_bit_variants with a single GT protocol:
  ALL rows (vanilla + pmc) evaluated against brute-force exact-IP
  on the ORIGINAL (unshifted) embedding space.

Methods: PureBinary, BinaryIVF, RotatedBinary, RaBitQ
Datasets: MSCOCO (CLIP-B/32, d=512), AudioCaps (ImageBind, d=1024)
Directions: both cross-modal directions per dataset
"""

import csv
import time
from pathlib import Path

import faiss
import numpy as np

# ─── Config ───────────────────────────────────────────────────────────────────

_PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "src").is_dir() and (parent / "data").is_dir()
)
FEATURES_DIR = _PROJECT_ROOT / "data" / "features"
RESULTS_DIR = _PROJECT_ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "signbit_original_gt.csv"

SEED = 42
np.random.seed(SEED)
faiss.omp_set_num_threads(1)

NLIST = 64
NPROBE = 16
K_VALUES = [1, 10, 100]
GT_PROTOCOL = "original_exact_ip"


# ─── Utilities ────────────────────────────────────────────────────────────────

def l2_normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / np.maximum(norms, 1e-8)).astype(np.float32)


def apply_pmc(db, queries, alpha=1.0):
    gap = queries.mean(axis=0) - db.mean(axis=0)
    db_s = l2_normalize(db + alpha * gap)
    q_s = l2_normalize(queries - (1 - alpha) * gap)
    return db_s, q_s


def compute_gt(db, queries, k=100):
    index = faiss.IndexFlatIP(db.shape[1])
    index.add(db.astype(np.float32))
    _, I = index.search(queries.astype(np.float32), k)
    return I


def compute_recall(I, gt_I, k):
    n = I.shape[0]
    return sum(
        len(set(I[i, :k].tolist()) & set(gt_I[i, :k].tolist()))
        for i in range(n)
    ) / (n * k)


def float_to_binary(vecs, centroid):
    bits = ((vecs - centroid) > 0).astype(np.uint8)
    d = bits.shape[1]
    pad = (8 - d % 8) % 8
    if pad > 0:
        bits = np.hstack([bits, np.zeros((bits.shape[0], pad), dtype=np.uint8)])
    return np.packbits(bits, axis=1).copy()


# ─── Methods ──────────────────────────────────────────────────────────────────

def run_pure_binary(db, queries, gt_I, centroid):
    d = db.shape[1]
    db_bin = float_to_binary(db, centroid)
    q_bin = float_to_binary(queries, centroid)
    nbits = db_bin.shape[1] * 8
    index = faiss.IndexBinaryFlat(nbits)
    index.add(db_bin)
    _, I = index.search(q_bin, max(K_VALUES))
    return {k: compute_recall(I, gt_I, k) for k in K_VALUES}


def run_binary_ivf(db, queries, gt_I, centroid):
    d = db.shape[1]
    db_bin = float_to_binary(db, centroid)
    q_bin = float_to_binary(queries, centroid)
    nbits = db_bin.shape[1] * 8
    quantizer = faiss.IndexBinaryFlat(nbits)
    index = faiss.IndexBinaryIVF(quantizer, nbits, NLIST)
    index.nprobe = NPROBE
    index.train(db_bin)
    index.add(db_bin)
    _, I = index.search(q_bin, max(K_VALUES))
    return {k: compute_recall(I, gt_I, k) for k in K_VALUES}


def run_rotated_binary(db, queries, gt_I, centroid, rot_matrix):
    db_rot = (db @ rot_matrix.T).astype(np.float32)
    q_rot = (queries @ rot_matrix.T).astype(np.float32)
    centroid_rot = (centroid @ rot_matrix.T).astype(np.float32)
    db_bin = float_to_binary(db_rot, centroid_rot)
    q_bin = float_to_binary(q_rot, centroid_rot)
    nbits = db_bin.shape[1] * 8
    index = faiss.IndexBinaryFlat(nbits)
    index.add(db_bin)
    _, I = index.search(q_bin, max(K_VALUES))
    return {k: compute_recall(I, gt_I, k) for k in K_VALUES}


def run_rabitq(db, queries, gt_I):
    d = db.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFRaBitQ(quantizer, d, NLIST)
    index.nprobe = NPROBE
    index.train(db)
    index.add(db)
    _, I = index.search(queries, max(K_VALUES))
    return {k: compute_recall(I, gt_I, k) for k in K_VALUES}


# ─── Experiment ───────────────────────────────────────────────────────────────

def run_experiment(db_orig, q_orig, dataset, backbone, direction, results):
    d = db_orig.shape[1]
    print(f"\n{'='*60}")
    print(f"  {dataset} | {backbone} | {direction}")
    print(f"  DB={db_orig.shape}  Q={q_orig.shape}")
    print(f"{'='*60}")

    # SINGLE GT: original brute-force (used for ALL variants)
    print("  Computing GT (original exact-IP)...")
    gt_I = compute_gt(db_orig, q_orig, k=max(K_VALUES))

    # PMC shift
    db_pmc, q_pmc = apply_pmc(db_orig, q_orig, alpha=1.0)

    # Centroids
    centroid_orig = db_orig.mean(axis=0)
    centroid_pmc = db_pmc.mean(axis=0)

    # Rotation matrix (fixed seed)
    rng = np.random.default_rng(SEED)
    H = rng.standard_normal((d, d)).astype(np.float64)
    Q, R = np.linalg.qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    rot = Q.astype(np.float32)

    methods = [
        ("PureBinary", "BinaryFlat",
         lambda db, q, gt, c: run_pure_binary(db, q, gt, c)),
        ("BinaryIVF", "BinaryIVF",
         lambda db, q, gt, c: run_binary_ivf(db, q, gt, c)),
        ("RotatedBinary", "BinaryFlat+Rotation",
         lambda db, q, gt, c: run_rotated_binary(db, q, gt, c, rot)),
        ("RaBitQ", "IVFRaBitQFastScan",
         lambda db, q, gt, c: run_rabitq(db, q, gt)),
    ]

    for method_name, index_type, run_fn in methods:
        print(f"\n  --- {method_name} ---")

        for variant, db_v, q_v, centroid in [
            ("vanilla", db_orig, q_orig, centroid_orig),
            ("pmc_a1", db_pmc, q_pmc, centroid_pmc),
        ]:
            try:
                recalls = run_fn(db_v, q_v, gt_I, centroid)
                print(f"    [{variant:7s}] R@1={recalls[1]:.4f} "
                      f"R@10={recalls[10]:.4f} R@100={recalls[100]:.4f}")
            except Exception as e:
                print(f"    [{variant:7s}] FAILED: {e}")
                recalls = {1: -1, 10: -1, 100: -1}

            results.append({
                "method": method_name, "index_type": index_type,
                "dataset": dataset, "backbone": backbone,
                "direction": direction, "variant": variant,
                "gt_protocol": GT_PROTOCOL,
                "r1": recalls[1], "r10": recalls[10], "r100": recalls[100],
            })


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    results = []

    # MSCOCO (CLIP-B/32, d=512)
    print("\n" + "="*70 + "\n  MSCOCO (CLIP-B/32, d=512)\n" + "="*70)
    img = l2_normalize(np.load(FEATURES_DIR / "mscoco_karpathy_val5k_clip_image_seed42.npy"))
    txt = l2_normalize(np.load(FEATURES_DIR / "mscoco_karpathy_val5k_clip_text_seed42.npy"))
    run_experiment(img, txt, "mscoco", "clip-b32", "text2image", results)
    run_experiment(txt, img, "mscoco", "clip-b32", "image2text", results)

    # AudioCaps (ImageBind, d=1024)
    # Asymmetric: 884 unique clips / 4415 captions
    # t→a: DB=audio_single(884), Q=text_full(4415)
    # a→t: DB=text_full(4415), Q=audio_single(884)
    print("\n" + "="*70 + "\n  AudioCaps (ImageBind, d=1024)\n" + "="*70)
    audio_single_path = FEATURES_DIR / "audiocaps_test_imagebind_audio_single_seed42.npy"
    text_full_path = FEATURES_DIR / "audiocaps_test_imagebind_text_seed42.npy"
    audio_single_884_path = FEATURES_DIR / "audiocaps_test_imagebind_audio_single_seed42.npy"
    if audio_single_path.exists() and text_full_path.exists():
        audio_single = l2_normalize(np.load(audio_single_path))  # (884, 1024)
        text_full = l2_normalize(np.load(text_full_path))        # (4415, 1024)
        print(f"  Audio(single): {audio_single.shape}, Text(full): {text_full.shape}")
        run_experiment(audio_single, text_full, "audiocaps", "imagebind", "text2audio", results)
        run_experiment(text_full, audio_single, "audiocaps", "imagebind", "audio2text", results)
    else:
        print(f"  WARNING: AudioCaps features not found, skipping")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["method", "index_type", "dataset", "backbone", "direction",
              "variant", "gt_protocol", "r1", "r10", "r100"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {OUTPUT_CSV}  ({len(results)} rows)")

    # Summary: Table 1 (sign-bit methods) format
    print("\n" + "="*80)
    print("  TABLE 1 FORMAT — ALL original_exact_ip GT")
    print("  Each cell: Van R@10 / PMC R@10")
    print("="*80)

    datasets_dirs = [
        ("mscoco", [("text2image", "t→i"), ("image2text", "i→t")]),
        ("audiocaps", [("text2audio", "t→a"), ("audio2text", "a→t")]),
    ]

    for method in ["PureBinary", "BinaryIVF", "RotatedBinary", "RaBitQ"]:
        print(f"\n  {method}:")
        for ds, dirs in datasets_dirs:
            for dir_key, dir_label in dirs:
                van = [r for r in results if r["method"] == method
                       and r["dataset"] == ds and r["direction"] == dir_key
                       and r["variant"] == "vanilla"]
                pmc = [r for r in results if r["method"] == method
                       and r["dataset"] == ds and r["direction"] == dir_key
                       and r["variant"] == "pmc_a1"]
                if van and pmc:
                    v10, p10 = van[0]["r10"], pmc[0]["r10"]
                    v100, p100 = van[0]["r100"], pmc[0]["r100"]
                    d10 = p10 - v10
                    d100 = p100 - v100
                    print(f"    {ds:10s} {dir_label:5s}  "
                          f"R@10: {v10:.3f}/{p10:.3f} ({d10:+.3f})  "
                          f"R@100: {v100:.3f}/{p100:.3f} ({d100:+.3f})")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nRuntime: {time.time() - t0:.1f}s")
