"""
reproduce_laion400m.py -- PMC-RaBitQ on LAION-400M at N~400M scale.

Runs 3 methods (vanilla_rabitq, pmc_1.00, vanilla_rabitq_meanshift) across
nprobe sweep [1, 2, 4, 8, 16, 32, 64, 128] with proper timing (single-thread,
5 runs, median). Handles the 400M scale memory constraint by:
  - Reading image embeddings as 410 .npy shards (~1M vectors each, float16)
  - Computing ground truth shard-by-shard and merging top-100 per shard
  - Training on a 5M sample from the first 5 shards
  - Adding vectors shard-by-shard and freeing each after add
  - Queries: 10K random samples from text_emb_0.npy (first text shard, ~1M)

Data layout:
  E:/laion400m/img_emb/img_emb_{0..409}.npy   float16 (N_shard, 512)
  E:/laion400m/text_emb/text_emb_{0..9}.npy   float16 (N_shard, 512)

Output: results/pmc_laion400m_seed42.csv

Usage
-----
  python scripts/reproduce_laion400m.py             # full 400M experiment
  python scripts/reproduce_laion400m.py --summary   # print CSV summary and exit
  python scripts/reproduce_laion400m.py --gt-only   # compute and save GT only
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# MKL reads thread count at library init; must set BEFORE importing numpy/faiss
_N_BUILD_THREADS = str(os.cpu_count() or 8)
os.environ["MKL_NUM_THREADS"] = _N_BUILD_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _N_BUILD_THREADS
os.environ["OMP_NUM_THREADS"] = _N_BUILD_THREADS


def _set_threads(n: int) -> None:
    """Set threading for FAISS OMP + MKL/OpenBLAS."""
    import faiss as _f
    _f.omp_set_num_threads(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    try:
        import mkl
        mkl.set_num_threads(n)
    except ImportError:
        pass

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

from src.core.metrics import recall_at_k
from src.core.pmc import SimpleRaBitQIndex, compute_gap, shift_db_vectors, shift_query_vectors
from src.runtime.config import CFG
from src.utils import ensure_float32_c, l2_normalize, timed_search

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_IMG_SHARDS = 410
N_TEXT_SHARDS = 10
N_QUERIES = 10_000
TOP_K = 100
NLIST = 20_000          # approx sqrt(400M)
NPROBE_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
N_WARMUP = 1
N_TIMED = 5
TRAIN_SHARDS = 5        # first 5 shards for training sample (~5M vectors)
GAP_SAMPLE_SHARDS = 5   # same 5 shards for gap estimation

# Default data directory; override via LAION400M_DIR env var
_DEFAULT_LAION_DIR = Path("E:/laion400m")
LAION_DIR = Path(os.environ.get("LAION400M_DIR", str(_DEFAULT_LAION_DIR)))

RESULTS_DIR = CFG.results_dir
INDEX_CACHE_DIR = LAION_DIR / "index_cache"

FIELDNAMES = [
    "method", "alpha", "nprobe",
    "r1", "r10", "r100",
    "qps", "bytes_per_vec", "n_db", "d", "seed",
]


# ---------------------------------------------------------------------------
# Shard loading
# ---------------------------------------------------------------------------

def load_shard(kind: str, idx: int) -> np.ndarray:
    """Load one .npy shard and convert float16 -> float32.

    Parameters
    ----------
    kind : 'img' or 'text'
    idx  : shard index

    Returns
    -------
    (N_shard, 512) float32 array
    """
    subdir = "img_emb" if kind == "img" else "text_emb"
    path = LAION_DIR / subdir / f"{subdir}_{idx}.npy"
    arr = np.load(str(path))     # float16
    return arr.astype(np.float32)


def shard_exists(kind: str, idx: int) -> bool:
    """Check whether a shard file exists on disk."""
    subdir = "img_emb" if kind == "img" else "text_emb"
    path = LAION_DIR / subdir / f"{subdir}_{idx}.npy"
    return path.exists()


# ---------------------------------------------------------------------------
# Query loading (from first text shard)
# ---------------------------------------------------------------------------

def load_queries() -> np.ndarray:
    """Load 10K query vectors sampled from text_emb_0.npy.

    Loads ~1M vectors from the first text shard (2 GB in float32), samples
    N_QUERIES rows, normalises, and frees the rest.

    Returns
    -------
    (N_QUERIES, 512) float32 L2-normalised array
    """
    print(f"\n[queries] Loading text_emb_0.npy ...")
    t0 = time.perf_counter()
    text_shard = load_shard("text", 0)   # float32 after cast
    text_norm = l2_normalize(text_shard)
    del text_shard

    np.random.seed(SEED)
    query_idx = np.random.choice(len(text_norm), size=N_QUERIES, replace=False)
    queries = text_norm[query_idx].copy()
    del text_norm
    gc.collect()

    print(
        f"  queries: {queries.shape}  "
        f"({time.perf_counter() - t0:.1f}s)"
    )
    return queries


# ---------------------------------------------------------------------------
# Ground truth — shard-by-shard IndexFlatIP
# ---------------------------------------------------------------------------

def compute_gt_shards(
    queries: np.ndarray,
    n_img_shards: int,
    top_k: int,
) -> tuple[np.ndarray, int]:
    """Compute brute-force ground truth over all image shards.

    For each shard:
      1. Build IndexFlatIP on the shard (inner product on unit vectors = cosine)
      2. Search queries -> top_k per shard
      3. Remap local shard IDs to global IDs using the cumulative offset

    Then merge all shard results and keep global top_k per query.

    Returns
    -------
    gt : (n_queries, top_k) int64 array of global DB indices
    n_db : total number of vectors across all shards
    """
    n_queries, d = queries.shape
    queries_c = ensure_float32_c(queries)

    all_dists: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    cumulative_offset = 0

    for shard_idx in range(n_img_shards):
        if not shard_exists("img", shard_idx):
            print(f"  [GT] shard {shard_idx} missing -- stopping at {shard_idx} shards")
            break

        t0 = time.perf_counter()
        print(
            f"  [GT] shard {shard_idx + 1}/{n_img_shards}  "
            f"offset={cumulative_offset:,} ...",
            end=" ",
            flush=True,
        )
        shard_raw = load_shard("img", shard_idx)
        shard_vecs = l2_normalize(shard_raw)
        del shard_raw
        n_shard = len(shard_vecs)

        idx = faiss.IndexFlatIP(d)
        idx.add(ensure_float32_c(shard_vecs))
        del shard_vecs

        dists, local_ids = idx.search(queries_c, top_k)
        del idx
        gc.collect()

        valid = local_ids >= 0
        global_ids = np.where(
            valid, local_ids + cumulative_offset, -1
        ).astype(np.int64)
        all_dists.append(dists.astype(np.float32))
        all_ids.append(global_ids)

        cumulative_offset += n_shard
        elapsed = time.perf_counter() - t0
        print(f"n_shard={n_shard:,}  done ({elapsed:.1f}s)")

    n_db = cumulative_offset
    print(f"  [GT] total DB vectors: {n_db:,}")

    # Merge all shard results and keep global top_k
    merged_dists = np.concatenate(all_dists, axis=1)   # (Q, n_shards * top_k)
    merged_ids   = np.concatenate(all_ids,   axis=1)   # (Q, n_shards * top_k)

    order = np.argsort(-merged_dists, axis=1)
    gt = np.take_along_axis(merged_ids, order[:, :top_k], axis=1)
    return gt.astype(np.int64), n_db


def load_or_compute_gt(
    gt_path: Path,
    queries: np.ndarray,
    n_img_shards: int,
    top_k: int,
) -> tuple[np.ndarray, int]:
    """Load cached GT or compute and save it.

    Returns (gt, n_db).  n_db is stored alongside the GT file as a .txt
    sidecar so it can be recovered on subsequent runs.
    """
    ndb_path = gt_path.with_suffix(".n_db.txt")

    if gt_path.exists() and ndb_path.exists():
        print(f"\n[GT] Loading cached ground truth from {gt_path} ...")
        t0 = time.perf_counter()
        gt = np.load(str(gt_path))
        gt = gt[:len(queries)]
        n_db = int(ndb_path.read_text().strip())
        print(f"  gt shape={gt.shape}  n_db={n_db:,}  ({time.perf_counter() - t0:.1f}s)")
        return gt, n_db

    print(
        f"\n[GT] Computing shard-by-shard brute-force ground truth "
        f"(Q={len(queries)}, top_k={top_k}, shards={n_img_shards}) ..."
    )
    t0 = time.perf_counter()
    gt, n_db = compute_gt_shards(queries, n_img_shards, top_k)
    elapsed = time.perf_counter() - t0
    print(f"  gt shape={gt.shape}  total time={elapsed:.1f}s")
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(gt_path), gt)
    ndb_path.write_text(str(n_db))
    print(f"  Saved to {gt_path}")
    print(f"  n_db saved to {ndb_path}")
    return gt, n_db


# ---------------------------------------------------------------------------
# Gap estimation
# ---------------------------------------------------------------------------

def estimate_gap(queries: np.ndarray, n_shards: int = GAP_SAMPLE_SHARDS) -> np.ndarray:
    """Estimate modality gap from first n_shards image shards.

    Loads ~5M image vectors (float32), computes mean, then discards.

    Returns
    -------
    gap : (d,) float32  =  mean(text_queries) - mean(img_db_sample)
    """
    print(f"\n[Gap] Loading {n_shards} image shards for gap estimation ...")
    t0 = time.perf_counter()
    parts = []
    for i in range(n_shards):
        shard_raw = load_shard("img", i)
        parts.append(l2_normalize(shard_raw))
        del shard_raw

    db_sample = np.vstack(parts)
    del parts
    gc.collect()

    gap = compute_gap(db_sample, queries)
    gap_norm = float(np.linalg.norm(gap))
    del db_sample
    gc.collect()

    print(
        f"  db_sample: {5 * 1_000_000:,} approx vectors  "
        f"gap L2 norm = {gap_norm:.6f}  ({time.perf_counter() - t0:.1f}s)"
    )
    return gap


# ---------------------------------------------------------------------------
# Shard-based index building
# ---------------------------------------------------------------------------

def _ensure_c32(arr: np.ndarray) -> np.ndarray:
    return ensure_float32_c(arr)


def build_index_shards(
    n_img_shards: int,
    train_shards: int,
    nlist: int,
    seed: int,
    gap: np.ndarray | None,
    alpha: float,
    label: str,
) -> tuple[SimpleRaBitQIndex, int]:
    """Build an IVFRaBitQFastScan index by training on a shard sample then
    adding all shards one at a time.

    Parameters
    ----------
    n_img_shards : total number of image shards to add
    train_shards : number of shards to use for training (e.g. 5 -> ~5M vectors)
    nlist        : IVF cell count
    seed         : random seed
    gap          : (d,) gap vector for PMC shift; None means vanilla (no shift)
    alpha        : PMC alpha (used only if gap is not None)
    label        : display label for progress messages

    Returns
    -------
    (index_wrapper, d)
    """
    d = 512  # CLIP ViT-B/32

    # --- Train phase ---------------------------------------------------------
    print(f"\n[{label}] Loading {train_shards} shards for training ...")
    t0 = time.perf_counter()
    parts = []
    for i in range(train_shards):
        shard_raw = load_shard("img", i)
        parts.append(l2_normalize(shard_raw))
        del shard_raw

    train_vecs = np.vstack(parts)
    del parts

    if gap is not None:
        train_vecs = shift_db_vectors(train_vecs, gap, alpha=alpha)

    train_vecs = _ensure_c32(train_vecs)
    n_train = len(train_vecs)
    print(f"  training sample: {n_train:,} vectors  ({time.perf_counter() - t0:.1f}s)")

    quantizer = faiss.IndexFlatL2(d)
    raw_index = faiss.IndexIVFRaBitQFastScan(quantizer, d, nlist, 0)
    raw_index.cp.seed = seed
    raw_index.cp.min_points_per_centroid = 1

    # --- GPU-accelerated k-means training ---
    _use_gpu_train = False
    try:
        if faiss.get_num_gpus() > 0:
            _use_gpu_train = True
    except Exception:
        pass

    print(f"[{label}] Training (d={d}, nlist={nlist}) on {n_train:,} vectors "
          f"({'GPU' if _use_gpu_train else 'CPU'}) ...")
    t0 = time.perf_counter()

    # Single GPU resource instance shared across training + shard adds
    _gpu_res = None

    if _use_gpu_train:
        # GPU k-means: use faiss.Clustering with GpuIndexFlatL2
        clus = faiss.Clustering(d, nlist)
        clus.seed = seed
        clus.niter = 25
        clus.min_points_per_centroid = 1
        clus.verbose = True

        _gpu_res = faiss.StandardGpuResources()
        _gpu_res.setTempMemory(512 * 1024 * 1024)  # 512 MB scratch
        gpu_flat = faiss.GpuIndexFlatL2(_gpu_res, d)

        clus.train(train_vecs, gpu_flat)

        # Copy trained centroids into the CPU quantizer
        centroids = faiss.vector_float_to_array(clus.centroids).reshape(nlist, d)
        quantizer.add(centroids)

        # Free k-means objects but keep _gpu_res alive for shard adds
        del gpu_flat, clus
        gc.collect()

        # Train the RaBitQ codec part (not k-means, just the per-cell codec)
        raw_index.quantizer_trains_alone = 2  # skip k-means, quantizer already done
        raw_index.train(train_vecs)
    else:
        raw_index.train(train_vecs)

    del train_vecs
    if _gpu_res is not None:
        del _gpu_res
        _gpu_res = None
    gc.collect()
    print(f"  Training done  ({time.perf_counter() - t0:.1f}s)")

    # GPU quantizer for shard adds disabled — FAISS GPU StackDeviceMemory
    # assertion crash on Windows (both noTempMemory and index_cpu_to_gpu fail).
    # CPU quantizer uses MKL BLAS for IVF assignment — fast enough.

    # --- Add phase (shard by shard) ------------------------------------------
    for shard_idx in range(n_img_shards):
        if not shard_exists("img", shard_idx):
            print(f"[{label}] shard {shard_idx} missing -- stopping add phase")
            break

        t0 = time.perf_counter()
        print(
            f"[{label}] Adding shard {shard_idx + 1}/{n_img_shards} ...",
            end=" ",
            flush=True,
        )
        shard_raw = load_shard("img", shard_idx)
        shard_vecs = l2_normalize(shard_raw)
        del shard_raw

        if gap is not None:
            shard_vecs = shift_db_vectors(shard_vecs, gap, alpha=alpha)

        raw_index.add(_ensure_c32(shard_vecs))
        n_added = len(shard_vecs)
        del shard_vecs
        gc.collect()
        print(f"n={n_added:,}  done ({time.perf_counter() - t0:.1f}s)")

    # (quantizer stays on CPU — no GPU move needed)

    print(
        f"[{label}] Index built.  "
        f"code_size={raw_index.code_size} bytes/vec  "
        f"ntotal={raw_index.ntotal:,}"
    )
    return SimpleRaBitQIndex(raw_index, d), d


# ---------------------------------------------------------------------------
# CSV summary
# ---------------------------------------------------------------------------

def check_laion400m_csv_summary() -> bool:
    """Print a compact summary of LAION-400M R@100 from result CSV.

    Returns True if the CSV exists and was printed.

    Invoked via:
      python scripts/reproduce_laion400m.py --summary
    """
    csv_path = RESULTS_DIR / f"pmc_laion400m_seed{SEED}.csv"
    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found")
        return False

    print("\n=== LAION-400M CSV summary ===")
    print(f"  File: {csv_path}")
    print()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("  (empty file)")
        return False

    header = (
        f"{'Method':<30} {'alpha':>5} {'np':>4} "
        f"{'R@1':>6} {'R@10':>6} {'R@100':>6} {'QPS':>8} {'B/v':>4}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['method']:<30} {float(row['alpha']):>5.2f} {int(row['nprobe']):>4} "
            f"{float(row['r1']):>6.4f} {float(row['r10']):>6.4f} "
            f"{float(row['r100']):>6.4f} {float(row['qps']):>8.1f} "
            f"{int(row['bytes_per_vec']):>4}"
        )
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(gt_only: bool = False) -> None:
    np.random.seed(SEED)
    _set_threads(1)  # default; overridden to multi-thread during build

    print("=" * 70)
    print("PMC-RaBitQ  --  LAION-400M  N~400M  d=512  (single-thread search, multi-thread build)")
    print(f"  LAION_DIR : {LAION_DIR}")
    print(f"  RESULTS   : {RESULTS_DIR}")
    print("=" * 70)

    # Verify data directory
    img_dir = LAION_DIR / "img_emb"
    text_dir = LAION_DIR / "text_emb"
    if not img_dir.is_dir():
        print(f"\n[ERROR] Image shard directory not found: {img_dir}")
        print("  Run: python scripts/download_laion400m.py --img-only")
        print("  Or set LAION400M_DIR environment variable.")
        sys.exit(1)
    if not text_dir.is_dir():
        print(f"\n[ERROR] Text shard directory not found: {text_dir}")
        print("  Run: python scripts/download_laion400m.py --text-only")
        sys.exit(1)

    # Count available shards
    n_img_available = sum(
        1 for i in range(N_IMG_SHARDS) if shard_exists("img", i)
    )
    print(f"\n[data] Image shards available: {n_img_available}/{N_IMG_SHARDS}")
    if n_img_available == 0:
        print("[ERROR] No image shards found. Run download_laion400m.py first.")
        sys.exit(1)

    # Paths for ground truth and results
    gt_path = LAION_DIR / f"groundtruth.laion400m.{N_QUERIES // 1000}K.npy"
    out_csv = RESULTS_DIR / f"pmc_laion400m_seed{SEED}.csv"

    # Load queries
    queries = load_queries()
    d = queries.shape[1]

    # Ground truth
    gt, n_db = load_or_compute_gt(
        gt_path=gt_path,
        queries=queries,
        n_img_shards=n_img_available,
        top_k=TOP_K,
    )

    if gt_only:
        print("\n[gt-only] Ground truth saved. Exiting.")
        return

    # Gap estimation (uses same 5 shards as training)
    gap = estimate_gap(queries, n_shards=GAP_SAMPLE_SHARDS)

    # Prepare shifted query variants
    # vanilla_rabitq_meanshift: query-side shift on vanilla index
    #   q' = q - (1 - alpha=0) * gap = q - gap  (shift_query_vectors with alpha=0)
    q_meanshift = shift_query_vectors(queries, gap, alpha=0.0)

    # pmc_1.00: DB fully shifted to query center (alpha=1), queries unchanged
    q_pmc = shift_query_vectors(queries, gap, alpha=1.0)

    all_records: list[dict] = []

    # ------------------------------------------------------------------
    # Build vanilla index
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    vanilla_cache = INDEX_CACHE_DIR / f"vanilla_rabitq_nlist{NLIST}_seed{SEED}.index"
    if vanilla_cache.exists():
        print(f"[vanilla_rabitq] Loading cached index from {vanilla_cache} ...")
        t0 = time.perf_counter()
        raw = faiss.read_index(str(vanilla_cache))
        vanilla_idx = SimpleRaBitQIndex(raw, d)
        print(f"[vanilla_rabitq] Loaded in {time.perf_counter() - t0:.1f}s  ntotal={raw.ntotal:,}")
    else:
        print("[vanilla_rabitq] Building index ...")
        _set_threads(os.cpu_count() or 8)
        vanilla_idx, _ = build_index_shards(
            n_img_shards=n_img_available,
            train_shards=TRAIN_SHARDS,
            nlist=NLIST,
            seed=SEED,
            gap=None,
            alpha=0.0,
            label="vanilla_rabitq",
        )
        _set_threads(1)
        vanilla_cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"[vanilla_rabitq] Saving index to {vanilla_cache} ...", flush=True)
        faiss.write_index(vanilla_idx.index, str(vanilla_cache))
        print(f"[vanilla_rabitq] Saved ({vanilla_cache.stat().st_size / 1e9:.1f} GB)")
    bpv = vanilla_idx.bytes_per_vec()

    # ------------------------------------------------------------------
    # Sweep: vanilla_rabitq
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("[vanilla_rabitq] nprobe sweep ...")
    for nprobe in NPROBE_VALUES:
        _, retrieved, qps = timed_search(
            vanilla_idx, queries, TOP_K, nprobe,
            n_warmup=N_WARMUP, n_timed=N_TIMED,
        )
        r1   = recall_at_k(retrieved, gt, k=1)
        r10  = recall_at_k(retrieved, gt, k=10)
        r100 = recall_at_k(retrieved, gt, k=100)
        print(
            f"  nprobe={nprobe:>3}  "
            f"R@1={r1:.4f}  R@10={r10:.4f}  R@100={r100:.4f}  QPS={qps:.0f}"
        )
        all_records.append({
            "method": "vanilla_rabitq",
            "alpha": 0.0,
            "nprobe": nprobe,
            "r1": round(r1, 4),
            "r10": round(r10, 4),
            "r100": round(r100, 4),
            "qps": round(qps, 1),
            "bytes_per_vec": bpv,
            "n_db": n_db,
            "d": d,
            "seed": SEED,
        })

    # ------------------------------------------------------------------
    # Sweep: vanilla_rabitq_meanshift (query-side shift, same vanilla index)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("[vanilla_rabitq_meanshift] nprobe sweep (query-side shift, vanilla index) ...")
    for nprobe in NPROBE_VALUES:
        _, retrieved, qps = timed_search(
            vanilla_idx, q_meanshift, TOP_K, nprobe,
            n_warmup=N_WARMUP, n_timed=N_TIMED,
        )
        r1   = recall_at_k(retrieved, gt, k=1)
        r10  = recall_at_k(retrieved, gt, k=10)
        r100 = recall_at_k(retrieved, gt, k=100)
        print(
            f"  nprobe={nprobe:>3}  "
            f"R@1={r1:.4f}  R@10={r10:.4f}  R@100={r100:.4f}  QPS={qps:.0f}"
        )
        all_records.append({
            "method": "vanilla_rabitq_meanshift",
            "alpha": 0.0,
            "nprobe": nprobe,
            "r1": round(r1, 4),
            "r10": round(r10, 4),
            "r100": round(r100, 4),
            "qps": round(qps, 1),
            "bytes_per_vec": bpv,
            "n_db": n_db,
            "d": d,
            "seed": SEED,
        })

    # Free vanilla index before building PMC index to reduce peak RAM
    del vanilla_idx
    gc.collect()

    # ------------------------------------------------------------------
    # Build PMC index (alpha=1.00)
    # ------------------------------------------------------------------
    alpha = 1.0
    method_name = f"pmc_{alpha:.2f}"
    print(f"\n{'=' * 70}")
    pmc_cache = INDEX_CACHE_DIR / f"{method_name}_nlist{NLIST}_seed{SEED}.index"
    if pmc_cache.exists():
        print(f"[{method_name}] Loading cached index from {pmc_cache} ...")
        t0 = time.perf_counter()
        raw = faiss.read_index(str(pmc_cache))
        pmc_idx = SimpleRaBitQIndex(raw, d)
        print(f"[{method_name}] Loaded in {time.perf_counter() - t0:.1f}s  ntotal={raw.ntotal:,}")
    else:
        print(f"[{method_name}] Building PMC-shifted index (alpha={alpha}) ...")
        _set_threads(os.cpu_count() or 8)
        pmc_idx, _ = build_index_shards(
            n_img_shards=n_img_available,
            train_shards=TRAIN_SHARDS,
            nlist=NLIST,
            seed=SEED,
            gap=gap,
            alpha=alpha,
            label=method_name,
        )
        _set_threads(1)
        pmc_cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{method_name}] Saving index to {pmc_cache} ...", flush=True)
        faiss.write_index(pmc_idx.index, str(pmc_cache))
        print(f"[{method_name}] Saved ({pmc_cache.stat().st_size / 1e9:.1f} GB)")
    pmc_bpv = pmc_idx.bytes_per_vec()

    # ------------------------------------------------------------------
    # Sweep: pmc_1.00
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"[{method_name}] nprobe sweep (queries unchanged at alpha=1) ...")
    for nprobe in NPROBE_VALUES:
        _, retrieved, qps = timed_search(
            pmc_idx, q_pmc, TOP_K, nprobe,
            n_warmup=N_WARMUP, n_timed=N_TIMED,
        )
        r1   = recall_at_k(retrieved, gt, k=1)
        r10  = recall_at_k(retrieved, gt, k=10)
        r100 = recall_at_k(retrieved, gt, k=100)
        print(
            f"  nprobe={nprobe:>3}  "
            f"R@1={r1:.4f}  R@10={r10:.4f}  R@100={r100:.4f}  QPS={qps:.0f}"
        )
        all_records.append({
            "method": method_name,
            "alpha": alpha,
            "nprobe": nprobe,
            "r1": round(r1, 4),
            "r10": round(r10, 4),
            "r100": round(r100, 4),
            "qps": round(qps, 1),
            "bytes_per_vec": pmc_bpv,
            "n_db": n_db,
            "d": d,
            "seed": SEED,
        })

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print(f"SUMMARY TABLE  --  LAION-400M  N={n_db:,}  d={d}  PMC-RaBitQ")
    print("=" * 100)

    header = (
        f"{'Method':<30} {'alpha':>5} {'np':>4} "
        f"{'R@1':>6} {'R@10':>6} {'R@100':>6} {'QPS':>8} {'B/v':>4}"
    )
    print(header)
    print("-" * len(header))
    for r in all_records:
        print(
            f"{r['method']:<30} {r['alpha']:>5.2f} {r['nprobe']:>4} "
            f"{r['r1']:>6.4f} {r['r10']:>6.4f} {r['r100']:>6.4f} "
            f"{r['qps']:>8.1f} {r['bytes_per_vec']:>4}"
        )

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n[laion400m] CSV written -> {out_csv}")
    print(f"[laion400m] Done. {len(all_records)} records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PMC-RaBitQ LAION-400M experiment"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print LAION-400M CSV summary and exit.",
    )
    parser.add_argument(
        "--gt-only",
        action="store_true",
        help="Compute and save ground truth only, then exit.",
    )
    args = parser.parse_args()

    if args.summary:
        ok = check_laion400m_csv_summary()
        sys.exit(0 if ok else 1)

    main(gt_only=args.gt_only)
