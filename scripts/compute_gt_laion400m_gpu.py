"""
compute_gt_laion400m_gpu.py  --  GPU-accelerated brute-force ground truth
                                 for LAION-400M (RTX 3080, 10 GB VRAM).

Computes inner-product (cosine on unit vectors) ground truth for 10K query
vectors sampled from text_emb_0.npy against 410 image shards (~400M vectors).

Query sampling is IDENTICAL to reproduce_laion400m.py so the resulting GT
file is a valid drop-in for load_or_compute_gt().

Output files (compatible with reproduce_laion400m.py):
  E:/laion400m/groundtruth.laion400m.10K.npy       (10000, 100)  int64
  E:/laion400m/groundtruth.laion400m.10K.n_db.txt  plain integer: total n_db

Usage
-----
  python scripts/compute_gt_laion400m_gpu.py
  python scripts/compute_gt_laion400m_gpu.py --laion-dir F:/laion400m
  python scripts/compute_gt_laion400m_gpu.py --dry-run  # check GPU, skip compute
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import faiss

# ---------------------------------------------------------------------------
# Constants  (must match reproduce_laion400m.py exactly for determinism)
# ---------------------------------------------------------------------------
SEED = 42
N_QUERIES = 10_000
N_IMG_SHARDS = 410
TOP_K = 100
DIM = 512

_DEFAULT_LAION_DIR = Path("E:/laion400m")
LAION_DIR = Path(os.environ.get("LAION400M_DIR", str(_DEFAULT_LAION_DIR)))

# GPU temp memory: 512 MB is enough for 1M × 512d per shard search
_GPU_TEMP_MB = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def l2_normalize(arr: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization.  Returns float32 array."""
    arr = arr.astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)  # avoid div-by-zero
    return arr / norms


def ensure_float32_c(arr: np.ndarray) -> np.ndarray:
    """Return a float32 C-contiguous copy (required by FAISS)."""
    return np.ascontiguousarray(arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# Query loading  (IDENTICAL logic to reproduce_laion400m.py)
# ---------------------------------------------------------------------------

def load_queries(laion_dir: Path) -> np.ndarray:
    """Load 10K query vectors from text_emb_0.npy.

    Steps (must match reproduce_laion400m.py):
      1. Load text_emb_0.npy (float16)
      2. Cast to float32
      3. L2-normalize the full shard
      4. np.random.seed(SEED), sample N_QUERIES without replacement
      5. Return the sampled rows (already normalized)

    Returns
    -------
    (N_QUERIES, 512) float32 L2-normalized array
    """
    text_path = laion_dir / "text_emb" / "text_emb_0.npy"
    if not text_path.exists():
        sys.exit(f"[ERROR] Text shard not found: {text_path}")

    print(f"[queries] Loading {text_path} ...", flush=True)
    t0 = time.perf_counter()

    raw = np.load(str(text_path))               # float16
    text_f32 = l2_normalize(raw.astype(np.float32))
    del raw

    np.random.seed(SEED)
    idx = np.random.choice(len(text_f32), size=N_QUERIES, replace=False)
    queries = text_f32[idx].copy()
    del text_f32
    gc.collect()

    elapsed = time.perf_counter() - t0
    print(
        f"  queries: shape={queries.shape}  dtype={queries.dtype}  "
        f"({elapsed:.1f}s)",
        flush=True,
    )
    return queries


# ---------------------------------------------------------------------------
# Shard loading
# ---------------------------------------------------------------------------

def load_shard(laion_dir: Path, shard_idx: int) -> np.ndarray:
    """Load one image shard, cast float16 -> float32, L2-normalize.

    Returns
    -------
    (N_shard, 512) float32 unit vectors
    """
    path = laion_dir / "img_emb" / f"img_emb_{shard_idx}.npy"
    raw = np.load(str(path))               # float16
    return l2_normalize(raw.astype(np.float32))


def shard_exists(laion_dir: Path, shard_idx: int) -> bool:
    path = laion_dir / "img_emb" / f"img_emb_{shard_idx}.npy"
    return path.exists()


# ---------------------------------------------------------------------------
# Incremental merge helper
# ---------------------------------------------------------------------------

def _incremental_merge(
    running_dists: np.ndarray | None,
    running_ids: np.ndarray | None,
    new_dists: np.ndarray,
    new_ids: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge running top-K with new shard results.

    Concatenates along axis-1, argsorts descending, keeps top_k.
    Uses only (n_queries, 2*top_k) memory at any time.

    Parameters
    ----------
    running_dists / running_ids : accumulated top-K so far (None on first shard)
    new_dists / new_ids         : results from the latest shard

    Returns
    -------
    (merged_dists, merged_ids) both shape (n_queries, top_k)
    """
    if running_dists is None:
        # First shard: nothing to merge
        cat_d = new_dists
        cat_i = new_ids
    else:
        cat_d = np.concatenate([running_dists, new_dists], axis=1)
        cat_i = np.concatenate([running_ids,   new_ids],   axis=1)

    order = np.argsort(-cat_d, axis=1)[:, :top_k]
    return (
        np.take_along_axis(cat_d, order, axis=1),
        np.take_along_axis(cat_i, order, axis=1),
    )


# ---------------------------------------------------------------------------
# GPU ground truth computation
# ---------------------------------------------------------------------------

def compute_gt_gpu(
    queries: np.ndarray,
    laion_dir: Path,
    top_k: int = TOP_K,
) -> tuple[np.ndarray, int]:
    """Compute brute-force inner-product ground truth on GPU (RTX 3080).

    Strategy: incremental merge — never keep more than (n_queries, 2*top_k)
    distance/id buffers in RAM at once.  Each shard is ~2 GB float32 which
    fits comfortably in 10 GB VRAM.

    Parameters
    ----------
    queries   : (N_QUERIES, 512) float32 unit vectors
    laion_dir : root directory containing img_emb/
    top_k     : number of nearest neighbours to return

    Returns
    -------
    gt    : (N_QUERIES, top_k) int64 global DB indices (inner-product order)
    n_db  : total number of database vectors processed
    """
    n_queries, d = queries.shape
    queries_c = ensure_float32_c(queries)

    # -- GPU setup -----------------------------------------------------------
    use_gpu = False
    try:
        n_gpus = faiss.get_num_gpus()
        if n_gpus > 0:
            res = faiss.StandardGpuResources()
            res.setTempMemory(_GPU_TEMP_MB * 1024 * 1024)
            use_gpu = True
            print(
                f"[GT-GPU] Using GPU (device 0, {n_gpus} GPU(s) detected).  "
                f"TempMemory={_GPU_TEMP_MB} MB",
                flush=True,
            )
        else:
            print("[GT-GPU] No GPU detected -- falling back to CPU.", flush=True)
    except Exception as exc:
        print(f"[GT-GPU] GPU init failed ({exc}) -- falling back to CPU.", flush=True)

    # -- Incremental merge state --------------------------------------------
    running_dists: np.ndarray | None = None
    running_ids: np.ndarray | None = None
    cumulative_offset = 0

    t_total_start = time.perf_counter()

    for shard_idx in range(N_IMG_SHARDS):
        if not shard_exists(laion_dir, shard_idx):
            print(
                f"[GT-GPU] shard {shard_idx} missing -- stopping at {shard_idx} shards",
                flush=True,
            )
            break

        t0 = time.perf_counter()

        # Load and normalize shard
        shard_vecs = load_shard(laion_dir, shard_idx)
        n_shard = len(shard_vecs)
        shard_c = ensure_float32_c(shard_vecs)
        del shard_vecs

        # Build FAISS index and search
        cpu_index = faiss.IndexFlatIP(d)
        cpu_index.add(shard_c)
        del shard_c

        gpu_index = None
        try:
            if use_gpu:
                gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                dists, local_ids = gpu_index.search(queries_c, top_k)
                del gpu_index
                gpu_index = None
            else:
                dists, local_ids = cpu_index.search(queries_c, top_k)
        except Exception as exc:
            # GPU search failed: fall back to CPU for this shard
            if gpu_index is not None:
                del gpu_index
            print(
                f"[GT-GPU] GPU search failed on shard {shard_idx} ({exc})"
                f" -- retrying on CPU",
                flush=True,
            )
            dists, local_ids = cpu_index.search(queries_c, top_k)

        del cpu_index
        gc.collect()

        # Remap local IDs to global IDs
        valid = local_ids >= 0
        global_ids = np.where(
            valid, local_ids + cumulative_offset, -1
        ).astype(np.int64)

        # Incremental merge: keep running top-K
        running_dists, running_ids = _incremental_merge(
            running_dists, running_ids,
            dists.astype(np.float32), global_ids,
            top_k,
        )

        cumulative_offset += n_shard
        elapsed = time.perf_counter() - t0

        print(
            f"[GT-GPU] shard {shard_idx + 1}/{N_IMG_SHARDS}  "
            f"offset={cumulative_offset - n_shard:,}  "
            f"n_shard={n_shard:,}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    n_db = cumulative_offset
    total_elapsed = time.perf_counter() - t_total_start

    if running_ids is None:
        sys.exit("[ERROR] No shards were processed.")

    gt = running_ids.astype(np.int64)

    print(
        f"\n[GT-GPU] All shards done.  n_db={n_db:,}  gt.shape={gt.shape}  "
        f"total_time={total_elapsed:.1f}s",
        flush=True,
    )
    return gt, n_db


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_gt(gt: np.ndarray, n_db: int, laion_dir: Path) -> None:
    """Save GT array and n_db sidecar to paths expected by load_or_compute_gt()."""
    gt_path = laion_dir / f"groundtruth.laion400m.{N_QUERIES // 1000}K.npy"
    ndb_path = gt_path.with_suffix(".n_db.txt")

    laion_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(gt_path), gt.astype(np.int64))
    ndb_path.write_text(str(n_db))

    print(f"[save] GT  -> {gt_path}  shape={gt.shape}  dtype={gt.dtype}", flush=True)
    print(f"[save] ndb -> {ndb_path}  value={n_db:,}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(laion_dir: Path, dry_run: bool = False) -> None:
    t_wall_start = time.perf_counter()

    print("=" * 70, flush=True)
    print("compute_gt_laion400m_gpu.py  --  GPU brute-force GT for LAION-400M")
    print(f"  LAION_DIR   : {laion_dir}")
    print(f"  N_QUERIES   : {N_QUERIES:,}  (seed={SEED})")
    print(f"  N_IMG_SHARDS: {N_IMG_SHARDS}")
    print(f"  TOP_K       : {TOP_K}")
    print(f"  DIM         : {DIM}")
    print("=" * 70, flush=True)

    # Verify directories exist
    img_dir = laion_dir / "img_emb"
    text_dir = laion_dir / "text_emb"
    if not img_dir.is_dir():
        sys.exit(f"[ERROR] Image shard directory not found: {img_dir}")
    if not text_dir.is_dir():
        sys.exit(f"[ERROR] Text shard directory not found: {text_dir}")

    # Count available shards
    n_available = sum(shard_exists(laion_dir, i) for i in range(N_IMG_SHARDS))
    print(f"[data] Image shards available: {n_available}/{N_IMG_SHARDS}", flush=True)
    if n_available == 0:
        sys.exit("[ERROR] No image shards found.")

    # GPU check
    try:
        n_gpus = faiss.get_num_gpus()
        print(f"[GPU] faiss.get_num_gpus() = {n_gpus}", flush=True)
    except Exception as exc:
        print(f"[GPU] faiss.get_num_gpus() failed: {exc}", flush=True)

    if dry_run:
        print("\n[dry-run] Checks complete.  Exiting without computation.", flush=True)
        return

    # Check for existing GT and offer to skip
    gt_path = laion_dir / f"groundtruth.laion400m.{N_QUERIES // 1000}K.npy"
    ndb_path = gt_path.with_suffix(".n_db.txt")
    if gt_path.exists() and ndb_path.exists():
        print(
            f"\n[GT] Existing GT found: {gt_path}",
            flush=True,
        )
        print(
            "  To recompute, delete the files above and re-run this script.",
            flush=True,
        )
        gt = np.load(str(gt_path))
        n_db = int(ndb_path.read_text().strip())
        print(f"  Loaded: shape={gt.shape}  n_db={n_db:,}", flush=True)
        print(
            f"\n[wall] Total wall-clock time: "
            f"{time.perf_counter() - t_wall_start:.1f}s",
            flush=True,
        )
        return

    # Load queries
    queries = load_queries(laion_dir)

    # Compute GT on GPU
    gt, n_db = compute_gt_gpu(queries, laion_dir, top_k=TOP_K)

    # Save
    save_gt(gt, n_db, laion_dir)

    print(
        f"\n[wall] Total wall-clock time: "
        f"{time.perf_counter() - t_wall_start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPU-accelerated brute-force GT for LAION-400M (RTX 3080)"
    )
    parser.add_argument(
        "--laion-dir",
        type=Path,
        default=LAION_DIR,
        help=f"Root directory for LAION-400M data (default: {LAION_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check GPU availability and shard counts, then exit without computing.",
    )
    args = parser.parse_args()

    main(laion_dir=args.laion_dir, dry_run=args.dry_run)
