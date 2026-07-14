"""
reproduce_gap_energy.py -- Reproduce gap energy concentration analysis (Table/claim).

For each dataset/backbone pair, compute the modality gap vector g = query_mean - db_mean,
then measure how much energy is concentrated in the top k% of dimensions (sorted by g^2).

Paper claims (Sections 2-3, CLIP encoders):
  - top 10% of dims capture 86-92% of gap energy (E(10%), across CLIP datasets)
  - top 5% capture ~86-87% (MSCOCO)
  - ImageBind distributes energy more uniformly (~70-74% in top 10%)

Datasets:
  Small:    MSCOCO (CLIP-B/32, CLIP-L/14, ImageBind)
            Flickr30K (CLIP-B/32, CLIP-L/14)
            AudioCaps (ImageBind), Clotho (ImageBind)
  Large:    LAION-400M (CLIP-B/32, first 10 shards per modality)

Output:
  - Formatted table to stdout
  - CSV saved to results/gap_energy_all_datasets.csv

Usage:
  python scripts/reproduce_gap_energy.py
"""

from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = next(
    p for p in Path(__file__).resolve().parents
    if (p / "src").is_dir() and (p / "config").is_dir()
)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.runtime.config import CFG  # noqa: E402

# ---------------------------------------------------------------------------
# LAION path
# ---------------------------------------------------------------------------
LAION_DIR = Path("E:/laion400m")

# ---------------------------------------------------------------------------
# Small-dataset specs: (label, backbone, db_file, query_file)
# ---------------------------------------------------------------------------
SMALL_DATASETS = [
    (
        "MSCOCO",
        "CLIP-B/32",
        "mscoco_karpathy_val5k_clip_image_seed42.npy",
        "mscoco_karpathy_val5k_clip_text_seed42.npy",
    ),
    (
        "MSCOCO",
        "CLIP-L/14",
        "mscoco_karpathy_val5k_clip-l_image_seed42.npy",
        "mscoco_karpathy_val5k_clip-l_text_seed42.npy",
    ),
    (
        "MSCOCO",
        "ImageBind",
        "mscoco_karpathy_val5k_imagebind_image_seed42.npy",
        "mscoco_karpathy_val5k_imagebind_text_seed42.npy",
    ),
    (
        "Flickr30K",
        "CLIP-B/32",
        "flickr30k_test1k_clip_image_seed42.npy",
        "flickr30k_test1k_clip_text_seed42.npy",
    ),
    (
        "Flickr30K",
        "CLIP-L/14",
        "flickr30k_test1k_clip-l_image_seed42.npy",
        "flickr30k_test1k_clip-l_text_seed42.npy",
    ),
    (
        "AudioCaps",
        "ImageBind",
        "audiocaps_test_imagebind_audio_seed42.npy",
        "audiocaps_test_imagebind_text_seed42.npy",
    ),
    (
        "Clotho",
        "ImageBind",
        "clotho_eval_imagebind_audio_seed42.npy",
        "clotho_eval_imagebind_text_5cap_seed42.npy",
    ),
]

PERCENTAGES = [1, 2, 5, 10, 20, 50]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_gap_energy(db_mean: np.ndarray, query_mean: np.ndarray) -> dict:
    """Return gap energy fractions for each percentage threshold."""
    g = query_mean.astype(np.float64) - db_mean.astype(np.float64)
    g_sq = g ** 2
    total_energy = g_sq.sum()
    d = len(g)
    gap_norm = float(np.linalg.norm(g))

    sorted_idx = np.argsort(g_sq)[::-1]

    result: dict = {
        "dim": d,
        "gap_norm": gap_norm,
        "total_energy": float(total_energy),
    }
    for pct in PERCENTAGES:
        k = max(1, int(pct / 100 * d))
        frac = float(g_sq[sorted_idx[:k]].sum() / total_energy)
        result[f"top{pct}pct"] = frac
    return result


def load_npy_mean(path: Path) -> np.ndarray:
    """Load .npy file and return the mean vector (float64)."""
    arr = np.load(str(path))
    return arr.mean(axis=0).astype(np.float64)


# ---------------------------------------------------------------------------
# Small datasets
# ---------------------------------------------------------------------------

def process_small_datasets(features_dir: Path) -> list[dict]:
    rows = []
    for dataset, backbone, db_fname, q_fname in SMALL_DATASETS:
        db_path = features_dir / db_fname
        q_path = features_dir / q_fname

        if not db_path.exists():
            print(f"  [SKIP] Missing db file: {db_path.name}")
            continue
        if not q_path.exists():
            print(f"  [SKIP] Missing query file: {q_path.name}")
            continue

        try:
            db_mean = load_npy_mean(db_path)
            q_mean = load_npy_mean(q_path)
        except Exception as exc:
            print(f"  [SKIP] Error loading {dataset}/{backbone}: {exc}")
            continue

        stats = compute_gap_energy(db_mean, q_mean)
        row = {
            "dataset": dataset,
            "backbone": backbone,
            "source": "small",
            **stats,
        }
        rows.append(row)
        print(f"  [OK] {dataset} / {backbone}  dim={stats['dim']}  gap_norm={stats['gap_norm']:.4f}")

    return rows


# ---------------------------------------------------------------------------
# LAION-400M (sharded)
# ---------------------------------------------------------------------------

def accumulate_mean_sharded(
    shard_dir: Path,
    prefix: str,
    n_shards: int = 10,
) -> tuple[np.ndarray, int]:
    """
    Accumulate sum over n_shards shards (one at a time) and return
    (mean_float32, total_count).  Shards are float16; sum accumulated in float64.
    """
    running_sum: np.ndarray | None = None
    running_count = 0

    for i in range(n_shards):
        shard_path = shard_dir / f"{prefix}_{i}.npy"
        if not shard_path.exists():
            print(f"    [WARN] Shard not found: {shard_path.name} -- stopping at shard {i}")
            break
        shard = np.load(str(shard_path)).astype(np.float64)  # (N, D)
        if running_sum is None:
            running_sum = shard.sum(axis=0)
        else:
            running_sum += shard.sum(axis=0)
        running_count += shard.shape[0]
        del shard  # free memory

    if running_sum is None or running_count == 0:
        raise RuntimeError(f"No shards loaded from {shard_dir} with prefix={prefix}")

    mean_vec = (running_sum / running_count).astype(np.float32)
    return mean_vec, running_count


def process_laion(laion_dir: Path, n_shards: int = 10) -> list[dict]:
    rows = []

    img_emb_dir = laion_dir / "img_emb"
    text_emb_dir = laion_dir / "text_emb"

    if not img_emb_dir.exists():
        print(f"  [SKIP] LAION img_emb dir not found: {img_emb_dir}")
        return rows
    if not text_emb_dir.exists():
        print(f"  [SKIP] LAION text_emb dir not found: {text_emb_dir}")
        return rows

    print(f"  Loading LAION img_emb ({n_shards} shards) ...")
    try:
        img_mean, img_count = accumulate_mean_sharded(img_emb_dir, "img_emb", n_shards)
    except Exception as exc:
        print(f"  [SKIP] LAION img_emb error: {exc}")
        return rows

    print(f"  Loading LAION text_emb ({n_shards} shards) ...")
    try:
        text_mean, text_count = accumulate_mean_sharded(text_emb_dir, "text_emb", n_shards)
    except Exception as exc:
        print(f"  [SKIP] LAION text_emb error: {exc}")
        return rows

    print(f"  img vectors: {img_count:,}  text vectors: {text_count:,}")

    stats = compute_gap_energy(img_mean, text_mean)
    row = {
        "dataset": "LAION-400M",
        "backbone": "CLIP-B/32",
        "source": f"laion_{n_shards}shards",
        **stats,
    }
    rows.append(row)
    print(f"  [OK] LAION-400M / CLIP-B/32  dim={stats['dim']}  gap_norm={stats['gap_norm']:.4f}")
    return rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_table(rows: list[dict]) -> None:
    pct_cols = [f"top{p}pct" for p in PERCENTAGES]
    hdr = (
        f"{'Dataset':<14} {'Backbone':<12} {'Dim':>4} {'GapNorm':>8} "
        + "  ".join(f"top{p:2d}%" for p in PERCENTAGES)
    )
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        pct_str = "  ".join(f"{r[c]*100:6.1f}%" for c in pct_cols)
        print(
            f"{r['dataset']:<14} {r['backbone']:<12} {r['dim']:>4d} {r['gap_norm']:>8.4f}  {pct_str}"
        )
    print()


def print_summary(rows: list[dict]) -> None:
    """Print encoder-level summary showing CLIP vs ImageBind concentration."""
    clip_rows = [r for r in rows if "CLIP" in r["backbone"]]
    ib_rows = [r for r in rows if r["backbone"] == "ImageBind"]

    if clip_rows:
        t10_vals = [r["top10pct"] for r in clip_rows]
        t5_vals = [r["top5pct"] for r in clip_rows]
        print(f"CLIP encoders (n={len(clip_rows)}):  "
              f"top10% = {min(t10_vals)*100:.0f}--{max(t10_vals)*100:.0f}%  "
              f"top5% = {min(t5_vals)*100:.0f}--{max(t5_vals)*100:.0f}%")

    if ib_rows:
        t10_vals = [r["top10pct"] for r in ib_rows]
        t5_vals = [r["top5pct"] for r in ib_rows]
        print(f"ImageBind (n={len(ib_rows)}):      "
              f"top10% = {min(t10_vals)*100:.0f}--{max(t10_vals)*100:.0f}%  "
              f"top5% = {min(t5_vals)*100:.0f}--{max(t5_vals)*100:.0f}%")

    print()


def save_csv(rows: list[dict], out_path: Path) -> None:
    pct_cols = [f"top{p}pct" for p in PERCENTAGES]
    fieldnames = ["dataset", "backbone", "source", "dim", "gap_norm"] + pct_cols
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings("ignore")

    features_dir = CFG.features_dir
    results_dir = CFG.results_dir
    csv_out = results_dir / "gap_energy_all_datasets.csv"

    print("=" * 70)
    print("Gap Energy Concentration Analysis -- All Datasets")
    print("=" * 70)
    print(f"features_dir : {features_dir}")
    print(f"laion_dir    : {LAION_DIR}")
    print(f"output csv   : {csv_out}")
    print()

    all_rows: list[dict] = []

    print("[1/2] Small datasets")
    print("-" * 40)
    small_rows = process_small_datasets(features_dir)
    all_rows.extend(small_rows)

    print()
    print("[2/2] LAION-400M (first 10 shards per modality)")
    print("-" * 40)
    laion_rows = process_laion(LAION_DIR, n_shards=10)
    all_rows.extend(laion_rows)

    if not all_rows:
        print("ERROR: No datasets processed successfully.")
        sys.exit(1)

    print_table(all_rows)
    print_summary(all_rows)
    save_csv(all_rows, csv_out)


if __name__ == "__main__":
    main()
