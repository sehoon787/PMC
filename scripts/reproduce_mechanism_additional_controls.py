"""Reproduce Table 4 additional mechanism controls.

Outputs:
- mechanism_additional_controls.csv

This lightweight MSCOCO/CLIP-B32 script records the controls used in Table 4:
component ablation, no-normalization, sign-flipped gap, same-norm random shift,
and shuffled gap. Recall is evaluated against original exact-IP ground truth.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

GT_PROTOCOL = "original_exact_ip"
TOP_K = 100
SEEDS = [0, 1, 2, 3, 4]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-8)).astype(np.float32)


def compute_topk_exact_ip(db: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    index = faiss.IndexFlatIP(db.shape[1])
    index.add(np.ascontiguousarray(db, dtype=np.float32))
    _, ids = index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
    return ids


def recall_to_gt(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    hit = 0
    for i in range(pred.shape[0]):
        hit += len(set(pred[i, :k].tolist()) & set(gt[i, :k].tolist()))
    return float(hit) / float(pred.shape[0] * k)


def float_to_binary(vecs: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    bits = (vecs > centroid).astype(np.uint8)
    pad = (8 - bits.shape[1] % 8) % 8
    if pad > 0:
        bits = np.hstack([bits, np.zeros((bits.shape[0], pad), dtype=np.uint8)])
    return np.packbits(bits, axis=1).copy()


def search_binary_flat(db: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    db_bin = float_to_binary(db, db.mean(axis=0))
    q_bin = float_to_binary(queries, db.mean(axis=0))
    index = faiss.IndexBinaryFlat(db_bin.shape[1] * 8)
    index.add(db_bin)
    _, ids = index.search(q_bin, k)
    return ids


def search_ivf_rabitq(db: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    d = db.shape[1]
    nlist = min(64, max(4, int(np.sqrt(db.shape[0]))))
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFRaBitQ(quantizer, d, nlist)
    index.nprobe = min(16, nlist)
    db_c = np.ascontiguousarray(db, dtype=np.float32)
    q_c = np.ascontiguousarray(queries, dtype=np.float32)
    index.train(db_c)
    index.add(db_c)
    _, ids = index.search(q_c, k)
    return ids


def add_rows(
    rows: List[Dict[str, object]],
    dataset: str,
    backbone: str,
    direction: str,
    mode: str,
    seed: int | None,
    db_v: np.ndarray,
    q_v: np.ndarray,
    gt: np.ndarray,
    gap_norm: float,
    note: str,
) -> None:
    for index_type, search_fn in [
        ("binary_flat", search_binary_flat),
        ("ivf_rabitq", search_ivf_rabitq),
    ]:
        pred = search_fn(db_v, q_v, TOP_K)
        rows.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "direction": direction,
                "index_type": index_type,
                "mode": mode,
                "seed": "" if seed is None else seed,
                "r10": f"{recall_to_gt(pred, gt, 10):.6f}",
                "r100": f"{recall_to_gt(pred, gt, 100):.6f}",
                "gap_norm": f"{gap_norm:.6f}",
                "db_mean_norm": f"{float(np.linalg.norm(db_v.mean(axis=0))):.6f}",
                "query_mean_norm": f"{float(np.linalg.norm(q_v.mean(axis=0))):.6f}",
                "gt_protocol": GT_PROTOCOL,
                "note": note,
            }
        )


def run_direction(dataset: str, backbone: str, direction: str, db_orig: np.ndarray, q_orig: np.ndarray) -> List[Dict[str, object]]:
    gt = compute_topk_exact_ip(db_orig, q_orig, TOP_K)
    gap = q_orig.mean(axis=0) - db_orig.mean(axis=0)
    gap_norm = float(np.linalg.norm(gap))
    rows: List[Dict[str, object]] = []

    base_variants: List[Tuple[str, np.ndarray, np.ndarray, str]] = [
        ("vanilla", db_orig, q_orig, "baseline no shift"),
        ("query_only", db_orig, l2_normalize(q_orig - gap), "normalized query-side shift only"),
        ("db_only", l2_normalize(db_orig + gap), q_orig, "normalized DB-side PMC alpha=1"),
        ("both", l2_normalize(db_orig + 0.5 * gap), l2_normalize(q_orig - 0.5 * gap), "normalized split correction alpha=0.5"),
        ("query_only_no_norm", db_orig, (q_orig - gap).astype(np.float32), "query-side shift without post-shift L2 normalization"),
        ("db_only_no_norm", (db_orig + gap).astype(np.float32), q_orig, "DB-side shift without post-shift L2 normalization"),
        ("both_no_norm", (db_orig + 0.5 * gap).astype(np.float32), (q_orig - 0.5 * gap).astype(np.float32), "split correction without post-shift L2 normalization"),
        ("sign_flipped_gap", l2_normalize(db_orig - gap), q_orig, "same gap magnitude but opposite direction, DB-side"),
    ]
    for mode, db_v, q_v, note in base_variants:
        add_rows(rows, dataset, backbone, direction, mode, None, db_v, q_v, gt, gap_norm, note)

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        random_gap = rng.normal(size=gap.shape).astype(np.float32)
        random_gap = random_gap / max(float(np.linalg.norm(random_gap)), 1e-12) * gap_norm
        add_rows(
            rows,
            dataset,
            backbone,
            direction,
            "random_direction_same_norm",
            seed,
            l2_normalize(db_orig + random_gap),
            q_orig,
            gt,
            gap_norm,
            "DB-side random direction with same norm as gap",
        )

        shuffled_gap = gap[rng.permutation(gap.shape[0])].astype(np.float32)
        add_rows(
            rows,
            dataset,
            backbone,
            direction,
            "shuffled_gap",
            seed,
            l2_normalize(db_orig + shuffled_gap),
            q_orig,
            gt,
            gap_norm,
            "DB-side coordinate-shuffled gap",
        )
    return rows


def project_root() -> Path:
    path = Path(__file__).resolve()
    if path.parent.name == "research":
        return path.parents[2]
    return path.parents[1]


def resolve_features_dir(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    env_dir = os.environ.get("PMC_FEATURES_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return root / "data" / "features"


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "backbone",
        "direction",
        "index_type",
        "mode",
        "seed",
        "r10",
        "r100",
        "gap_norm",
        "db_mean_norm",
        "query_mean_norm",
        "gt_protocol",
        "note",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_table5_controls(results_path: Path) -> None:
    with open(results_path, newline="") as f:
        rows = list(csv.DictReader(f))
    expected = {
        ("text2image", "db_only"): 0.637,
        ("image2text", "db_only"): 0.608,
        ("text2image", "sign_flipped_gap"): 0.514,
        ("image2text", "sign_flipped_gap"): 0.491,
        ("text2image", "db_only_no_norm"): 0.555,
        ("image2text", "db_only_no_norm"): 0.520,
    }
    for (direction, mode), value in expected.items():
        match = [
            float(r["r100"])
            for r in rows
            if r["direction"] == direction and r["index_type"] == "ivf_rabitq" and r["mode"] == mode
        ]
        if not match:
            raise ValueError(f"missing IVF-RaBitQ control: {direction}/{mode}")
        if abs(round(match[0], 3) - value) > 5e-4:
            raise ValueError(f"{direction}/{mode}: expected {value:.3f}, got {match[0]:.6f}")
    print(f"[reproduce_mechanism_additional_controls] validation passed using {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Table 4 additional mechanism controls")
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--validate-paper", action="store_true")
    args = parser.parse_args()

    root = project_root()
    out_dir = args.results_dir.resolve() if args.results_dir else root / "results"
    out_path = out_dir / "mechanism_additional_controls.csv"
    if args.validate_paper:
        validate_table5_controls(out_path)
        return

    features_dir = resolve_features_dir(root, args.features_dir)
    image = l2_normalize(np.load(features_dir / "mscoco_karpathy_val5k_clip_image_seed42.npy").astype(np.float32))
    text = l2_normalize(np.load(features_dir / "mscoco_karpathy_val5k_clip_text_seed42.npy").astype(np.float32))

    faiss.omp_set_num_threads(1)
    rows: List[Dict[str, object]] = []
    rows.extend(run_direction("mscoco", "clip-b32", "text2image", image, text))
    rows.extend(run_direction("mscoco", "clip-b32", "image2text", text, image))
    write_csv(out_path, rows)
    validate_table5_controls(out_path)
    print(f"[reproduce_mechanism_additional_controls] features_dir={features_dir}")
    print(f"[reproduce_mechanism_additional_controls] wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[reproduce_mechanism_additional_controls][error] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
