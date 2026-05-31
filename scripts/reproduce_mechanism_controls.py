"""
Reproduce PMC mechanism/control experiments with original exact-IP GT protocol.

Outputs (under final/results):
- mechanism_bitflip.csv
- mechanism_exact_control.csv
- mechanism_component_ablation.csv
- mechanism_calibration_sensitivity.csv

Feature directory resolution:
1) $PMC_FEATURES_DIR (if set)
2) final/data/features
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import faiss
import numpy as np

GT_PROTOCOL = "original_exact_ip"
TOP_K = 100
SEED = 42

SETTINGS = {
    "mscoco_clip": {
        "dataset": "mscoco",
        "backbone": "clip-b32",
        "db_file": "mscoco_karpathy_val5k_clip_image_seed42.npy",
        "query_file": "mscoco_karpathy_val5k_clip_text_seed42.npy",
        "directions": [
            ("text2image", "query_to_db"),
            ("image2text", "db_to_query"),
        ],
    },
    "audiocaps_imagebind": {
        "dataset": "audiocaps",
        "backbone": "imagebind",
        "db_file": "audiocaps_test_imagebind_audio_single_seed42.npy",
        "query_file": "audiocaps_test_imagebind_text_seed42.npy",
        "directions": [
            ("text2audio", "query_to_db"),
            ("audio2text", "db_to_query"),
        ],
    },
}


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-8)).astype(np.float32)


def apply_gap(db: np.ndarray, queries: np.ndarray, gap: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    db_s = l2_normalize(db + alpha * gap)
    q_s = l2_normalize(queries - (1.0 - alpha) * gap)
    return db_s, q_s


def compute_topk_exact_ip(db: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    index = faiss.IndexFlatIP(db.shape[1])
    index.add(np.ascontiguousarray(db, dtype=np.float32))
    _, ids = index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
    return ids


def recall_to_gt(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    n = pred.shape[0]
    if n == 0 or k == 0:
        return 0.0
    hit = 0
    for i in range(n):
        hit += len(set(pred[i, :k].tolist()) & set(gt[i, :k].tolist()))
    return float(hit) / float(n * k)


def topk_jaccard(pred: np.ndarray, ref: np.ndarray, k: int) -> float:
    n = pred.shape[0]
    if n == 0 or k == 0:
        return 0.0
    s = 0.0
    for i in range(n):
        a = set(pred[i, :k].tolist())
        b = set(ref[i, :k].tolist())
        u = len(a | b)
        s += 0.0 if u == 0 else len(a & b) / u
    return s / n


def float_to_binary(vecs: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    bits = (vecs > centroid).astype(np.uint8)
    d = bits.shape[1]
    pad = (8 - d % 8) % 8
    if pad > 0:
        bits = np.hstack([bits, np.zeros((bits.shape[0], pad), dtype=np.uint8)])
    return np.packbits(bits, axis=1).copy()


def search_binary_flat(db: np.ndarray, queries: np.ndarray, centroid: np.ndarray, k: int) -> np.ndarray:
    db_bin = float_to_binary(db, centroid)
    q_bin = float_to_binary(queries, centroid)
    nbits = db_bin.shape[1] * 8
    index = faiss.IndexBinaryFlat(nbits)
    index.add(db_bin)
    _, ids = index.search(q_bin, k)
    return ids


def safe_unit_cos(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 1e-12 or bn <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / (an * bn), -1.0, 1.0))


def gap_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(safe_unit_cos(a, b))))


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if rows:
            w.writerows(rows)


def run_bitflip_rows(db_orig: np.ndarray, q_orig: np.ndarray, dataset: str, backbone: str, direction: str) -> List[Dict]:
    gap = q_orig.mean(axis=0) - db_orig.mean(axis=0)
    db_pmc, _ = apply_gap(db_orig, q_orig, gap, alpha=1.0)

    rows: List[Dict] = []
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    for vector_type, vecs, c0, c1 in [
        ("db", db_orig, db_orig.mean(axis=0), db_pmc.mean(axis=0)),
        ("query", q_orig, db_orig.mean(axis=0), db_pmc.mean(axis=0)),
    ]:
        s0 = vecs > c0
        s1 = vecs > c1
        flips = s0 != s1
        margins = np.abs(vecs - c0)

        rows.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "direction": direction,
                "vector_type": vector_type,
                "margin_bin": "global",
                "flip_rate": float(flips.mean()),
                "n_vectors": int(vecs.shape[0]),
                "n_dims": int(vecs.shape[1]),
                "gt_protocol": GT_PROTOCOL,
            }
        )

        q_edges = np.quantile(margins.reshape(-1), quantiles)
        bin_labels = ["q000_010", "q010_025", "q025_050", "q050_075", "q075_090"]
        prev = -np.inf
        for label, edge in zip(bin_labels, q_edges):
            mask = (margins > prev) & (margins <= edge)
            denom = int(mask.sum())
            rate = float(flips[mask].mean()) if denom > 0 else 0.0
            rows.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "direction": direction,
                    "vector_type": vector_type,
                    "margin_bin": label,
                    "flip_rate": rate,
                    "n_vectors": int(vecs.shape[0]),
                    "n_dims": int(vecs.shape[1]),
                    "gt_protocol": GT_PROTOCOL,
                }
            )
            prev = edge
    return rows


def run_exact_control_rows(db_orig: np.ndarray, q_orig: np.ndarray, dataset: str, backbone: str, direction: str) -> List[Dict]:
    gt_orig = compute_topk_exact_ip(db_orig, q_orig, TOP_K)
    rows: List[Dict] = []
    gap = q_orig.mean(axis=0) - db_orig.mean(axis=0)

    for alpha in [0.0, 0.25, 0.5, 1.0]:
        db_s, q_s = apply_gap(db_orig, q_orig, gap, alpha=alpha)
        pred = compute_topk_exact_ip(db_s, q_s, TOP_K)
        rows.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "direction": direction,
                "alpha": alpha,
                "r10_to_original_gt": recall_to_gt(pred, gt_orig, 10),
                "r100_to_original_gt": recall_to_gt(pred, gt_orig, 100),
                "jaccard10_vs_original_ranking": topk_jaccard(pred, gt_orig, 10),
                "jaccard100_vs_original_ranking": topk_jaccard(pred, gt_orig, 100),
                "gt_protocol": GT_PROTOCOL,
            }
        )
    return rows


def run_component_ablation_rows(
    db_orig: np.ndarray,
    q_orig: np.ndarray,
    dataset: str,
    backbone: str,
    direction: str,
    skip_heavy: bool,
) -> List[Dict]:
    gt_orig = compute_topk_exact_ip(db_orig, q_orig, TOP_K)
    gap = q_orig.mean(axis=0) - db_orig.mean(axis=0)

    variants = {
        "vanilla": (db_orig, q_orig),
        "query_only": (db_orig, l2_normalize(q_orig - gap)),
        "db_only": (l2_normalize(db_orig + gap), q_orig),
        "both": (l2_normalize(db_orig + 0.5 * gap), l2_normalize(q_orig - 0.5 * gap)),
    }

    rows: List[Dict] = []
    for mode, (db_v, q_v) in variants.items():
        centroid = db_v.mean(axis=0)
        pred_bin = search_binary_flat(db_v, q_v, centroid, TOP_K)
        rows.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "direction": direction,
                "index_type": "binary_flat",
                "mode": mode,
                "r10": recall_to_gt(pred_bin, gt_orig, 10),
                "r100": recall_to_gt(pred_bin, gt_orig, 100),
                "gt_protocol": GT_PROTOCOL,
            }
        )

        if not skip_heavy:
            try:
                d = db_v.shape[1]
                nlist = min(64, max(4, int(np.sqrt(db_v.shape[0]))))
                quantizer = faiss.IndexFlatIP(d)
                ivf = faiss.IndexIVFRaBitQ(quantizer, d, nlist)
                ivf.nprobe = min(16, nlist)
                ivf.train(np.ascontiguousarray(db_v, dtype=np.float32))
                ivf.add(np.ascontiguousarray(db_v, dtype=np.float32))
                _, pred_h = ivf.search(np.ascontiguousarray(q_v, dtype=np.float32), TOP_K)
                rows.append(
                    {
                        "dataset": dataset,
                        "backbone": backbone,
                        "direction": direction,
                        "index_type": "ivf_rabitq",
                        "mode": mode,
                        "r10": recall_to_gt(pred_h, gt_orig, 10),
                        "r100": recall_to_gt(pred_h, gt_orig, 100),
                        "gt_protocol": GT_PROTOCOL,
                    }
                )
            except Exception as exc:
                print(
                    f"[reproduce_mechanism_controls][warn] heavy ablation failed dataset={dataset} direction={direction} mode={mode}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

    return rows


def run_calibration_rows(db_orig: np.ndarray, q_orig: np.ndarray, dataset: str, backbone: str, direction: str) -> List[Dict]:
    gt_orig = compute_topk_exact_ip(db_orig, q_orig, TOP_K)
    full_gap = q_orig.mean(axis=0) - db_orig.mean(axis=0)

    vanilla_pred = compute_topk_exact_ip(db_orig, q_orig, TOP_K)
    r100_van = recall_to_gt(vanilla_pred, gt_orig, 100)

    db_full, q_full = apply_gap(db_orig, q_orig, full_gap, alpha=1.0)
    full_pred = compute_topk_exact_ip(db_full, q_full, TOP_K)
    r100_full = recall_to_gt(full_pred, gt_orig, 100)

    n_q = q_orig.shape[0]
    sizes = [25, 50, 100, 200, 400]
    usable_sizes = sorted(set(max(1, min(n_q, s)) for s in sizes))
    rows: List[Dict] = []

    for n_calib in usable_sizes:
        for sample_seed in [0, 1, 2, 3, 4]:
            rng = np.random.default_rng(sample_seed)
            pick = rng.choice(n_q, size=n_calib, replace=False)
            gap_est = q_orig[pick].mean(axis=0) - db_orig.mean(axis=0)
            db_s, q_s = apply_gap(db_orig, q_orig, gap_est, alpha=1.0)
            pred = compute_topk_exact_ip(db_s, q_s, TOP_K)
            r10 = recall_to_gt(pred, gt_orig, 10)
            r100 = recall_to_gt(pred, gt_orig, 100)
            rows.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "direction": direction,
                    "n_calib": int(n_calib),
                    "sample_seed": int(sample_seed),
                    "gap_cos_to_full": safe_unit_cos(gap_est, full_gap),
                    "gap_angle_deg": gap_angle_deg(gap_est, full_gap),
                    "r10": r10,
                    "r100": r100,
                    "delta_r100_vs_original_exact": r100 - r100_van,
                    "delta_r100_vs_fullgap": r100 - r100_full,
                    "gt_protocol": GT_PROTOCOL,
                }
            )
    return rows


def load_setting_views(features_dir: Path, setting_name: str) -> Iterable[Tuple[str, str, str, np.ndarray, np.ndarray]]:
    cfg = SETTINGS[setting_name]
    db = l2_normalize(np.load(features_dir / cfg["db_file"]).astype(np.float32))
    q = l2_normalize(np.load(features_dir / cfg["query_file"]).astype(np.float32))

    for direction, mode in cfg["directions"]:
        if mode == "query_to_db":
            yield cfg["dataset"], cfg["backbone"], direction, db, q
        else:
            yield cfg["dataset"], cfg["backbone"], direction, q, db


def resolve_features_dir(final_root: Path) -> Path:
    env_dir = os.environ.get("PMC_FEATURES_DIR", "").strip()
    if env_dir:
        p = Path(env_dir).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"PMC_FEATURES_DIR is not a directory: {p}")
        return p

    default_dir = final_root / "data" / "features"
    return default_dir


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _expect_close(name: str, actual: float, expected: float, tol: float = 5e-4) -> None:
    if abs(actual - expected) > tol:
        raise ValueError(f"{name}: expected {expected:.6f}, got {actual:.6f}")


TAB_MECHCONTROL_EXPECTED = {
    ("mscoco", "text2image"): {"flip_pct": 16.14, "j100": 0.5852, "bin_van": 0.5073, "bin_pmc": 0.5697, "n25_cos": 0.9860},
    ("mscoco", "image2text"): {"flip_pct": 16.00, "j100": 0.5422, "bin_van": 0.4579, "bin_pmc": 0.5675, "n25_cos": 0.9863},
    ("audiocaps", "text2audio"): {"flip_pct": 14.20, "j100": 0.8016, "bin_van": 0.6298, "bin_pmc": 0.6597, "n25_cos": 0.9653},
    ("audiocaps", "audio2text"): {"flip_pct": 17.75, "j100": 0.8036, "bin_van": 0.5143, "bin_pmc": 0.6442, "n25_cos": 0.9546},
}

TAB_MECH_EXTRA_CALIBRATION_EXPECTED_FULL = {
    25: {"text2image": (0.7294, 0.9860), "image2text": (0.6938, 0.9863)},
    50: {"text2image": (0.7321, 0.9933), "image2text": (0.6962, 0.9932)},
    100: {"text2image": (0.7296, 0.9965), "image2text": (0.6950, 0.9964)},
    200: {"text2image": (0.7312, 0.9984), "image2text": (0.6948, 0.9983)},
    400: {"text2image": (0.7296, 0.9992), "image2text": (0.6947, 0.9992)},
}

TAB_MECH_EXTRA_CALIBRATION_DISPLAYED = {25, 100, 400}

TAB_MECH_EXTRA_ABLATION_EXPECTED = {
    ("binary_flat", "text2image"): {"vanilla": 0.507, "query_only": 0.569, "db_only": 0.570, "both": 0.570},
    ("binary_flat", "image2text"): {"vanilla": 0.458, "query_only": 0.571, "db_only": 0.568, "both": 0.569},
    ("ivf_rabitq", "text2image"): {"vanilla": 0.578, "query_only": 0.541, "db_only": 0.637, "both": 0.599},
    ("ivf_rabitq", "image2text"): {"vanilla": 0.515, "query_only": 0.495, "db_only": 0.608, "both": 0.554},
}


def _build_calibration_means(calib_rows: List[Dict[str, str]]) -> Dict[Tuple[str, str, str, int], Dict[str, float]]:
    calib_mean: Dict[Tuple[str, str, str, int], Dict[str, float]] = {}
    calib_keys = {
        (r["dataset"], r["backbone"], r["direction"], int(r["n_calib"]))
        for r in calib_rows
    }
    for dataset, backbone, direction, n_calib in sorted(calib_keys):
        vals_r100 = [
            float(r["r100"])
            for r in calib_rows
            if r["dataset"] == dataset
            and r["backbone"] == backbone
            and r["direction"] == direction
            and int(r["n_calib"]) == n_calib
        ]
        vals_cos = [
            float(r["gap_cos_to_full"])
            for r in calib_rows
            if r["dataset"] == dataset
            and r["backbone"] == backbone
            and r["direction"] == direction
            and int(r["n_calib"]) == n_calib
        ]
        if not vals_r100 or not vals_cos:
            continue
        calib_mean[(dataset, backbone, direction, n_calib)] = {
            "r100": float(np.mean(vals_r100)),
            "cos": float(np.mean(vals_cos)),
        }
    return calib_mean


def _validate_tab_mechcontrol(
    bitflip: Dict[Tuple[str, str], float],
    j100: Dict[Tuple[str, str], float],
    ablation: Dict[Tuple[str, str, str, str], float],
    calib_mean: Dict[Tuple[str, str, str, int], Dict[str, float]],
) -> None:
    for key, exp in TAB_MECHCONTROL_EXPECTED.items():
        dataset, direction = key
        if key in bitflip:
            _expect_close(f"flip% {dataset}/{direction}", bitflip[key] * 100.0, exp["flip_pct"], tol=0.02)
        else:
            print(f"[reproduce_mechanism_controls] skip flip% check (missing row): {dataset}/{direction}")

        if key in j100:
            _expect_close(f"J@100 {dataset}/{direction}", j100[key], exp["j100"], tol=5e-4)
        else:
            print(f"[reproduce_mechanism_controls] skip J@100 check (missing row): {dataset}/{direction}")

        calib_key = (dataset, "clip-b32" if dataset == "mscoco" else "imagebind", direction, 25)
        if calib_key in calib_mean:
            _expect_close(
                f"n=25 cos {dataset}/{direction}",
                calib_mean[calib_key]["cos"],
                exp["n25_cos"],
                tol=5e-4,
            )
        else:
            print(f"[reproduce_mechanism_controls] skip n=25 cos check (missing rows): {dataset}/{direction}")

        ablation_key = (dataset, direction, "binary_flat", "vanilla")
        if ablation_key in ablation:
            _expect_close(
                f"BinaryFlat vanilla {dataset}/{direction}",
                ablation[ablation_key],
                exp["bin_van"],
                tol=5e-4,
            )
            db_only_key = (dataset, direction, "binary_flat", "db_only")
            if db_only_key not in ablation:
                raise ValueError(f"missing BinaryFlat db_only variant for {dataset}/{direction}")
            _expect_close(
                f"BinaryFlat DB-PMC {dataset}/{direction}",
                ablation[db_only_key],
                exp["bin_pmc"],
                tol=5e-4,
            )


def _validate_tab_mech_extra(
    ablation: Dict[Tuple[str, str, str, str], float],
    calib_mean: Dict[Tuple[str, str, str, int], Dict[str, float]],
) -> None:
    for n_calib in TAB_MECH_EXTRA_CALIBRATION_DISPLAYED:
        for direction in ("text2image", "image2text"):
            if ("mscoco", "clip-b32", direction, n_calib) not in calib_mean:
                raise ValueError(
                    f"missing displayed calibration row for paper table: n={n_calib}, direction={direction}"
                )

    for n_calib, per_dir in TAB_MECH_EXTRA_CALIBRATION_EXPECTED_FULL.items():
        for direction, (exp_r100, exp_cos) in per_dir.items():
            stats = calib_mean[("mscoco", "clip-b32", direction, n_calib)]
            _expect_close(f"calib r100 n={n_calib} {direction}", stats["r100"], exp_r100, tol=5e-4)
            _expect_close(f"calib cos n={n_calib} {direction}", stats["cos"], exp_cos, tol=5e-4)

    for (index_type, direction), exp_modes in TAB_MECH_EXTRA_ABLATION_EXPECTED.items():
        for mode, exp in exp_modes.items():
            actual = ablation[("mscoco", direction, index_type, mode)]
            _expect_close(f"ablation {index_type}/{direction}/{mode}", round(actual, 3), exp, tol=5e-4)


def validate_paper_tables(results_dir: Path) -> None:
    bitflip_rows = _load_rows(results_dir / "mechanism_bitflip.csv")
    exact_rows = _load_rows(results_dir / "mechanism_exact_control.csv")
    calib_rows = _load_rows(results_dir / "mechanism_calibration_sensitivity.csv")
    ablation_rows = _load_rows(results_dir / "mechanism_component_ablation.csv")

    bitflip = {
        (r["dataset"], r["direction"]): float(r["flip_rate"])
        for r in bitflip_rows
        if r["vector_type"] == "db" and r["margin_bin"] == "global"
    }
    j100 = {
        (r["dataset"], r["direction"]): float(r["jaccard100_vs_original_ranking"])
        for r in exact_rows
        if abs(float(r["alpha"]) - 1.0) < 1e-12
    }
    ablation = {
        (r["dataset"], r["direction"], r["index_type"], r["mode"]): float(r["r100"])
        for r in ablation_rows
    }
    calib_mean = _build_calibration_means(calib_rows)

    _validate_tab_mechcontrol(bitflip, j100, ablation, calib_mean)
    _validate_tab_mech_extra(ablation, calib_mean)

    print(f"[reproduce_mechanism_controls] validation passed using {results_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce PMC mechanism/control experiments")
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["mscoco_clip"],
        choices=sorted(SETTINGS.keys()),
        help="One or more preset settings",
    )
    parser.add_argument("--skip-heavy", action="store_true", help="Skip IVF-heavy ablations")
    parser.add_argument("--no-component-ablation", action="store_true", help="Skip component ablation CSV rows")
    parser.add_argument("--validate-paper", action="store_true", help="Validate paper table values from existing result CSVs")
    parser.add_argument("--results-dir", type=Path, default=None, help="Optional override for results directory")
    args = parser.parse_args()

    final_root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "src").is_dir() and (parent / "data").is_dir()
    )
    features_dir = resolve_features_dir(final_root)
    out_dir = args.results_dir.resolve() if args.results_dir else (final_root / "results")

    if args.validate_paper:
        validate_paper_tables(out_dir)
        return

    faiss.omp_set_num_threads(1)
    np.random.seed(SEED)

    bitflip_rows: List[Dict] = []
    exact_rows: List[Dict] = []
    component_rows: List[Dict] = []
    calibration_rows: List[Dict] = []

    for setting in args.settings:
        for dataset, backbone, direction, db_orig, q_orig in load_setting_views(features_dir, setting):
            bitflip_rows.extend(run_bitflip_rows(db_orig, q_orig, dataset, backbone, direction))
            exact_rows.extend(run_exact_control_rows(db_orig, q_orig, dataset, backbone, direction))
            if not args.no_component_ablation:
                component_rows.extend(
                    run_component_ablation_rows(db_orig, q_orig, dataset, backbone, direction, args.skip_heavy)
                )
            calibration_rows.extend(run_calibration_rows(db_orig, q_orig, dataset, backbone, direction))

    write_csv(
        out_dir / "mechanism_bitflip.csv",
        ["dataset", "backbone", "direction", "vector_type", "margin_bin", "flip_rate", "n_vectors", "n_dims", "gt_protocol"],
        bitflip_rows,
    )
    write_csv(
        out_dir / "mechanism_exact_control.csv",
        ["dataset", "backbone", "direction", "alpha", "r10_to_original_gt", "r100_to_original_gt", "jaccard10_vs_original_ranking", "jaccard100_vs_original_ranking", "gt_protocol"],
        exact_rows,
    )
    write_csv(
        out_dir / "mechanism_component_ablation.csv",
        ["dataset", "backbone", "direction", "index_type", "mode", "r10", "r100", "gt_protocol"],
        component_rows,
    )
    write_csv(
        out_dir / "mechanism_calibration_sensitivity.csv",
        ["dataset", "backbone", "direction", "n_calib", "sample_seed", "gap_cos_to_full", "gap_angle_deg", "r10", "r100", "delta_r100_vs_original_exact", "delta_r100_vs_fullgap", "gt_protocol"],
        calibration_rows,
    )

    print(f"[reproduce_mechanism_controls] features_dir={features_dir}")
    print(f"[reproduce_mechanism_controls] wrote {out_dir / 'mechanism_bitflip.csv'} ({len(bitflip_rows)} rows)")
    print(f"[reproduce_mechanism_controls] wrote {out_dir / 'mechanism_exact_control.csv'} ({len(exact_rows)} rows)")
    print(f"[reproduce_mechanism_controls] wrote {out_dir / 'mechanism_component_ablation.csv'} ({len(component_rows)} rows)")
    print(f"[reproduce_mechanism_controls] wrote {out_dir / 'mechanism_calibration_sensitivity.csv'} ({len(calibration_rows)} rows)")


if __name__ == "__main__":
    main()
