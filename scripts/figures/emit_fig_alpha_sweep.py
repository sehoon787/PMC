"""Emit results/figures/fig_alpha_sweep_rabitq.csv for reproduce_fig_analysis.py panel (a).

REQUIRES FAISS + feature .npy files. DO NOT run while a large FAISS experiment
owns the machine's RAM. Run it later, once the machine is free.

What it produces (one row per (dataset, direction, alpha)):
    dataset, direction, alpha, r100
for dataset in {mscoco, clotho}, direction in the two cross-modal directions,
and alpha in {0.0, 0.25, 0.5, 0.75, 1.0}. r100 is the IVF-RaBitQ recall vs the
original exact-IP GT, mirroring scripts/research/R13_signbit_original_gt.py's
RaBitQ path (run_rabitq) under apply_pmc(alpha) for each alpha.

CROSS-CHECK target (the values currently HARDCODED in
paper/figures/fig3_analysis.py:plot_alpha_sweep, alphas [0,.25,.5,.75,1]):
    mscoco/text2image (CLIP t->i): [0.5702, 0.5529, 0.5808, 0.6092, 0.6237]
    mscoco/image2text (CLIP i->t): [0.4940, 0.5143, 0.5478, 0.5846, 0.6031]
    clotho/text2audio (Clotho t->a): [0.6558, 0.6898, 0.7227, 0.7494, 0.7550]
    clotho/audio2text (Clotho a->t): [0.6308, 0.6536, 0.6895, 0.7168, 0.7364]
  (The figure also plots IB/MSCOCO; that pair is not part of panel (a)'s
   {mscoco, clotho} reproduce scope, so it is omitted here.)
After running, verify the emitted r100 matches these arrays cell-exact (2 dp);
if a cell disagrees, do NOT overwrite the figure -- flag the mismatch.

==========================  UNCERTAINTY / TODO  =============================
1) NPROBE / NLIST: R13 uses NLIST=64, NPROBE=16 for run_rabitq. Confirm the
   alpha-sweep figure used the SAME operating point (the figure caption / the
   commit that produced fig3 should be checked). If the figure used a different
   nprobe, set NPROBE accordingly below; otherwise the cross-check will fail.

2) CLOTHO MULTI-CAPTION PROTOCOL (the load-bearing unknown): the Clotho text
   features file is `clotho_eval_imagebind_text_5cap_seed42.npy` (5 captions per
   audio). The db<->query cardinalities and the GT/recall definition for the
   5-caption setting are NOT obvious and differ from the 1:1 MSCOCO setting.
   R13/R14 handle this somewhere -- READ scripts/research/R14_clotho_signbit.py
   to see exactly how Clotho query/db arrays and GT are built (how the 5 caps
   map to audio rows, whether captions are averaged or kept per-row, and how
   compute_gt / compute_recall are applied). Wire CLOTHO_QUERY_FILE / db file /
   GT handling to match R14 BEFORE trusting the Clotho rows. The MSCOCO rows
   below follow R13 exactly and should be correct; the Clotho rows are marked
   TODO and will be wrong until the 5-caption protocol is wired in.
============================================================================
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import faiss  # noqa: F401  (import guarded; only used when run)
import numpy as np

ALPHA_LADDER = [0.0, 0.25, 0.5, 0.75, 1.0]
SEED = 42
NLIST = 64
NPROBE = 16  # TODO(1): confirm matches the figure's operating point.
K_VALUES = [1, 10, 100]

# Cross-check target arrays from paper/figures/fig3_analysis.py (alphas above).
FIG3_TARGETS = {
    ("mscoco", "text2image"): [0.5702, 0.5529, 0.5808, 0.6092, 0.6237],
    ("mscoco", "image2text"): [0.4940, 0.5143, 0.5478, 0.5846, 0.6031],
    ("clotho", "text2audio"): [0.6558, 0.6898, 0.7227, 0.7494, 0.7550],
    ("clotho", "audio2text"): [0.6308, 0.6536, 0.6895, 0.7168, 0.7364],
}

# Feature-file config. MSCOCO follows R13 exactly (1 text per image row).
# Clotho is a TODO: the text file holds 5 captions per audio.
SETTINGS = {
    "mscoco": {
        "backbone": "clip-b32",
        "db_file": "mscoco_karpathy_val5k_clip_image_seed42.npy",
        "query_file": "mscoco_karpathy_val5k_clip_text_seed42.npy",
        "directions": [
            ("text2image", "query_to_db"),
            ("image2text", "db_to_query"),
        ],
        "multi_caption": False,
    },
    "clotho": {
        "backbone": "imagebind",
        "db_file": "clotho_eval_imagebind_audio_seed42.npy",
        # TODO(2): 5-caption text; mapping must match R14_clotho_signbit.py.
        "query_file": "clotho_eval_imagebind_text_5cap_seed42.npy",
        "directions": [
            ("text2audio", "query_to_db"),
            ("audio2text", "db_to_query"),
        ],
        "multi_caption": True,
    },
}

OUTPUT_FIELDNAMES = ["dataset", "direction", "alpha", "r100"]


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "results").is_dir() and (parent / "data").is_dir():
            return parent
    raise RuntimeError("Could not find project root containing results/ and data/")


PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "results"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_CSV = RESULTS_DIR / "figures" / "fig_alpha_sweep_rabitq.csv"


# --- Core (mirrors R13_signbit_original_gt.py) -------------------------------

def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / np.maximum(norms, 1e-8)).astype(np.float32)


def apply_pmc(db: np.ndarray, queries: np.ndarray,
              alpha: float) -> tuple[np.ndarray, np.ndarray]:
    gap = queries.mean(axis=0) - db.mean(axis=0)
    db_s = l2_normalize(db + alpha * gap)
    q_s = l2_normalize(queries - (1.0 - alpha) * gap)
    return db_s, q_s


def compute_gt(db: np.ndarray, queries: np.ndarray, k: int = 100) -> np.ndarray:
    index = faiss.IndexFlatIP(db.shape[1])
    index.add(db.astype(np.float32))
    _, ids = index.search(queries.astype(np.float32), k)
    return ids


def compute_recall(ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    n = ids.shape[0]
    return sum(
        len(set(ids[i, :k].tolist()) & set(gt_ids[i, :k].tolist()))
        for i in range(n)
    ) / (n * k)


def run_rabitq(db: np.ndarray, queries: np.ndarray, gt_ids: np.ndarray) -> float:
    d = db.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFRaBitQ(quantizer, d, NLIST)
    index.nprobe = NPROBE
    index.train(db)
    index.add(db)
    _, ids = index.search(queries, max(K_VALUES))
    return compute_recall(ids, gt_ids, 100)


def load_views(dataset: str):
    cfg = SETTINGS[dataset]
    db = l2_normalize(np.load(FEATURES_DIR / cfg["db_file"]).astype(np.float32))
    q = l2_normalize(np.load(FEATURES_DIR / cfg["query_file"]).astype(np.float32))
    if cfg["multi_caption"]:
        # TODO(2): the 5-caption Clotho query array does not line up 1:1 with
        # the audio db rows. Reshape / index per R14_clotho_signbit.py before
        # using these. As-is, db/q row counts will mismatch and recall will be
        # meaningless. DO NOT trust Clotho output until this is wired correctly.
        pass
    for direction, mode in cfg["directions"]:
        if mode == "query_to_db":
            yield direction, db, q
        else:
            yield direction, q, db


def build_records() -> list[dict[str, str]]:
    faiss.omp_set_num_threads(1)
    np.random.seed(SEED)
    records: list[dict[str, str]] = []
    for dataset in SETTINGS:
        for direction, db_view, q_view in load_views(dataset):
            gt = compute_gt(db_view, q_view, k=max(K_VALUES))
            for alpha in ALPHA_LADDER:
                db_s, q_s = apply_pmc(db_view, q_view, alpha)
                r100 = run_rabitq(db_s, q_s, gt)
                records.append({
                    "dataset": dataset,
                    "direction": direction,
                    "alpha": f"{alpha:g}",
                    "r100": f"{r100:.4f}",
                })
                _cross_check(dataset, direction, alpha, r100)
    return records


def _cross_check(dataset: str, direction: str, alpha: float, r100: float) -> None:
    targets = FIG3_TARGETS.get((dataset, direction))
    if targets is None:
        return
    idx = ALPHA_LADDER.index(alpha)
    expected = targets[idx]
    if abs(round(r100, 2) - round(expected, 2)) > 1e-9:
        print(f"[emit_fig_alpha_sweep][CROSS-CHECK MISMATCH] {dataset}/{direction} "
              f"alpha={alpha:g}: got {r100:.4f} (.{round(r100,2):.2f}) vs "
              f"fig3 {expected:.4f}")


def write_output(records: list[dict[str, str]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit fig_alpha_sweep_rabitq.csv (REQUIRES FAISS)."
    )
    parser.parse_args()
    records = build_records()
    write_output(records)
    print(f"[emit_fig_alpha_sweep] wrote {OUTPUT_CSV} ({len(records)} rows)")
    print("[emit_fig_alpha_sweep] cross-check any MISMATCH lines above against "
          "paper/figures/fig3_analysis.py before trusting the output.")


if __name__ == "__main__":
    main()
