"""Diagnose PMC vs Vanilla candidate-pool coverage for cross-modal RaBitQ.

Hypothesis under test
---------------------
PMC (Per-Modality-Centroid correction, alpha=1) sharpens the HEAD of the
RaBitQ binary ranking but may produce a broader top-K' candidate pool that
COVERS fewer of the true top-100 neighbours.  When the top-K' pool is exactly
re-scored, Vanilla can therefore recover more (this is the only cell where
Vanilla overtakes PMC in Table 2: LAION-400M reverse, image->text).

This script measures, for every Table-2 small-dataset config (6 configs x 2
directions = 12 measurements), the pool-coverage of the true top-100 inside
the K'=400 candidate pool of each method, plus modality-concentration and
probe-budget diagnostics.  Two LAION-400M proxy rows are appended from the
existing rerank-ladder CSVs (the 407M index is NOT rebuilt).

Outputs
-------
- results/diagnostics/pool_coverage_diagnostic_seed42.csv
- a markdown summary table on stdout

All ground truth and recall are computed on the ORIGINAL (unshifted) L2-
normalized embeddings, mirroring src/experiments/paired_recall_eval.py.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.index_wrappers import build_vanilla_rabitq
from src.core.metrics import compute_ground_truth, recall_at_k
from src.core.pmc import compute_gap, shift_db_vectors, shift_query_vectors
from src.utils import l2_normalize

SEED = 42
ALPHA = 1.0
KPRIME = 400  # Table-2 rerank operating point
GT_TOPK = 100  # true top-100 used for ground truth and coverage
COV_DENOM = 100  # coverage normalizer (true top-100)


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "results").is_dir() and (parent / "data").is_dir():
            return parent
    raise RuntimeError("Could not find project root with results/ and data/")


PROJECT_ROOT = find_project_root()
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "diagnostics" / "pool_coverage_diagnostic_seed42.csv"

LAION_FWD_CSV = RESULTS_DIR / "sources" / "pmc_laion400m_rerank_nlist80k_k400_seed42.csv"
LAION_REV_CSV = RESULTS_DIR / "sources" / "pmc_laion400m_reverse_rerank_nlist80k_k400_seed42.csv"


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """One (dataset, encoder) cell of Table 2 with its two modality files."""

    dataset: str          # display label
    enc: str              # display encoder label
    prefix: str           # file prefix
    backbone: str         # file backbone token
    db_modality: str      # the "image"/"audio" side modality name
    text_modality: str    # always "text"


CONFIGS: list[Config] = [
    Config("MSCOCO", "CLIP", "mscoco_karpathy_val5k", "clip", "image", "text"),
    Config("MSCOCO", "CL-L", "mscoco_karpathy_val5k", "clip-l", "image", "text"),
    Config("MSCOCO", "IB", "mscoco_karpathy_val5k", "imagebind", "image", "text"),
    Config("Flickr30K", "CL-L", "flickr30k_full", "clip-l", "image", "text"),
    Config("Clotho", "IB", "clotho_all", "imagebind", "audio", "text"),
    Config("AudioCaps", "IB", "audiocaps_test", "imagebind", "audio", "text"),
]

# Table-2 rerank verdict per (dataset, enc, direction): does PMC win under
# exact reranking?  All 6 small configs win both directions; only LAION
# reverse loses.  Recorded here for reference (not recomputed).
RERANK_VERDICT_SMALL = "PMC wins"


def load_modality(prefix: str, backbone: str, modality: str) -> np.ndarray | None:
    """Load + L2-normalize one modality file, or return None if missing."""
    path = FEATURES_DIR / f"{prefix}_{backbone}_{modality}_seed{SEED}.npy"
    if not path.exists():
        print(f"[WARN] Missing embedding file: {path}", flush=True)
        return None
    arr = np.load(str(path)).astype(np.float32)
    return l2_normalize(arr)


# ---------------------------------------------------------------------------
# Concentration metrics (on an L2-normalized modality matrix)
# ---------------------------------------------------------------------------

def concentration_metrics(emb: np.ndarray) -> tuple[float, float, float]:
    """Return (mean_norm, sigma2, eff_dim) for an L2-normalized matrix.

    mean_norm : ||mean(X)||                higher => tighter / anisotropic
    sigma2    : mean_i ||x_i - mean(X)||^2  lower  => tighter
    eff_dim   : (sum lambda)^2 / sum lambda^2   participation ratio of the
                covariance eigenvalues; lower => energy in fewer directions
    """
    mean_vec = emb.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_vec))

    centered = emb - mean_vec[np.newaxis, :]
    sigma2 = float(np.mean(np.sum(centered * centered, axis=1)))

    # Covariance eigenvalues via SVD of the centered matrix (numerically stable).
    # cov = (1/n) C^T C ; eigenvalues of cov = (singular values^2) / n.
    n = centered.shape[0]
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eig = (singular.astype(np.float64) ** 2) / n
    sum_eig = float(eig.sum())
    sum_eig_sq = float(np.sum(eig ** 2))
    eff_dim = (sum_eig ** 2) / sum_eig_sq if sum_eig_sq > 0 else float("nan")
    return mean_norm, sigma2, eff_dim


# ---------------------------------------------------------------------------
# Index params (Table-2 matched-budget rule)
# ---------------------------------------------------------------------------

def table2_params(n_db: int) -> tuple[int, int]:
    """nlist = ceil(sqrt(n_db)); nprobe = round-half-up(nlist / 4), min 1."""
    nlist = math.ceil(math.sqrt(n_db))
    nprobe = max(1, int(math.floor(nlist / 4 + 0.5)))
    return nlist, nprobe


# ---------------------------------------------------------------------------
# Coverage diagnostic
# ---------------------------------------------------------------------------

def pool_coverage(retrieved_kprime: np.ndarray, gt_top100: np.ndarray) -> float:
    """Mean over queries of |pool_K' intersect gt_top100| / 100."""
    q = len(retrieved_kprime)
    total = 0.0
    for i in range(q):
        pool = {int(x) for x in retrieved_kprime[i] if x >= 0}
        gt = {int(x) for x in gt_top100[i] if x >= 0}
        total += len(pool & gt) / COV_DENOM
    return total / q if q > 0 else 0.0


def mean_jaccard(pool_a: np.ndarray, pool_b: np.ndarray) -> float:
    """Mean over queries of |A intersect B| / |A union B| on the K' pools."""
    q = len(pool_a)
    total = 0.0
    for i in range(q):
        a = {int(x) for x in pool_a[i] if x >= 0}
        b = {int(x) for x in pool_b[i] if x >= 0}
        union = a | b
        total += (len(a & b) / len(union)) if union else 0.0
    return total / q if q > 0 else 0.0


# ---------------------------------------------------------------------------
# One (config, direction) measurement
# ---------------------------------------------------------------------------

def measure_direction(
    *,
    dataset: str,
    enc: str,
    direction: str,
    db_emb: np.ndarray,
    query_emb: np.ndarray,
    shifted_db_modality: str,
) -> dict[str, object]:
    """Run GT -> build (van/pmc) -> search K' -> coverage/jaccard/recall."""
    n_db, d = db_emb.shape
    nlist, nprobe = table2_params(n_db)
    probe_budget = nprobe / nlist

    print(f"\n{'=' * 72}")
    print(f"{dataset}/{enc}  dir={direction}  n_db={n_db} d={d} "
          f"nlist={nlist} nprobe={nprobe}")
    print("=" * 72, flush=True)

    # Ground truth on ORIGINAL vectors (top-100), and the K'=400 GT denom set
    # we only need the top-100 for coverage.
    gt100 = compute_ground_truth(query_emb, db_emb, top_k=GT_TOPK)

    # Concentration of the SHIFTED-DB modality (the modality whose vectors are
    # pushed into the index, i.e. db_emb for this direction).
    mean_norm, sigma2, eff_dim = concentration_metrics(db_emb)

    gap = compute_gap(db_emb, query_emb)

    # Vanilla index + K'=400 pool.
    van_idx = build_vanilla_rabitq(db_emb, nlist=nlist, seed=SEED)
    _, van_pool = van_idx.search(query_emb, top_k=KPRIME, nprobe=nprobe)
    _, van_top100 = van_idx.search(query_emb, top_k=GT_TOPK, nprobe=nprobe)

    # PMC index (shifted DB) + K'=400 pool with shifted queries.
    db_shifted = shift_db_vectors(db_emb, gap, alpha=ALPHA)
    pmc_idx = build_vanilla_rabitq(db_shifted, nlist=nlist, seed=SEED)
    q_shifted = shift_query_vectors(query_emb, gap, alpha=ALPHA)
    _, pmc_pool = pmc_idx.search(q_shifted, top_k=KPRIME, nprobe=nprobe)
    _, pmc_top100 = pmc_idx.search(q_shifted, top_k=GT_TOPK, nprobe=nprobe)

    cov_van = pool_coverage(van_pool, gt100)
    cov_pmc = pool_coverage(pmc_pool, gt100)
    jacc = mean_jaccard(van_pool, pmc_pool)

    r100_van = recall_at_k(van_top100, gt100, k=GT_TOPK)
    r100_pmc = recall_at_k(pmc_top100, gt100, k=GT_TOPK)

    print(f"  cov_van={cov_van:.4f}  cov_pmc={cov_pmc:.4f}  "
          f"cov_gap={cov_pmc - cov_van:+.4f}  jaccard={jacc:.4f}", flush=True)
    print(f"  r100_van={r100_van:.4f}  r100_pmc={r100_pmc:.4f}", flush=True)

    return {
        "Dataset": dataset,
        "Enc": enc,
        "Dir": direction,
        "n_db": n_db,
        "nlist": nlist,
        "nprobe": nprobe,
        "probe_budget": round(probe_budget, 6),
        "shifted_db_modality": shifted_db_modality,
        "mean_norm": round(mean_norm, 6),
        "sigma2": round(sigma2, 6),
        "eff_dim": round(eff_dim, 4),
        "cov_van": round(cov_van, 6),
        "cov_pmc": round(cov_pmc, 6),
        "cov_gap": round(cov_pmc - cov_van, 6),
        "jaccard": round(jacc, 6),
        "r100_van": round(r100_van, 6),
        "r100_pmc": round(r100_pmc, 6),
        "rerank_verdict": RERANK_VERDICT_SMALL,
    }


# ---------------------------------------------------------------------------
# LAION proxy rows from existing rerank CSVs
# ---------------------------------------------------------------------------

def _read_rerank_r100(path: Path, method: str, rerank_k: int) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == method and int(row["rerank_k"]) == rerank_k:
                return float(row["r100"])
    raise ValueError(f"No row method={method} rerank_k={rerank_k} in {path}")


def laion_proxy_row(
    *, direction: str, csv_path: Path, shifted_db_modality: str, verdict: str,
) -> dict[str, object]:
    """Build a LAION proxy row.  cov_* are the reranked-at-K'=400 R@100 values
    from the CSV (a tight proxy for true-top-100 pool coverage)."""
    cov_van = _read_rerank_r100(csv_path, "vanilla_rabitq", KPRIME)
    cov_pmc = _read_rerank_r100(csv_path, "pmc_1.00", KPRIME)
    r100_van = _read_rerank_r100(csv_path, "vanilla_rabitq", 0)
    r100_pmc = _read_rerank_r100(csv_path, "pmc_1.00", 0)
    probe_budget = 256 / 80000
    return {
        "Dataset": "LAION-400M",
        "Enc": "CLIP",
        "Dir": direction,
        "n_db": 407314954,
        "nlist": 80000,
        "nprobe": 256,
        "probe_budget": round(probe_budget, 6),
        "shifted_db_modality": shifted_db_modality,
        "mean_norm": "n/m (407M)",
        "sigma2": "n/m (407M)",
        "eff_dim": "n/m (407M)",
        "cov_van": round(cov_van, 6),
        "cov_pmc": round(cov_pmc, 6),
        "cov_gap": round(cov_pmc - cov_van, 6),
        "jaccard": "n/m (407M)",
        "r100_van": round(r100_van, 6),
        "r100_pmc": round(r100_pmc, 6),
        "rerank_verdict": verdict + " (proxy from rerank CSV)",
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "Dataset", "Enc", "Dir", "n_db", "nlist", "nprobe", "probe_budget",
    "shifted_db_modality", "mean_norm", "sigma2", "eff_dim",
    "cov_van", "cov_pmc", "cov_gap", "jaccard",
    "r100_van", "r100_pmc", "rerank_verdict",
]


def write_csv(rows: list[dict[str, object]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_markdown(rows: list[dict[str, object]]) -> None:
    print("\n" + "#" * 72)
    print("# POOL-COVERAGE DIAGNOSTIC (seed=42, K'=400, alpha=1)")
    print("#" * 72)
    print("| " + " | ".join(FIELDNAMES) + " |")
    print("|" + "|".join(["---"] * len(FIELDNAMES)) + "|")
    for row in rows:
        print("| " + " | ".join(str(row.get(h, "")) for h in FIELDNAMES) + " |")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import faiss

    faiss.omp_set_num_threads(1)

    forward_rows: list[dict[str, object]] = []
    reverse_rows: list[dict[str, object]] = []

    for cfg in CONFIGS:
        db_side = load_modality(cfg.prefix, cfg.backbone, cfg.db_modality)
        text = load_modality(cfg.prefix, cfg.backbone, cfg.text_modality)
        if db_side is None or text is None:
            print(f"[SKIP] {cfg.dataset}/{cfg.enc}: missing embedding file.",
                  flush=True)
            continue

        # forward q->db: query=TEXT, db=IMAGE/AUDIO (shifted-DB modality = db_side)
        forward_rows.append(measure_direction(
            dataset=cfg.dataset, enc=cfg.enc, direction="forward",
            db_emb=db_side, query_emb=text,
            shifted_db_modality=cfg.db_modality,
        ))

        # reverse db->q: query=IMAGE/AUDIO, db=TEXT (shifted-DB modality = text)
        reverse_rows.append(measure_direction(
            dataset=cfg.dataset, enc=cfg.enc, direction="reverse",
            db_emb=text, query_emb=db_side,
            shifted_db_modality=cfg.text_modality,
        ))

    laion_rows = [
        laion_proxy_row(
            direction="forward", csv_path=LAION_FWD_CSV,
            shifted_db_modality="image", verdict="PMC wins",
        ),
        laion_proxy_row(
            direction="reverse", csv_path=LAION_REV_CSV,
            shifted_db_modality="text", verdict="Vanilla wins",
        ),
    ]

    all_rows = forward_rows + reverse_rows + laion_rows
    write_csv(all_rows)
    print_markdown(all_rows)
    print(f"\nWrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
