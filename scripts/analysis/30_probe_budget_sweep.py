"""Probe-budget sweep: does PMC's rerank advantage flip at LAION-scale budget?

Decisive single-variable test
-----------------------------
The pool-coverage diagnostic (29_diagnose_pool_coverage.py) showed that the
ONLY Table-2 cell where Vanilla overtakes PMC under exact reranking is
LAION-400M reverse, and it is the ONLY row with a negative cov_gap.  The
separating variable between that cell and all 12 positive small-scale cells is
the IVF *probe budget*: LAION scans only nprobe/nlist = 256/80000 = 0.0032 of
its clusters, whereas every small config scans ~0.25.  Modality-concentration
metrics (mean_norm/eff_dim/sigma2) do NOT track the verdict, so the
"shifted-text-DB is tighter" story is unsupported.

This script isolates probe budget as the single causal variable.  We fix one
small dataset (MSCOCO/CLIP) and one index granularity (nlist=320, chosen so
nprobe=1 reaches budget 0.0031 ~= LAION's 0.0032), then sweep nprobe so the
budget ranges from 0.0031 up to 0.25.  nlist and the data are held constant;
only the fraction of clusters scanned changes.

If PMC's positive cov_gap collapses to negative as the budget approaches the
LAION level, probe budget is directly confirmed as the cause and the
modality-concentration explanation can be ruled out.  If cov_gap stays
positive even at budget 0.003, probe budget alone is NOT the cause.

cov (true-top-100 coverage inside the K'=400 pool) is the ceiling of rerank
R@100: exact rescoring cannot recover a neighbour the pool never contained, so
cov_gap < 0 implies the rerank verdict flips to Vanilla -- the same proxy used
for the LAION rows in the diagnostic.

Outputs
-------
- results/probe_budget_sweep_mscoco_seed42.csv
- a markdown summary table on stdout
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.core.index_wrappers import build_vanilla_rabitq
from src.core.metrics import compute_ground_truth
from src.core.pmc import compute_gap, shift_db_vectors, shift_query_vectors
from src.utils import l2_normalize

SEED = 42
ALPHA = 1.0
KPRIME = 400          # Table-2 rerank operating point
GT_TOPK = 100         # true top-100 used for ground truth and coverage
COV_DENOM = 100       # coverage normalizer (true top-100)

# Fixed index granularity so nprobe=1 reaches LAION-level budget (1/320=0.0031).
FIXED_NLIST = 320
# nprobe ladder -> budgets {0.0031, 0.0063, 0.0094, 0.0156, 0.025, 0.05, 0.1,
# 0.2, 0.25}.  Spans the LAION budget (0.0032) up to the small-config budget.
NPROBE_LADDER = [1, 2, 3, 5, 8, 16, 32, 64, 80]

DATASET = "MSCOCO"
ENC = "CLIP"
PREFIX = "mscoco_karpathy_val5k"
BACKBONE = "clip"


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "results").is_dir() and (parent / "data").is_dir():
            return parent
    raise RuntimeError("Could not find project root with results/ and data/")


PROJECT_ROOT = find_project_root()
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "probe_budget_sweep_mscoco_seed42.csv"


def load_modality(modality: str) -> np.ndarray:
    path = FEATURES_DIR / f"{PREFIX}_{BACKBONE}_{modality}_seed{SEED}.npy"
    arr = np.load(str(path)).astype(np.float32)
    return l2_normalize(arr)


def pool_coverage(retrieved_kprime: np.ndarray, gt_top100: np.ndarray) -> float:
    """Mean over queries of |pool_K' intersect gt_top100| / 100."""
    total = 0.0
    for i in range(len(retrieved_kprime)):
        pool = {int(x) for x in retrieved_kprime[i] if x >= 0}
        gt = {int(x) for x in gt_top100[i] if x >= 0}
        total += len(pool & gt) / COV_DENOM
    return total / len(retrieved_kprime) if len(retrieved_kprime) else 0.0


def sweep_direction(
    *, direction: str, db_emb: np.ndarray, query_emb: np.ndarray,
    shifted_db_modality: str,
) -> list[dict[str, object]]:
    """Build van/pmc indexes once at FIXED_NLIST, then sweep nprobe."""
    n_db = db_emb.shape[0]
    print(f"\n{'=' * 72}")
    print(f"{DATASET}/{ENC} dir={direction} n_db={n_db} nlist={FIXED_NLIST} "
          f"shifted-DB={shifted_db_modality}")
    print("=" * 72, flush=True)

    gt100 = compute_ground_truth(query_emb, db_emb, top_k=GT_TOPK)
    gap = compute_gap(db_emb, query_emb)

    van_idx = build_vanilla_rabitq(db_emb, nlist=FIXED_NLIST, seed=SEED)
    db_shifted = shift_db_vectors(db_emb, gap, alpha=ALPHA)
    pmc_idx = build_vanilla_rabitq(db_shifted, nlist=FIXED_NLIST, seed=SEED)
    q_shifted = shift_query_vectors(query_emb, gap, alpha=ALPHA)

    rows: list[dict[str, object]] = []
    for nprobe in NPROBE_LADDER:
        budget = nprobe / FIXED_NLIST
        _, van_pool = van_idx.search(query_emb, top_k=KPRIME, nprobe=nprobe)
        _, pmc_pool = pmc_idx.search(q_shifted, top_k=KPRIME, nprobe=nprobe)
        cov_van = pool_coverage(van_pool, gt100)
        cov_pmc = pool_coverage(pmc_pool, gt100)
        cov_gap = cov_pmc - cov_van
        verdict = "PMC" if cov_gap > 0 else "Vanilla"
        print(f"  nprobe={nprobe:>3} budget={budget:.4f}  "
              f"cov_van={cov_van:.4f} cov_pmc={cov_pmc:.4f} "
              f"cov_gap={cov_gap:+.4f}  -> {verdict}", flush=True)
        rows.append({
            "Dataset": DATASET,
            "Enc": ENC,
            "Dir": direction,
            "shifted_db_modality": shifted_db_modality,
            "n_db": n_db,
            "nlist": FIXED_NLIST,
            "nprobe": nprobe,
            "probe_budget": round(budget, 6),
            "cov_van": round(cov_van, 6),
            "cov_pmc": round(cov_pmc, 6),
            "cov_gap": round(cov_gap, 6),
            "rerank_verdict": verdict,
        })
    return rows


FIELDNAMES = [
    "Dataset", "Enc", "Dir", "shifted_db_modality", "n_db", "nlist", "nprobe",
    "probe_budget", "cov_van", "cov_pmc", "cov_gap", "rerank_verdict",
]


def write_csv(rows: list[dict[str, object]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_markdown(rows: list[dict[str, object]]) -> None:
    print("\n" + "#" * 72)
    print(f"# PROBE-BUDGET SWEEP (seed={SEED}, K'={KPRIME}, alpha={ALPHA}, "
          f"nlist={FIXED_NLIST})")
    print("#" * 72)
    print("| " + " | ".join(FIELDNAMES) + " |")
    print("|" + "|".join(["---"] * len(FIELDNAMES)) + "|")
    for row in rows:
        print("| " + " | ".join(str(row.get(h, "")) for h in FIELDNAMES) + " |")


def main() -> None:
    import faiss

    faiss.omp_set_num_threads(1)

    image = load_modality("image")
    text = load_modality("text")

    # reverse db->q (the cell of interest): query=IMAGE, db=TEXT (shift text DB)
    reverse_rows = sweep_direction(
        direction="reverse", db_emb=text, query_emb=image,
        shifted_db_modality="text",
    )
    # forward q->db control: query=TEXT, db=IMAGE (shift image DB)
    forward_rows = sweep_direction(
        direction="forward", db_emb=image, query_emb=text,
        shifted_db_modality="image",
    )

    all_rows = reverse_rows + forward_rows
    write_csv(all_rows)
    print_markdown(all_rows)
    print(f"\nWrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
