"""Reproduce the "with reranking" column group of tab:mainresults: §4.3 Robustness under Exact Reranking.

This script runs NO FAISS and no heavy experiment. It reads the recall-tuned
rerank artifact (results/sources/rerank_subset_sqrtN_kfine_seed42.csv) and reproduces the
operating-point numbers that back the "with reranking" half of tab:mainresults:

  R@100 after exact reranking of the top-K' binary candidates, with a
  **recall-tuned first stage** (n_probe ~ n_list -- the largest swept n_probe,
  ~90% of cells probed on these 5K--31K-vector subset corpora), K'=400.

Under this budget PMC leads in every cell at every depth K' in {100,200,400,500};
this script verifies (i) the printed K'=400 cell values match the manuscript,
(ii) PMC >= max(Vanilla, MeanShift) at every K' in the ladder (uniform
dominance), and (iii) the deployable-recall (R@100 >= 0.90) min-K' diagnostic
that backs the §4.3 "PMC reaches deployable recall at a smaller rerank budget"
sentence.

n_probe selection (recall-tuned, per direction):

  Choose the LARGEST swept n_probe available in the data for that
  (backbone, dataset, direction) group (capped <= n_list). On these subset
  corpora the largest swept value is n_probe ~ n_list (near-exhaustive), which
  is the recall-tuned operating point the cascade table reports. When n_list
  differs across directions (Clotho: text->audio n_list=32, audio->text
  n_list=72) the two directions pick different n_probe (32 / 64), shown as
  "32/64" in the table.

LAION-400M (407M vectors) is NOT in this subset CSV; its forward np256 rerank
cell (.198/.149/.277) and the omitted reverse cell are reproduced separately
(see reproduce_ablation_rerank.py and PAPER_RESULT_PROVENANCE.md).

A "# REPRODUCE-MISMATCH" line on any run means a printed rerank cell, the
uniform-dominance property, or an expectation drifted from its committed value.

Outputs:
  - stdout: human-readable operating-point table + dominance/min-K' diagnostics
            + LaTeX rows (tab:mainresults rerank format) + any # REPRODUCE-MISMATCH lines
  - results/tables/rerank_deployable_reproduced.csv
"""

from __future__ import annotations

import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# --- Configuration ----------------------------------------------------------

DEPLOYABLE_THRESHOLD = 0.90          # R@100 target for "deployable" recall
RERANK_K_LADDER = [0, 100, 200, 400, 500]
CELL_K = 400                         # printed cascade cell depth (tab:mainresults)

# (backbone, dataset, display_enc, q_dir, db_dir)
# n_probe is the largest swept value per direction (recall-tuned ~ n_list).
ROWS: list[tuple[str, str, str, str, str]] = [
    ("clip",      "mscoco",    "CLIP", "text->image", "image->text"),
    ("clip-l",    "mscoco",    "CL-L", "text->image", "image->text"),
    ("imagebind", "mscoco",    "IB",   "text->image", "image->text"),
    ("clip-l",    "flickr30k", "CL-L", "text->image", "image->text"),
    ("imagebind", "clotho",    "IB",   "text->audio", "audio->text"),
    ("imagebind", "audiocaps", "IB",   "text->audio", "audio->text"),
]

DATASET_LABELS = {
    "mscoco": "MSCOCO",
    "flickr30k": "Flickr30K",
    "clotho": "Clotho",
    "audiocaps": "AudioCaps",
}

METHODS = {
    "vanilla_rabitq": "van",
    "vanilla_rabitq_meanshift": "ms",
    "pmc_1.00": "pmc",
}

# Manuscript tab:mainresults "with reranking" cell values (K'=400), as printed
# "V/M/P" strings. Keyed by (dataset_label, display_enc) -> {"q": ..., "db": ...}.
PAPER_EXPECTED: dict[tuple[str, str], dict[str, str]] = {
    ("MSCOCO", "CLIP"):    {"q": ".81/.79/.90", "db": ".72/.74/.87"},
    ("MSCOCO", "CL-L"):    {"q": ".79/.77/.91", "db": ".66/.73/.89"},
    ("MSCOCO", "IB"):      {"q": ".93/.87/.97", "db": ".93/.88/.95"},
    ("Flickr30K", "CL-L"): {"q": ".91/.85/.97", "db": ".81/.82/.96"},
    ("Clotho", "IB"):      {"q": "1.00/.96/1.00", "db": ".92/.87/.98"},
    ("AudioCaps", "IB"):   {"q": ".98/.88/.98", "db": ".90/.84/.97"},
}

FIELDNAMES = [
    "dataset", "enc", "direction_role", "direction", "nprobe", "rerank_k",
    "r100_van", "r100_ms", "r100_pmc",
    "delta_vp", "delta_mp", "qps_cellmean",
    "minK_van", "minK_ms", "minK_pmc",
]


# --- IO helpers --------------------------------------------------------------

def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "results").is_dir():
            return parent
    raise RuntimeError("Could not find project root containing results/")


PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "results"
SOURCE_CSV = RESULTS_DIR / "sources" / "rerank_subset_sqrtN_kfine_seed42.csv"
OUTPUT_CSV = RESULTS_DIR / "tables" / "rerank_deployable_reproduced.csv"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- Formatting --------------------------------------------------------------

def round_half_up(value: float, digits: int) -> float:
    quant = Decimal("1") if digits == 0 else Decimal("1").scaleb(-digits)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def fmt_recall(value: float, digits: int = 2) -> str:
    out = f"{round_half_up(value, digits):.{digits}f}"
    return out[1:] if out.startswith("0") else out


def fmt_delta(base_r100: float, pmc_r100: float, digits: int = 2) -> str:
    """Relative percent delta computed from the DISPLAYED (rounded) recalls."""
    quant = Decimal("1").scaleb(-digits)
    base = Decimal(str(base_r100)).quantize(quant, rounding=ROUND_HALF_UP)
    pmc = Decimal(str(pmc_r100)).quantize(quant, rounding=ROUND_HALF_UP)
    pct = ((pmc - base) / base * Decimal(100)).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    )
    return f"{int(pct):+d}%"


# --- Lookups -----------------------------------------------------------------

def pick(rows: list[dict[str, str]], backbone: str, dataset: str, method: str,
         direction: str, nprobe: int, rerank_k: int) -> dict[str, str]:
    hits = [
        r for r in rows
        if r["backbone"] == backbone and r["dataset"] == dataset
        and r["method"] == method and r["direction"] == direction
        and int(r["nprobe"]) == nprobe and int(r["rerank_k"]) == rerank_k
    ]
    if len(hits) != 1:
        raise ValueError(
            f"Expected one row for backbone={backbone} dataset={dataset} "
            f"method={method} dir={direction} np={nprobe} k={rerank_k}, got {len(hits)}"
        )
    return hits[0]


def r100_at(rows, backbone, dataset, method, direction, nprobe, rerank_k) -> float:
    return float(pick(rows, backbone, dataset, method, direction, nprobe, rerank_k)["r100"])


def min_k_for_deployable(rows, backbone, dataset, method, direction, nprobe) -> int | None:
    """Smallest K' on the ladder whose 2-decimal R@100 reaches the threshold."""
    for k in RERANK_K_LADDER:
        r100 = r100_at(rows, backbone, dataset, method, direction, nprobe, k)
        if round_half_up(r100, 2) >= DEPLOYABLE_THRESHOLD:
            return k
    return None


def qps_cell_mean(rows, backbone, dataset, direction, nprobe, rerank_k) -> float:
    vals = [
        float(pick(rows, backbone, dataset, method, direction, nprobe, rerank_k)["qps"])
        for method in METHODS
    ]
    return sum(vals) / len(vals)


def fmt_min_k(k: int | None) -> str:
    return "--" if k is None else str(k)


# --- n_probe selection (recall-tuned ~ n_list) -------------------------------

def recall_tuned_nprobe(
    rows: list[dict[str, str]],
    backbone: str,
    dataset: str,
    direction: str,
) -> int:
    """Largest swept n_probe (capped <= n_list) for the group -- recall-tuned.

    On these subset corpora the largest swept n_probe is ~ n_list (near
    exhaustive), the operating point the cascade table reports.
    """
    group = [
        r for r in rows
        if r["backbone"] == backbone
        and r["dataset"] == dataset
        and r["direction"] == direction
    ]
    if not group:
        raise ValueError(
            f"No rows for backbone={backbone!r} dataset={dataset!r} "
            f"direction={direction!r}"
        )
    nlist = int(group[0]["nlist"])
    available = [int(r["nprobe"]) for r in group if int(r["nprobe"]) <= nlist]
    return max(available)


# --- Build -------------------------------------------------------------------

def build_records(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for backbone, dataset, enc, q_dir, db_dir in ROWS:
        q_nprobe = recall_tuned_nprobe(rows, backbone, dataset, q_dir)
        db_nprobe = recall_tuned_nprobe(rows, backbone, dataset, db_dir)
        for role, direction, nprobe in (
            ("q->db", q_dir, q_nprobe),
            ("db->q", db_dir, db_nprobe),
        ):
            van = r100_at(rows, backbone, dataset, "vanilla_rabitq", direction, nprobe, CELL_K)
            ms = r100_at(rows, backbone, dataset, "vanilla_rabitq_meanshift", direction, nprobe, CELL_K)
            pmc = r100_at(rows, backbone, dataset, "pmc_1.00", direction, nprobe, CELL_K)
            records.append({
                "dataset": DATASET_LABELS[dataset],
                "enc": enc,
                "direction_role": role,
                "direction": direction,
                "nprobe": str(nprobe),
                "rerank_k": str(CELL_K),
                "r100_van": fmt_recall(van),
                "r100_ms": fmt_recall(ms),
                "r100_pmc": fmt_recall(pmc),
                "delta_vp": fmt_delta(van, pmc),
                "delta_mp": fmt_delta(ms, pmc),
                "qps_cellmean": f"{qps_cell_mean(rows, backbone, dataset, direction, nprobe, CELL_K):.0f}",
                "minK_van": fmt_min_k(min_k_for_deployable(rows, backbone, dataset, "vanilla_rabitq", direction, nprobe)),
                "minK_ms": fmt_min_k(min_k_for_deployable(rows, backbone, dataset, "vanilla_rabitq_meanshift", direction, nprobe)),
                "minK_pmc": fmt_min_k(min_k_for_deployable(rows, backbone, dataset, "pmc_1.00", direction, nprobe)),
            })
    return records


def write_output_csv(records: list[dict[str, str]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


# --- Verification ------------------------------------------------------------

def check_paper_values(records: list[dict[str, str]]) -> list[str]:
    """Compare printed cascade cells against PAPER_EXPECTED; return mismatches."""
    by_key: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for r in records:
        by_key.setdefault((r["dataset"], r["enc"]), {})[r["direction_role"]] = r
    mismatches: list[str] = []
    for key, expect in PAPER_EXPECTED.items():
        roles = by_key.get(key)
        if roles is None:
            mismatches.append(f"{key}: no reproduced row")
            continue
        for role_short, role_key in (("q", "q->db"), ("db", "db->q")):
            rec = roles[role_key]
            actual = f"{rec['r100_van']}/{rec['r100_ms']}/{rec['r100_pmc']}"
            if actual != expect[role_short]:
                mismatches.append(
                    f"{key[0]}/{key[1]} {role_key}: actual {actual} vs paper {expect[role_short]}"
                )
    return mismatches


def check_uniform_dominance(rows: list[dict[str, str]]) -> list[str]:
    """Verify PMC >= max(Vanilla, MeanShift) at every K' for every cell."""
    violations: list[str] = []
    for backbone, dataset, enc, q_dir, db_dir in ROWS:
        for direction in (q_dir, db_dir):
            nprobe = recall_tuned_nprobe(rows, backbone, dataset, direction)
            for k in RERANK_K_LADDER:
                van = round_half_up(r100_at(rows, backbone, dataset, "vanilla_rabitq", direction, nprobe, k), 2)
                ms = round_half_up(r100_at(rows, backbone, dataset, "vanilla_rabitq_meanshift", direction, nprobe, k), 2)
                pmc = round_half_up(r100_at(rows, backbone, dataset, "pmc_1.00", direction, nprobe, k), 2)
                if pmc < van or pmc < ms:
                    violations.append(
                        f"{DATASET_LABELS[dataset]}/{enc} {direction} np={nprobe} K'={k}: "
                        f"PMC {pmc:.2f} < max(van {van:.2f}, ms {ms:.2f})"
                    )
    return violations


# --- Reporting ---------------------------------------------------------------

def print_console_table(records: list[dict[str, str]]) -> None:
    header = (
        f"{'Dataset':<10} {'Enc':<5} {'Role':<6} {'Dir':<12} {'np':>3} {'K':>4} "
        f"{'van':>5} {'ms':>5} {'pmc':>5} {'dVP':>5} {'dMP':>5} {'QPS':>7} "
        f"{'mK_v':>5} {'mK_m':>5} {'mK_p':>5}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        print(
            f"{r['dataset']:<10} {r['enc']:<5} {r['direction_role']:<6} "
            f"{r['direction']:<12} {r['nprobe']:>3} {r['rerank_k']:>4} "
            f"{r['r100_van']:>5} {r['r100_ms']:>5} {r['r100_pmc']:>5} "
            f"{r['delta_vp']:>5} {r['delta_mp']:>5} {r['qps_cellmean']:>7} "
            f"{r['minK_van']:>5} {r['minK_ms']:>5} {r['minK_pmc']:>5}"
        )


def print_latex_rows(records: list[dict[str, str]]) -> None:
    """Emit tab:mainresults rerank rows: Dataset & Enc & n_probe & q->db & db->q.

    Each cell is Vanilla/MeanShift/PMC; PMC is bolded (best-or-tied in every
    cell under the recall-tuned budget). When q and db directions pick a
    different n_probe (Clotho) the column shows "q_np/db_np".
    """
    print("% --- auto-generated by reproduce_tab2_rerank.py ---")
    print("% tab:mainresults 'with reranking' group, §4.3 Robustness under Exact Reranking")
    print("% columns: Dataset & Enc & n_probe & q->db R@100 (V/M/P) & db->q R@100 (V/M/P)")
    by_key: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for r in records:
        by_key.setdefault((r["dataset"], r["enc"]), {})[r["direction_role"]] = r
    for (dataset, enc), roles in by_key.items():
        q = roles["q->db"]
        db = roles["db->q"]
        q_np = q["nprobe"]
        db_np = db["nprobe"]
        np_label = q_np if q_np == db_np else f"{q_np}/{db_np}"
        line = (
            f"{dataset:<10} & {enc:<4} & {np_label} & "
            f"{q['r100_van']}\\,/\\,{q['r100_ms']}\\,/\\,\\textbf{{{q['r100_pmc']}}} & "
            f"{db['r100_van']}\\,/\\,{db['r100_ms']}\\,/\\,\\textbf{{{db['r100_pmc']}}} \\\\"
        )
        print(line)


def main() -> None:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Missing source CSV: {SOURCE_CSV}")
    rows = read_csv_rows(SOURCE_CSV)
    records = build_records(rows)
    write_output_csv(records)
    print_console_table(records)
    print()
    print_latex_rows(records)

    mismatches = check_paper_values(records)
    violations = check_uniform_dominance(rows)
    print()
    if mismatches:
        print("# REPRODUCE-MISMATCH cells (actual vs paper):")
        for m in mismatches:
            print(f"  {m}")
    else:
        print("# OK: all tab:mainresults rerank cells (K'=400) match the manuscript.")
    if violations:
        print("# REPRODUCE-MISMATCH uniform-dominance violations:")
        for v in violations:
            print(f"  {v}")
    else:
        print("# OK: PMC >= max(Vanilla, MeanShift) at every K' in "
              f"{RERANK_K_LADDER} for every cell.")

    print(f"\nWrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
