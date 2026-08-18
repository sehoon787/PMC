"""Reproduce the rerank K'-sweep ABLATION (oversampling deployable-recall).

NOTE: This script provides the LAION-400M K'-sweep deployable-recall ablation
backing §4.3 Robustness (a repo-only ablation; LAION rerank is omitted "---" in
Table 2 at 407M scale). Table 2's first-stage columns are reproduced by
reproduce_tab2_main.py; its reranked columns by reproduce_tab2_rerank.py.

This script runs NO FAISS. It reads existing rerank artifacts:

  Small 6 rows (MSCOCO/CLIP, MSCOCO/CL-L, MSCOCO/IB, Flickr30K/CL-L,
    Clotho/IB, AudioCaps/IB): source results/sources/rerank_subset_seed42.csv. Each
    cell is Vanilla / MeanShift / PMC R@100 at fixed nprobe with the chosen
    oversampling K' (image family K'=500, audio family K'=200 by default), with
    relative R@100 deltas and a min-K' deployable-recall diagnostic. This ports
    the verified logic from reproduce_tab2_rerank.py.

  LAION-400M row: forward source
    results/sources/pmc_laion400m_rerank_nlist80k_seed42.csv; reverse source
    results/sources/pmc_laion400m_reverse_rerank_nlist80k_seed42.csv. The reverse file
    may not exist (its experiment was deferred off the critical path). When it
    is missing, the forward (q->db) cells are printed and the reverse (db->q)
    cells emit "--" with a clear "[PENDING reverse rerank CSV]" note. No crash.

CLI options (encoded in the output filename):
  --rerank / --no-rerank   default on. When off, uses rerank_k=0 (no oversampling).
  --nprobe                 default 64.
  --rerank-k               override the K' (default per-family: 500 image / 200 audio).

Outputs:
  - stdout: human-readable table + LaTeX rows
  - results/ablation_rerank_reproduced__np{nprobe}_k{rerankk}_{rerank|norerank}.csv
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# --- Configuration ----------------------------------------------------------

DEPLOYABLE_THRESHOLD = 0.90
RERANK_K_LADDER = [0, 100, 200, 500, 1000]
DEFAULT_NPROBE = 64

# (backbone, dataset, display_enc, q->db dir, db->q dir, default K').
ROWS = [
    ("clip", "mscoco", "CLIP", "text->image", "image->text", 500),
    ("clip-l", "mscoco", "CL-L", "text->image", "image->text", 500),
    ("imagebind", "mscoco", "IB", "text->image", "image->text", 500),
    ("clip-l", "flickr30k", "CL-L", "text->image", "image->text", 500),
    ("imagebind", "clotho", "IB", "text->audio", "audio->text", 200),
    ("imagebind", "audiocaps", "IB", "text->audio", "audio->text", 200),
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

# LAION-400M method names in the rerank CSVs.
LAION_METHODS = ["vanilla_rabitq", "vanilla_rabitq_meanshift", "pmc_1.00"]

PENDING = "--"
LAION_PENDING_NOTE = "[PENDING reverse rerank CSV]"

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
SUBSET_CSV = RESULTS_DIR / "sources" / "rerank_subset_seed42.csv"
LAION_FWD_CSV = RESULTS_DIR / "sources" / "pmc_laion400m_rerank_nlist80k_seed42.csv"
LAION_REV_CSV = RESULTS_DIR / "sources" / "pmc_laion400m_reverse_rerank_nlist80k_seed42.csv"


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
    """Relative percent delta computed from the DISPLAYED (rounded) recalls.

    Both recalls are first rounded to `digits` decimals (the same precision
    shown in the table), then the percent change is computed end-to-end in
    Decimal with round-half-up, so the delta is consistent with the printed
    recall values and free of float rounding artifacts.
    """
    quant = Decimal("1").scaleb(-digits)
    base = Decimal(str(base_r100)).quantize(quant, rounding=ROUND_HALF_UP)
    pmc = Decimal(str(pmc_r100)).quantize(quant, rounding=ROUND_HALF_UP)
    pct = ((pmc - base) / base * Decimal(100)).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    )
    return f"{int(pct):+d}%"


def fmt_min_k(k: int | None) -> str:
    return "--" if k is None else str(k)


# --- Small-subset lookups (ported from the deployable aggregator) -----------

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
            f"method={method} dir={direction} np={nprobe} k={rerank_k}, "
            f"got {len(hits)}"
        )
    return hits[0]


def r100_at(rows, backbone, dataset, method, direction, nprobe, rerank_k) -> float:
    return float(pick(rows, backbone, dataset, method, direction, nprobe, rerank_k)["r100"])


def min_k_for_deployable(rows, backbone, dataset, method, direction,
                         nprobe) -> int | None:
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


# --- Build: small subset -----------------------------------------------------

def build_subset_records(rows: list[dict[str, str]], nprobe: int,
                         rerank_k_override: int | None,
                         use_rerank: bool) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for backbone, dataset, enc, q_dir, db_dir, default_k in ROWS:
        if not use_rerank:
            kprime = 0
        elif rerank_k_override is not None:
            kprime = rerank_k_override
        else:
            kprime = default_k
        for role, direction in (("q->db", q_dir), ("db->q", db_dir)):
            van = r100_at(rows, backbone, dataset, "vanilla_rabitq", direction, nprobe, kprime)
            ms = r100_at(rows, backbone, dataset, "vanilla_rabitq_meanshift", direction, nprobe, kprime)
            pmc = r100_at(rows, backbone, dataset, "pmc_1.00", direction, nprobe, kprime)
            records.append({
                "dataset": DATASET_LABELS[dataset],
                "enc": enc,
                "direction_role": role,
                "direction": direction,
                "nprobe": str(nprobe),
                "rerank_k": str(kprime),
                "r100_van": fmt_recall(van),
                "r100_ms": fmt_recall(ms),
                "r100_pmc": fmt_recall(pmc),
                "delta_vp": fmt_delta(van, pmc),
                "delta_mp": fmt_delta(ms, pmc),
                "qps_cellmean": f"{qps_cell_mean(rows, backbone, dataset, direction, nprobe, kprime):.0f}",
                "minK_van": fmt_min_k(min_k_for_deployable(rows, backbone, dataset, "vanilla_rabitq", direction, nprobe)),
                "minK_ms": fmt_min_k(min_k_for_deployable(rows, backbone, dataset, "vanilla_rabitq_meanshift", direction, nprobe)),
                "minK_pmc": fmt_min_k(min_k_for_deployable(rows, backbone, dataset, "pmc_1.00", direction, nprobe)),
            })
    return records


# --- Build: LAION-400M -------------------------------------------------------

def laion_r100(rows: list[dict[str, str]], method: str, nprobe: int,
               rerank_k: int) -> float | None:
    hits = [
        r for r in rows
        if r["method"] == method and int(r["nprobe"]) == nprobe
        and int(r["rerank_k"]) == rerank_k
    ]
    if len(hits) != 1:
        return None
    return float(hits[0]["r100"])


def build_laion_record(role: str, direction: str, source_rows: list[dict[str, str]] | None,
                       nprobe: int, rerank_k: int) -> dict[str, str]:
    """Build one LAION row half (q->db forward, or db->q reverse).

    When source_rows is None (CSV missing), every cell is the pending marker.
    """
    base = {
        "dataset": "LAION-400M",
        "enc": "CLIP",
        "direction_role": role,
        "direction": direction,
        "nprobe": str(nprobe),
        "rerank_k": str(rerank_k),
        "qps_cellmean": "",
        "minK_van": PENDING,
        "minK_ms": PENDING,
        "minK_pmc": PENDING,
    }
    if source_rows is None:
        base.update({
            "r100_van": PENDING, "r100_ms": PENDING, "r100_pmc": PENDING,
            "delta_vp": PENDING, "delta_mp": PENDING,
        })
        return base
    van = laion_r100(source_rows, "vanilla_rabitq", nprobe, rerank_k)
    ms = laion_r100(source_rows, "vanilla_rabitq_meanshift", nprobe, rerank_k)
    pmc = laion_r100(source_rows, "pmc_1.00", nprobe, rerank_k)
    if van is None or ms is None or pmc is None:
        base.update({
            "r100_van": PENDING if van is None else fmt_recall(van, 3),
            "r100_ms": PENDING if ms is None else fmt_recall(ms, 3),
            "r100_pmc": PENDING if pmc is None else fmt_recall(pmc, 3),
            "delta_vp": PENDING, "delta_mp": PENDING,
        })
        return base
    base.update({
        "r100_van": fmt_recall(van, 3),
        "r100_ms": fmt_recall(ms, 3),
        "r100_pmc": fmt_recall(pmc, 3),
        "delta_vp": fmt_delta(van, pmc, digits=3),
        "delta_mp": fmt_delta(ms, pmc, digits=3),
    })
    return base


def build_laion_records(nprobe: int, rerank_k: int,
                        ) -> tuple[list[dict[str, str]], bool]:
    """Returns (records, reverse_pending)."""
    fwd_rows = read_csv_rows(LAION_FWD_CSV) if LAION_FWD_CSV.exists() else None
    rev_rows = read_csv_rows(LAION_REV_CSV) if LAION_REV_CSV.exists() else None
    records = [
        build_laion_record("q->db", "forward", fwd_rows, nprobe, rerank_k),
        build_laion_record("db->q", "reverse", rev_rows, nprobe, rerank_k),
    ]
    return records, rev_rows is None


# --- Output ------------------------------------------------------------------

def output_csv_path(nprobe: int, rerank_k_label: str, use_rerank: bool) -> Path:
    suffix = "rerank" if use_rerank else "norerank"
    return RESULTS_DIR / "tables" / f"ablation_rerank_reproduced__np{nprobe}_k{rerank_k_label}_{suffix}.csv"


def write_output_csv(records: list[dict[str, str]], path: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


# --- Reporting ---------------------------------------------------------------

def print_console_table(records: list[dict[str, str]]) -> None:
    header = (
        f"{'Dataset':<11} {'Enc':<5} {'Role':<6} {'Dir':<12} {'np':>3} {'K':>4} "
        f"{'van':>5} {'ms':>5} {'pmc':>5} {'dVP':>5} {'dMP':>5} {'QPS':>7} "
        f"{'mK_v':>5} {'mK_m':>5} {'mK_p':>5}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        print(
            f"{r['dataset']:<11} {r['enc']:<5} {r['direction_role']:<6} "
            f"{r['direction']:<12} {r['nprobe']:>3} {r['rerank_k']:>4} "
            f"{r['r100_van']:>5} {r['r100_ms']:>5} {r['r100_pmc']:>5} "
            f"{r['delta_vp']:>5} {r['delta_mp']:>5} {r['qps_cellmean']:>7} "
            f"{r['minK_van']:>5} {r['minK_ms']:>5} {r['minK_pmc']:>5}"
        )


def print_latex_rows(records: list[dict[str, str]]) -> None:
    print("\n% --- auto-generated by reproduce_ablation_rerank.py ---")
    print("% columns: Dataset & Enc & (np,K') & q->db R@100 (V/M/P) & dVP & dMP "
          "& db->q R@100 (V/M/P) & dVP & dMP & QPS(q;db)")
    by_key: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    order: list[tuple[str, str]] = []
    for r in records:
        key = (r["dataset"], r["enc"])
        if key not in by_key:
            by_key[key] = {}
            order.append(key)
        by_key[key][r["direction_role"]] = r
    for key in order:
        roles = by_key[key]
        q = roles["q->db"]
        db = roles["db->q"]
        kprime = q["rerank_k"]
        qps = f"{q['qps_cellmean']}\\,;\\,{db['qps_cellmean']}" if q["qps_cellmean"] else "--"
        line = (
            f"{q['dataset']:<11} & {q['enc']:<4} & $({q['nprobe']},{kprime})$ & "
            f"{q['r100_van']}\\,/\\,{q['r100_ms']}\\,/\\,\\textbf{{{q['r100_pmc']}}} & "
            f"{q['delta_vp']} & {q['delta_mp']} & "
            f"{db['r100_van']}\\,/\\,{db['r100_ms']}\\,/\\,\\textbf{{{db['r100_pmc']}}} & "
            f"{db['delta_vp']} & {db['delta_mp']} & "
            f"{qps} \\\\"
        )
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce tab:main (small + LAION)")
    parser.add_argument("--rerank", dest="rerank", action="store_true", default=True,
                        help="Use oversampling K' (default).")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false",
                        help="Disable oversampling (rerank_k=0).")
    parser.add_argument("--nprobe", type=int, default=DEFAULT_NPROBE)
    parser.add_argument("--rerank-k", type=int, default=None,
                        help="Override K' (default per-family: 500 image / 200 audio).")
    args = parser.parse_args()

    if not SUBSET_CSV.exists():
        raise FileNotFoundError(f"Missing source CSV: {SUBSET_CSV}")
    subset_rows = read_csv_rows(SUBSET_CSV)

    subset_records = build_subset_records(
        subset_rows, args.nprobe, args.rerank_k, args.rerank,
    )

    # LAION uses the same K' as the image family unless overridden; with
    # --no-rerank it uses rerank_k=0.
    if not args.rerank:
        laion_k = 0
    elif args.rerank_k is not None:
        laion_k = args.rerank_k
    else:
        laion_k = 500
    laion_records, reverse_pending = build_laion_records(args.nprobe, laion_k)

    all_records = subset_records + laion_records

    # Output filename encodes the K' label (subset image-family default if unset).
    if args.rerank_k is not None:
        k_label = str(args.rerank_k)
    elif not args.rerank:
        k_label = "0"
    else:
        k_label = "perfamily"
    out_path = output_csv_path(args.nprobe, k_label, args.rerank)
    write_output_csv(all_records, out_path)

    print_console_table(all_records)
    if reverse_pending:
        print(f"\nNOTE: LAION-400M db->q (reverse) cells {LAION_PENDING_NOTE}")
    print_latex_rows(all_records)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
