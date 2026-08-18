"""Reproduce fig:analysis-bcd (three panels).

This script runs NO FAISS.

  Panel (b) -- selective PMC: from results/figures/selective_pmc_rabitq.csv, R@100 by
    dataset {mscoco, clotho} x direction x top_p_percent. CLEAN.

  Panel (c) -- Pareto frontier: from
    results/figures/pmc_qps_pareto_clip_mscoco_seed42.csv (text->image), R@100 + QPS
    across the nprobe ladder per method. CLEAN.

  Panel (a) -- alpha sweep (RaBitQ R@100 across alpha in {0,.25,.5,.75,1}): no
    faithful CSV exists (the figure currently hardcodes the array in
    paper/figures/fig3_analysis.py). Wired to the intermediate CSV produced by
    the FAISS-bound emit script scripts/figures/emit_fig_alpha_sweep.py. PENDING until
    that runs.

Outputs:
  - stdout: per-panel tables
  - results/figures/fig3_analysis_bcd_reproduced.csv
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# --- Configuration ----------------------------------------------------------

# Panel (b): selective PMC
SELECTIVE_DATASETS = ["mscoco", "clotho"]
TOP_P_LADDER = [0, 5, 10, 20, 50, 100]

# Panel (c): Pareto (text->image only) -- methods present in the pareto CSV.
PARETO_DIRECTION = "text->image"
PARETO_METHODS = ["vanilla_rabitq", "ivfpq_meanshift_64B", "pmc_1.00"]

# Panel (a): alpha sweep operating points.
ALPHA_LADDER = [0.0, 0.25, 0.5, 0.75, 1.0]
ALPHA_SERIES = [
    ("mscoco", "text2image"),
    ("mscoco", "image2text"),
    ("clotho", "text2audio"),
    ("clotho", "audio2text"),
]

PENDING = "--"
PENDING_NOTE = "[PENDING -- run emit_fig_alpha_sweep.py]"


# --- IO helpers --------------------------------------------------------------

def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "results").is_dir():
            return parent
    raise RuntimeError("Could not find project root containing results/")


PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "results"
SELECTIVE_CSV = RESULTS_DIR / "figures" / "selective_pmc_rabitq.csv"
PARETO_CSV = RESULTS_DIR / "figures" / "pmc_qps_pareto_clip_mscoco_seed42.csv"
ALPHA_CSV = RESULTS_DIR / "figures" / "fig_alpha_sweep_rabitq.csv"
OUTPUT_CSV = RESULTS_DIR / "figures" / "fig3_analysis_bcd_reproduced.csv"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- Formatting --------------------------------------------------------------

def round_half_up(value: float, digits: int) -> float:
    quant = Decimal("1") if digits == 0 else Decimal("1").scaleb(-digits)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def fmt_recall(value: float, digits: int = 4) -> str:
    return f"{round_half_up(value, digits):.{digits}f}"


# --- Panel (b): selective PMC -----------------------------------------------

def pick_selective(rows: list[dict[str, str]], dataset: str, direction: str,
                   top_p: int) -> dict[str, str]:
    hits = [
        r for r in rows
        if r["dataset"] == dataset and r["direction"] == direction
        and int(r["top_p_percent"]) == top_p
    ]
    if len(hits) != 1:
        raise ValueError(
            f"Expected one selective row for dataset={dataset} dir={direction} "
            f"top_p={top_p}, got {len(hits)}"
        )
    return hits[0]


def build_selective(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    directions = {
        "mscoco": ["t2i", "i2t"],
        "clotho": ["t2a", "a2t"],
    }
    records: list[dict[str, str]] = []
    for dataset in SELECTIVE_DATASETS:
        for direction in directions[dataset]:
            for top_p in TOP_P_LADDER:
                row = pick_selective(rows, dataset, direction, top_p)
                records.append({
                    "panel": "b",
                    "dataset": dataset,
                    "direction": direction,
                    "x": str(top_p),
                    "r100": fmt_recall(float(row["r100"])),
                    "qps": "",
                })
    return records


# --- Panel (c): Pareto -------------------------------------------------------

def build_pareto(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for method in PARETO_METHODS:
        method_rows = [
            r for r in rows
            if r["method"] == method and r["direction"] == PARETO_DIRECTION
        ]
        if not method_rows:
            raise ValueError(f"No pareto rows for method={method}")
        for r in sorted(method_rows, key=lambda x: int(x["nprobe"])):
            records.append({
                "panel": "c",
                "dataset": "mscoco",
                "direction": method,
                "x": r["nprobe"],
                "r100": fmt_recall(float(r["r100"])),
                "qps": f"{float(r['qps']):.1f}",
            })
    return records


# --- Panel (a): alpha sweep --------------------------------------------------

def load_alpha(path: Path) -> dict[tuple[str, str, float], float]:
    if not path.exists():
        return {}
    out: dict[tuple[str, str, float], float] = {}
    for r in read_csv_rows(path):
        out[(r["dataset"], r["direction"], float(r["alpha"]))] = float(r["r100"])
    return out


def build_alpha(alpha_map: dict[tuple[str, str, float], float],
                ) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for dataset, direction in ALPHA_SERIES:
        for alpha in ALPHA_LADDER:
            val = alpha_map.get((dataset, direction, alpha))
            records.append({
                "panel": "a",
                "dataset": dataset,
                "direction": direction,
                "x": f"{alpha:g}",
                "r100": PENDING if val is None else fmt_recall(val),
                "qps": "",
            })
    return records


# --- Output ------------------------------------------------------------------

FIELDNAMES = ["panel", "dataset", "direction", "x", "r100", "qps"]


def write_output_csv(records: list[dict[str, str]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


# --- Reporting ---------------------------------------------------------------

def print_selective(records: list[dict[str, str]]) -> None:
    print("== Panel (b): selective PMC (R@100 by top_p%) ==")
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for r in records:
        by_key.setdefault((r["dataset"], r["direction"]), {})[r["x"]] = r["r100"]
    header = f"{'Dataset/Dir':<16}" + "".join(f"{'p=' + str(p):>9}" for p in TOP_P_LADDER)
    print(header)
    print("-" * len(header))
    for (dataset, direction), vals in by_key.items():
        print(f"{dataset + '/' + direction:<16}"
              + "".join(f"{vals[str(p)]:>9}" for p in TOP_P_LADDER))


def print_pareto(records: list[dict[str, str]]) -> None:
    print("\n== Panel (c): Pareto frontier (text->image; R@100 @ QPS) ==")
    by_method: dict[str, list[dict[str, str]]] = {}
    for r in records:
        by_method.setdefault(r["direction"], []).append(r)
    for method, rs in by_method.items():
        print(f"{method}:")
        for r in rs:
            print(f"    np={r['x']:>3}  R@100={r['r100']}  QPS={r['qps']}")


def print_alpha(records: list[dict[str, str]], pending: bool) -> None:
    print("\n== Panel (a): alpha sweep (RaBitQ R@100) ==")
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for r in records:
        by_key.setdefault((r["dataset"], r["direction"]), {})[r["x"]] = r["r100"]
    header = f"{'Dataset/Dir':<18}" + "".join(f"{'a=' + f'{a:g}':>9}" for a in ALPHA_LADDER)
    print(header)
    print("-" * len(header))
    for (dataset, direction), vals in by_key.items():
        print(f"{dataset + '/' + direction:<18}"
              + "".join(f"{vals[f'{a:g}']:>9}" for a in ALPHA_LADDER))
    if pending:
        print(f"NOTE: panel (a) {PENDING_NOTE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce fig:analysis-bcd")
    parser.parse_args()

    if not SELECTIVE_CSV.exists():
        raise FileNotFoundError(f"Missing source CSV: {SELECTIVE_CSV}")
    if not PARETO_CSV.exists():
        raise FileNotFoundError(f"Missing source CSV: {PARETO_CSV}")

    selective = build_selective(read_csv_rows(SELECTIVE_CSV))
    pareto = build_pareto(read_csv_rows(PARETO_CSV))
    alpha_map = load_alpha(ALPHA_CSV)
    alpha = build_alpha(alpha_map)

    records = alpha + selective + pareto
    write_output_csv(records)

    print_selective(selective)
    print_pareto(pareto)
    print_alpha(alpha, pending=len(alpha_map) == 0)
    print(f"\nWrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
