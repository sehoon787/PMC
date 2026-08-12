"""Reproduce Table 4 (tab:mech_extra): component ablation and IVF-RaBitQ controls.

This script runs NO FAISS.

  Part (b) -- component ablation: from results/mechanism_additional_controls.csv,
    R@100 by index_type {binary_flat, ivf_rabitq} x direction {text2image,
    image2text} x mode {vanilla, query_only, db_only, both}, 3 decimals. CLEAN.

  Part (c) -- IVF controls: same CSV, modes {vanilla, db_only,
    random_direction_same_norm, shuffled_gap, sign_flipped_gap, db_only_no_norm}.
    Rand/Shuf are reported as mean +/- SAMPLE std (ddof=1) over 5 seeds; that
    reproduces the paper's +/-.009 (population std would give .008). CLEAN.

NOTE: The calibration-sensitivity panel (formerly panel (a)) was removed from
Table 4. Its logic now lives in scripts/analysis/verify_calibration.py, which
backs the §4.4 Ablation prose claims. This script reproduces ONLY parts (b) and
(c) of Table 4.

Outputs:
  - stdout: human-readable tables + LaTeX rows (parts b and c)
  - results/tab5_mech_extra_reproduced.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# --- Configuration ----------------------------------------------------------

DIRECTIONS = ["text2image", "image2text"]

# Part (b): component ablation
ABLATION_INDEX = ["binary_flat", "ivf_rabitq"]
ABLATION_MODES = ["vanilla", "query_only", "db_only", "both"]

# Part (c): IVF controls (ivf_rabitq only)
IVF_CONTROL_MODES = [
    "vanilla", "db_only",
    "random_direction_same_norm", "shuffled_gap",
    "sign_flipped_gap", "db_only_no_norm",
]
# Modes whose value is a mean +/- std across 5 seeds.
SEEDED_MODES = {"random_direction_same_norm", "shuffled_gap"}


# --- IO helpers --------------------------------------------------------------

def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "results").is_dir():
            return parent
    raise RuntimeError("Could not find project root containing results/")


PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "results"
CONTROLS_CSV = RESULTS_DIR / "mechanism_additional_controls.csv"
OUTPUT_CSV = RESULTS_DIR / "tab5_mech_extra_reproduced.csv"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- Formatting --------------------------------------------------------------

def round_half_up(value: float, digits: int) -> float:
    quant = Decimal("1") if digits == 0 else Decimal("1").scaleb(-digits)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def fmt_recall(value: float, digits: int = 3) -> str:
    out = f"{round_half_up(value, digits):.{digits}f}"
    return out[1:] if out.startswith("0") else out


def fmt_mean_std(mean: float, std: float, digits: int = 3) -> str:
    return f"{fmt_recall(mean, digits)}+/-{fmt_recall(std, digits)}"


# --- Lookups -----------------------------------------------------------------

def control_r100(rows: list[dict[str, str]], index_type: str, direction: str,
                 mode: str) -> float:
    hits = [
        r for r in rows
        if r["index_type"] == index_type and r["direction"] == direction
        and r["mode"] == mode
    ]
    if len(hits) != 1:
        raise ValueError(
            f"Expected one row for index={index_type} dir={direction} "
            f"mode={mode}, got {len(hits)}"
        )
    return float(hits[0]["r100"])


def control_r100_seeded(rows: list[dict[str, str]], index_type: str,
                        direction: str, mode: str) -> list[float]:
    return [
        float(r["r100"])
        for r in rows
        if r["index_type"] == index_type and r["direction"] == direction
        and r["mode"] == mode
    ]


# --- Build: part (b) ---------------------------------------------------------

def build_ablation(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for index_type in ABLATION_INDEX:
        for direction in DIRECTIONS:
            rec = {"part": "b", "index_type": index_type, "direction": direction}
            for mode in ABLATION_MODES:
                rec[mode] = fmt_recall(control_r100(rows, index_type, direction, mode))
            records.append(rec)
    return records


# --- Build: part (c) ---------------------------------------------------------

def build_ivf_controls(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for direction in DIRECTIONS:
        rec = {"part": "c", "index_type": "ivf_rabitq", "direction": direction}
        for mode in IVF_CONTROL_MODES:
            if mode in SEEDED_MODES:
                vals = control_r100_seeded(rows, "ivf_rabitq", direction, mode)
                if len(vals) < 2:
                    raise ValueError(
                        f"Expected >=2 seeds for {mode}/{direction}, got {len(vals)}"
                    )
                rec[mode] = fmt_mean_std(statistics.mean(vals), statistics.stdev(vals))
            else:
                rec[mode] = fmt_recall(control_r100(rows, "ivf_rabitq", direction, mode))
        records.append(rec)
    return records


# --- Output ------------------------------------------------------------------

def write_output_csv(ablation: list[dict[str, str]],
                     ivf_controls: list[dict[str, str]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Union of all keys, stable order: meta first then the rest as encountered.
    meta = ["part", "index_type", "direction"]
    seen: list[str] = list(meta)
    for rec in ablation + ivf_controls:
        for k in rec:
            if k not in seen:
                seen.append(k)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=seen, restval="")
        writer.writeheader()
        writer.writerows(ablation + ivf_controls)


# --- Reporting ---------------------------------------------------------------

def print_ablation(records: list[dict[str, str]]) -> None:
    print("== Part (b): component ablation (R@100, 3 dp) ==")
    header = f"{'Index':<12} {'Dir':<11}" + "".join(f"{m:>11}" for m in ABLATION_MODES)
    print(header)
    print("-" * len(header))
    for r in records:
        print(f"{r['index_type']:<12} {r['direction']:<11}"
              + "".join(f"{r[m]:>11}" for m in ABLATION_MODES))


def print_ivf_controls(records: list[dict[str, str]]) -> None:
    print("\n== Part (c): IVF controls (R@100, 3 dp; rand/shuf = mean+/-sstd) ==")
    for r in records:
        print(f"{r['index_type']} {r['direction']}:")
        for mode in IVF_CONTROL_MODES:
            print(f"    {mode:<28} {r[mode]}")


def print_latex_rows(ablation: list[dict[str, str]],
                     ivf_controls: list[dict[str, str]]) -> None:
    print("\n% --- auto-generated by reproduce_tab5_mech_extra.py ---")
    print("% Table 4 (tab:mech_extra)")
    print("% part (b) ablation rows: Index & Dir & vanilla & query_only & db_only & both")
    for r in ablation:
        cells = " & ".join(r[m] for m in ABLATION_MODES)
        print(f"{r['index_type']:<12} & {r['direction']:<11} & {cells} \\\\")
    print("% part (c) IVF control rows: Dir & " + " & ".join(IVF_CONTROL_MODES))
    for r in ivf_controls:
        cells = " & ".join(r[m] for m in IVF_CONTROL_MODES)
        print(f"{r['direction']:<11} & {cells} \\\\")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce Table 4 (tab:mech_extra): component ablation and IVF controls"
    )
    parser.parse_args()

    if not CONTROLS_CSV.exists():
        raise FileNotFoundError(f"Missing source CSV: {CONTROLS_CSV}")
    rows = read_csv_rows(CONTROLS_CSV)

    ablation = build_ablation(rows)
    ivf_controls = build_ivf_controls(rows)

    write_output_csv(ablation, ivf_controls)
    print_ablation(ablation)
    print_ivf_controls(ivf_controls)
    print_latex_rows(ablation, ivf_controls)
    print(f"\nWrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
