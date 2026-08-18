"""Verify mechanism metrics backing §3.3 Sign-Bit Corruption Analysis prose.

This script runs NO FAISS. It verifies the metrics that back the §3.3 Sign-Bit
Corruption Analysis prose claims. These numbers appear in §3.3 prose only:
  - BinaryFlat sign-recall (vanilla -> PMC-alpha1)
  - Flip%, J@100, and n=25 gap-direction cosine

These numbers support the inline claims:
  "PMC induces 14-18% sign-bit flips with J@100 of 0.54-0.80 ...
   n=25 gap-direction cosine >= 0.95"

Outputs:
  - stdout: verified metric values
  - results/diagnostics/signbit_analysis_verified.csv
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# --- Configuration ----------------------------------------------------------

# (dataset, direction, signbit-direction-label, display label) row order.
ROWS = [
    ("mscoco", "text2image", "text2image", "MSCOCO t->i"),
    ("mscoco", "image2text", "image2text", "MSCOCO i->t"),
    ("audiocaps", "text2audio", "text2audio", "AudioCaps t->a"),
    ("audiocaps", "audio2text", "audio2text", "AudioCaps a->t"),
]

PENDING = "--"
PENDING_NOTE = "[PENDING -- run emit_signbit_metrics.py]"

FIELDNAMES = [
    "dataset", "direction", "label",
    "bin_van", "bin_pmc",
    "flip_pct", "j100", "n25_cos",
]


# --- IO helpers --------------------------------------------------------------

def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "results").is_dir():
            return parent
    raise RuntimeError("Could not find project root containing results/")


PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "results"
SIGNBIT_CSV = RESULTS_DIR / "sources" / "signbit_original_gt.csv"
METRICS_CSV = RESULTS_DIR / "sources" / "mechcontrol_metrics.csv"
OUTPUT_CSV = RESULTS_DIR / "diagnostics" / "signbit_analysis_verified.csv"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- Formatting --------------------------------------------------------------

def round_half_up(value: float, digits: int) -> float:
    quant = Decimal("1") if digits == 0 else Decimal("1").scaleb(-digits)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def fmt_recall(value: float, digits: int = 4) -> str:
    out = f"{round_half_up(value, digits):.{digits}f}"
    return out[1:] if out.startswith("0") else out


# --- Lookups -----------------------------------------------------------------

def pick_signbit(rows: list[dict[str, str]], direction: str,
                 variant: str) -> dict[str, str]:
    hits = [
        r for r in rows
        if r["method"] == "BinaryFlat" and r["direction"] == direction
        and r["variant"] == variant
    ]
    if len(hits) != 1:
        raise ValueError(
            f"Expected one BinaryFlat row for dir={direction} variant={variant}, "
            f"got {len(hits)}"
        )
    return hits[0]


def signbit_r100(rows: list[dict[str, str]], direction: str,
                 variant: str) -> float:
    return float(pick_signbit(rows, direction, variant)["r100"])


def load_metrics(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Map (dataset, direction) -> {flip_pct, j100, n25_cos} from the emit CSV."""
    if not path.exists():
        return {}
    out: dict[tuple[str, str], dict[str, str]] = {}
    for r in read_csv_rows(path):
        out[(r["dataset"], r["direction"])] = r
    return out


# --- Build -------------------------------------------------------------------

def build_records(signbit_rows: list[dict[str, str]],
                  metrics: dict[tuple[str, str], dict[str, str]],
                  ) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for dataset, direction, sb_dir, label in ROWS:
        van = signbit_r100(signbit_rows, sb_dir, "vanilla")
        pmc = signbit_r100(signbit_rows, sb_dir, "pmc_a1")
        m = metrics.get((dataset, direction))
        if m is not None:
            flip = f"{float(m['flip_pct']):.2f}"
            j100 = fmt_recall(float(m["j100"]))
            n25 = fmt_recall(float(m["n25_cos"]))
        else:
            flip = j100 = n25 = PENDING
        records.append({
            "dataset": dataset,
            "direction": direction,
            "label": label,
            "bin_van": fmt_recall(van),
            "bin_pmc": fmt_recall(pmc),
            "flip_pct": flip,
            "j100": j100,
            "n25_cos": n25,
        })
    return records


def write_output_csv(records: list[dict[str, str]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


# --- Reporting ---------------------------------------------------------------

def print_console_table(records: list[dict[str, str]], pending: bool) -> None:
    header = (
        f"{'Row':<16} {'binVan':>7} {'binPMC':>7} "
        f"{'Flip%':>7} {'J@100':>7} {'n25cos':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        print(
            f"{r['label']:<16} {r['bin_van']:>7} {r['bin_pmc']:>7} "
            f"{r['flip_pct']:>7} {r['j100']:>7} {r['n25_cos']:>7}"
        )
    if pending:
        print(f"\nNOTE: Flip%/J@100/n=25 cos columns {PENDING_NOTE}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify mechanism metrics backing Sign-Bit Corruption Analysis prose"
    )
    parser.parse_args()

    if not SIGNBIT_CSV.exists():
        raise FileNotFoundError(f"Missing source CSV: {SIGNBIT_CSV}")
    signbit_rows = read_csv_rows(SIGNBIT_CSV)
    metrics = load_metrics(METRICS_CSV)
    records = build_records(signbit_rows, metrics)
    write_output_csv(records)
    pending = len(metrics) == 0
    print_console_table(records, pending)
    print(f"\nWrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
