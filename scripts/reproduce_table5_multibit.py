"""Lightweight Table 5 (tab:multibit) reproduction/validation aggregator.

This script does not run FAISS or any heavy experiment.
It reads existing PQ CSV artifacts from results/ and reproduces the
manuscript Table 5 (Multi-Bit Generality) values, then emits:
  - stdout (markdown table matching paper layout)
  - results/table3_multibit_reproduced.csv

Source CSVs are produced by reproduce_table3_pq_sweep.py (MSCOCO CLIP sweep)
and the scripts in current/pmc_crossmodal/scripts/pq/:
  - pmc_pq_alpha_sweep_clip_mscoco_seed42.csv  (COCO / CLIP-B32)
  - pmc_pq_clipl_mscoco_seed42.csv             (COCO / CLIP-L)
  - pmc_pq_imagebind_mscoco_seed42.csv         (COCO / ImageBind)
  - pmc_pq_flickr30k_clipl_seed42.csv          (Flickr30K / CLIP-L)
"""

from __future__ import annotations

import csv
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# Windows consoles default to cp949 (Korean locale); force UTF-8 so the
# arrow/delta/em-dash characters in the table render without UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "results").is_dir():
            return parent
    raise RuntimeError("Could not find project root containing results/")


PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "table3_multibit_reproduced.csv"

# Source CSVs keyed by logical dataset/encoder group
SOURCE_FILES = {
    "coco_clip": RESULTS_DIR / "pmc_pq_alpha_sweep_clip_mscoco_seed42.csv",
    "coco_clipl": RESULTS_DIR / "pmc_pq_clipl_mscoco_seed42.csv",
    "coco_ib": RESULTS_DIR / "pmc_pq_imagebind_mscoco_seed42.csv",
    "flickr_clipl": RESULTS_DIR / "pmc_pq_flickr30k_clipl_seed42.csv",
}

FIELDNAMES = [
    "Dataset", "Enc", "Dir",
    "IVFPQ_van", "IVFPQ_pmc", "IVFPQ_delta",
    "OPQ_van", "OPQ_pmc", "OPQ_delta",
]

# Paper values for validation (from tab:multibit in 03_experiments.tex)
PAPER_VALUES = {
    ("COCO", "CLIP", "t→i"): {"IVFPQ_van": .54, "IVFPQ_pmc": .63, "OPQ_van": .61, "OPQ_pmc": .66},
    ("COCO", "CLIP", "i→t"): {"IVFPQ_van": .50, "IVFPQ_pmc": .64, "OPQ_van": .67, "OPQ_pmc": .67},
    ("COCO", "CL-L", "t→i"): {"IVFPQ_van": .43, "IVFPQ_pmc": .63, "OPQ_van": .56, "OPQ_pmc": .67},
    ("COCO", "CL-L", "i→t"): {"IVFPQ_van": .35, "IVFPQ_pmc": .58, "OPQ_van": .60, "OPQ_pmc": .67},
    ("COCO", "IB",   "t→i"): {"IVFPQ_van": .59, "IVFPQ_pmc": .70, "OPQ_van": .69, "OPQ_pmc": .76},
    ("COCO", "IB",   "i→t"): {"IVFPQ_van": .59, "IVFPQ_pmc": .68, "OPQ_van": .72, "OPQ_pmc": .78},
    ("Flickr", "CL-L", "t→i"): {"IVFPQ_van": .51, "IVFPQ_pmc": .54, "OPQ_van": .55, "OPQ_pmc": .54},
    ("Flickr", "CL-L", "i→t"): {"IVFPQ_van": .43, "IVFPQ_pmc": .47, "OPQ_van": .48, "OPQ_pmc": .50},
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ensure_sources_readable() -> dict[str, list[dict[str, str]]]:
    missing = [f"{key}: {path}" for key, path in SOURCE_FILES.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source CSV files:\n" + "\n".join(missing))
    return {key: read_csv_rows(path) for key, path in SOURCE_FILES.items()}


def round_half_up(value: float, digits: int) -> float:
    quant = Decimal("1") if digits == 0 else Decimal("1").scaleb(-digits)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def fmt_r100(value: float) -> str:
    """Format R@100 as '.XX' dropping leading zero."""
    out = f"{round_half_up(value, 2):.2f}"
    return out[1:] if out.startswith("0") else out


def fmt_delta(vanilla: float, pmc: float) -> str:
    pct = int(round_half_up((pmc - vanilla) / vanilla * 100, 0))
    return f"{pct:+d}%"


def pick_r100(
    rows: list[dict[str, str]],
    method_col: str,
    method_val: str,
    direction: str,
    alpha: str | None = None,
) -> float:
    """Return r100 for the given method and direction (optionally filtering by alpha)."""
    picked = [
        row for row in rows
        if row[method_col] == method_val and row["direction"] == direction
        and (alpha is None or row.get("alpha") == alpha)
    ]
    if len(picked) != 1:
        raise ValueError(
            f"Expected 1 row for method={method_val!r} direction={direction!r} alpha={alpha!r}, "
            f"got {len(picked)}"
        )
    return float(picked[0]["r100"])


def build_table_rows(data: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    # COCO / CLIP-B32 — from alpha_sweep CSV (alpha=0.0 is vanilla, alpha=1.0 is PMC)
    coco_clip = data["coco_clip"]
    for direction, dir_label in [("text->image", "t→i"), ("image->text", "i→t")]:
        ivf_van = pick_r100(coco_clip, "method", "ivfpq", direction, alpha="0.0")
        ivf_pmc = pick_r100(coco_clip, "method", "ivfpq", direction, alpha="1.0")
        opq_van = pick_r100(coco_clip, "method", "opq_ivfpq", direction, alpha="0.0")
        opq_pmc = pick_r100(coco_clip, "method", "opq_ivfpq", direction, alpha="1.0")
        rows.append({
            "Dataset": "COCO", "Enc": "CLIP", "Dir": dir_label,
            "IVFPQ_van": fmt_r100(ivf_van), "IVFPQ_pmc": fmt_r100(ivf_pmc),
            "IVFPQ_delta": fmt_delta(ivf_van, ivf_pmc),
            "OPQ_van": fmt_r100(opq_van), "OPQ_pmc": fmt_r100(opq_pmc),
            "OPQ_delta": fmt_delta(opq_van, opq_pmc),
        })

    # COCO / CLIP-L — method column named "method", vanilla=ivfpq/opq_ivfpq, pmc=ivfpq_pmc/opq_ivfpq_pmc
    coco_clipl = data["coco_clipl"]
    for direction, dir_label in [("text->image", "t→i"), ("image->text", "i→t")]:
        ivf_van = pick_r100(coco_clipl, "method", "ivfpq", direction)
        ivf_pmc = pick_r100(coco_clipl, "method", "ivfpq_pmc", direction)
        opq_van = pick_r100(coco_clipl, "method", "opq_ivfpq", direction)
        opq_pmc = pick_r100(coco_clipl, "method", "opq_ivfpq_pmc", direction)
        rows.append({
            "Dataset": "COCO", "Enc": "CL-L", "Dir": dir_label,
            "IVFPQ_van": fmt_r100(ivf_van), "IVFPQ_pmc": fmt_r100(ivf_pmc),
            "IVFPQ_delta": fmt_delta(ivf_van, ivf_pmc),
            "OPQ_van": fmt_r100(opq_van), "OPQ_pmc": fmt_r100(opq_pmc),
            "OPQ_delta": fmt_delta(opq_van, opq_pmc),
        })

    # COCO / ImageBind
    coco_ib = data["coco_ib"]
    for direction, dir_label in [("text->image", "t→i"), ("image->text", "i→t")]:
        ivf_van = pick_r100(coco_ib, "method", "ivfpq", direction)
        ivf_pmc = pick_r100(coco_ib, "method", "ivfpq_pmc", direction)
        opq_van = pick_r100(coco_ib, "method", "opq_ivfpq", direction)
        opq_pmc = pick_r100(coco_ib, "method", "opq_ivfpq_pmc", direction)
        rows.append({
            "Dataset": "COCO", "Enc": "IB", "Dir": dir_label,
            "IVFPQ_van": fmt_r100(ivf_van), "IVFPQ_pmc": fmt_r100(ivf_pmc),
            "IVFPQ_delta": fmt_delta(ivf_van, ivf_pmc),
            "OPQ_van": fmt_r100(opq_van), "OPQ_pmc": fmt_r100(opq_pmc),
            "OPQ_delta": fmt_delta(opq_van, opq_pmc),
        })

    # Flickr30K / CLIP-L
    flickr = data["flickr_clipl"]
    for direction, dir_label in [("text->image", "t→i"), ("image->text", "i→t")]:
        ivf_van = pick_r100(flickr, "method", "ivfpq", direction)
        ivf_pmc = pick_r100(flickr, "method", "ivfpq_pmc", direction)
        opq_van = pick_r100(flickr, "method", "opq_ivfpq", direction)
        opq_pmc = pick_r100(flickr, "method", "opq_ivfpq_pmc", direction)
        rows.append({
            "Dataset": "Flickr", "Enc": "CL-L", "Dir": dir_label,
            "IVFPQ_van": fmt_r100(ivf_van), "IVFPQ_pmc": fmt_r100(ivf_pmc),
            "IVFPQ_delta": fmt_delta(ivf_van, ivf_pmc),
            "OPQ_van": fmt_r100(opq_van), "OPQ_pmc": fmt_r100(opq_pmc),
            "OPQ_delta": fmt_delta(opq_van, opq_pmc),
        })

    return rows


def validate_against_paper(rows: list[dict[str, str]]) -> int:
    """Check reproduced values match paper Table 5. Returns count of mismatches."""
    mismatches = 0
    for row in rows:
        key = (row["Dataset"], row["Enc"], row["Dir"])
        paper = PAPER_VALUES.get(key)
        if paper is None:
            continue
        for col, paper_val in paper.items():
            reproduced = float("0" + row[col] if row[col].startswith(".") else row[col])
            if abs(reproduced - paper_val) > 0.005:
                print(f"  MISMATCH {key} {col}: paper={paper_val:.2f} reproduced={reproduced:.2f}")
                mismatches += 1
    return mismatches


def write_output_csv(table_rows: list[dict[str, str]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(table_rows)


def print_markdown_table(table_rows: list[dict[str, str]]) -> None:
    header = (
        "| Dataset | Enc | Dir | IVFPQ Van | IVFPQ PMC | IVFPQ Δ"
        " | OPQ Van | OPQ PMC | OPQ Δ |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for row in table_rows:
        print(
            f"| {row['Dataset']} | {row['Enc']} | {row['Dir']}"
            f" | {row['IVFPQ_van']} | {row['IVFPQ_pmc']} | {row['IVFPQ_delta']}"
            f" | {row['OPQ_van']} | {row['OPQ_pmc']} | {row['OPQ_delta']} |"
        )


def main() -> None:
    source_data = ensure_sources_readable()
    table_rows = build_table_rows(source_data)
    write_output_csv(table_rows)
    print_markdown_table(table_rows)

    print(f"\nValidating against paper Table 5 values ...")
    n_mismatches = validate_against_paper(table_rows)
    if n_mismatches == 0:
        print("  All values match paper (within ±0.005).")
    else:
        print(f"  {n_mismatches} mismatch(es) found — check source CSVs.")

    print(f"\nWrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
