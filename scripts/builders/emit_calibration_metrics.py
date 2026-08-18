"""Emit results/sources/mech_extra_calibration.csv for analysis/verify_calibration.py.

REQUIRES FAISS + feature .npy files. DO NOT run while a large FAISS experiment
owns the machine's RAM. Run it later, once the machine is free.

What it produces (one row per (direction, n_calib, sample_seed)):
    direction, n_calib, sample_seed, r100, cos
where r100 is the PMC(alpha=1, estimated gap from n_calib samples) recall to the
original exact-IP GT, and cos = cos(estimated gap, full gap). The reproduce
script means these over the 5 seeds per (direction, n_calib).

These are exactly the per-row quantities in
mechanism_calibration_sensitivity.csv produced by
scripts/research/R15_mechanism_controls.py (columns r100 and gap_cos_to_full).
This script runs R15 for the mscoco_clip setting (the calibration sub-part is
reported for CLIP/MSCOCO only), reads that CSV, and reshapes it.

Paper target means for later verification
(from R15.TAB_MECH_EXTRA_CALIBRATION_EXPECTED_FULL; displayed n in {25,100,400}):
    n=25  : t2i R@100 .7294 cos .9860 ; i2t R@100 .6938 cos .9863
    n=100 : t2i R@100 .7296 cos .9965 ; i2t R@100 .6950 cos .9964
    n=400 : t2i R@100 .7296 cos .9992 ; i2t R@100 .6947 cos .9992
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

# CLIP/MSCOCO only; pass R15 direction tokens straight through.
DIRECTIONS = ["text2image", "image2text"]
DATASET = "mscoco"
BACKBONE = "clip-b32"

OUTPUT_FIELDNAMES = ["direction", "n_calib", "sample_seed", "r100", "cos"]


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "results").is_dir():
            return parent
    raise RuntimeError("Could not find project root containing results/")


PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "results"
RESEARCH_DIR = RESULTS_DIR / "research"
R15_SCRIPT = PROJECT_ROOT / "scripts" / "research" / "R15_mechanism_controls.py"
CALIB_SRC = RESEARCH_DIR / "diagnostics" / "mechanism_calibration_sensitivity.csv"
OUTPUT_CSV = RESULTS_DIR / "sources" / "mech_extra_calibration.csv"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_r15() -> None:
    """Run R15 for CLIP/MSCOCO to (re)write the calibration CSV. Uses FAISS."""
    cmd = [sys.executable, str(R15_SCRIPT), "--settings", "mscoco_clip"]
    print(f"[emit_calibration_metrics] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def build_records() -> list[dict[str, str]]:
    rows = read_csv_rows(CALIB_SRC)
    records: list[dict[str, str]] = []
    for r in rows:
        if r["dataset"] != DATASET or r["backbone"] != BACKBONE:
            continue
        if r["direction"] not in DIRECTIONS:
            continue
        records.append({
            "direction": r["direction"],
            "n_calib": r["n_calib"],
            "sample_seed": r["sample_seed"],
            "r100": f"{float(r['r100']):.6f}",
            "cos": f"{float(r['gap_cos_to_full']):.6f}",
        })
    return records


def write_output(records: list[dict[str, str]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit mech_extra_calibration.csv (REQUIRES FAISS)."
    )
    parser.add_argument(
        "--skip-run", action="store_true",
        help="Reuse existing calibration CSV instead of re-running R15.",
    )
    args = parser.parse_args()

    if not args.skip_run:
        run_r15()

    records = build_records()
    if not records:
        raise RuntimeError(
            "No records produced. Ensure R15 ran for mscoco_clip and wrote "
            f"{CALIB_SRC}."
        )
    write_output(records)
    print(f"[emit_calibration_metrics] wrote {OUTPUT_CSV} ({len(records)} rows)")


if __name__ == "__main__":
    main()
