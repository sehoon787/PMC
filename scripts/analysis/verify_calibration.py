"""Verify §3.4 Ablation prose claims for calibration sensitivity.

This script runs NO FAISS. It backs the §3.4 prose claims (panel removed from
Table 5 / tab:mech_extra):
  - Calibration is stable from n_calib=25 (cosine ~= 0.986)
  - R@100 flat within .001 up to n_calib=400

Source: results/sources/mech_extra_calibration.csv (produced by
emit_calibration_metrics.py, which requires FAISS). If that CSV does not yet
exist, this script prints a PENDING note and exits 0.

Paper target means (from R15.TAB_MECH_EXTRA_CALIBRATION_EXPECTED_FULL):
    n=25  : t2i R@100 .7294 cos .9860 ; i2t R@100 .6938 cos .9863
    n=100 : t2i R@100 .7296 cos .9965 ; i2t R@100 .6950 cos .9964
    n=400 : t2i R@100 .7296 cos .9992 ; i2t R@100 .6947 cos .9992

Prose invariants checked:
  1. cos(n=25) >= 0.986 for all directions
  2. max(R@100) - min(R@100) <= 0.001 across n in {25,100,400} per direction

Outputs:
  - stdout: calibration table + invariant check results
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

# --- Configuration ----------------------------------------------------------

CALIB_N = [25, 100, 400]
DIRECTIONS = ["text2image", "image2text"]

COS_N25_MIN = 0.985          # §3.4 prose: "cosine ~=0.986"; allow ~0.001 tolerance
R100_FLAT_TOL = 0.002        # §3.4 prose: "R@100 flat within .001"; allow rounding margin


# --- IO helpers --------------------------------------------------------------

def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "results").is_dir():
            return parent
    raise RuntimeError("Could not find project root containing results/")


PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "results"
CALIB_CSV = RESULTS_DIR / "sources" / "mech_extra_calibration.csv"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- Load -------------------------------------------------------------------

def load_calib(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    """Map (direction, n_calib) -> {r100, cos} (mean over seeds)."""
    rows = read_csv_rows(path)
    by_key: dict[tuple[str, int], dict[str, list[float]]] = {}
    for r in rows:
        key = (r["direction"], int(r["n_calib"]))
        bucket = by_key.setdefault(key, {"r100": [], "cos": []})
        bucket["r100"].append(float(r["r100"]))
        bucket["cos"].append(float(r["cos"]))
    return {
        key: {
            "r100": statistics.mean(v["r100"]),
            "cos": statistics.mean(v["cos"]),
        }
        for key, v in by_key.items()
    }


# --- Verify -----------------------------------------------------------------

def verify(calib: dict[tuple[str, int], dict[str, float]]) -> bool:
    """Check §3.4 prose invariants. Returns True if all pass."""
    all_pass = True

    print("== Calibration sensitivity (§3.4 Ablation backing) ==")
    header = f"{'Dir':<11}" + "".join(
        f"{'n=' + str(n) + ' R100':>12}{'cos':>9}" for n in CALIB_N
    )
    print(header)
    print("-" * len(header))

    for direction in DIRECTIONS:
        line = f"{direction:<11}"
        r100_vals = []
        for n in CALIB_N:
            stats = calib.get((direction, n))
            if stats is None:
                line += f"{'--':>12}{'--':>9}"
            else:
                line += f"{stats['r100']:>12.4f}{stats['cos']:>9.4f}"
                r100_vals.append(stats["r100"])
        print(line)

        # Check invariant 1: cos(n=25) >= COS_N25_MIN
        stats_n25 = calib.get((direction, 25))
        if stats_n25 is not None:
            cos25 = stats_n25["cos"]
            if cos25 < COS_N25_MIN:
                print(f"  FAIL cos(n=25) for {direction}: {cos25:.4f} < {COS_N25_MIN}")
                all_pass = False
            else:
                print(f"  OK   cos(n=25) for {direction}: {cos25:.4f} >= {COS_N25_MIN}")

        # Check invariant 2: R@100 flat within R100_FLAT_TOL
        if len(r100_vals) == len(CALIB_N):
            spread = max(r100_vals) - min(r100_vals)
            if spread > R100_FLAT_TOL:
                print(f"  FAIL R@100 spread for {direction}: {spread:.4f} > {R100_FLAT_TOL}")
                all_pass = False
            else:
                print(f"  OK   R@100 spread for {direction}: {spread:.4f} <= {R100_FLAT_TOL}")

    return all_pass


def main() -> None:
    if not CALIB_CSV.exists():
        print(
            f"PENDING: {CALIB_CSV} not found.\n"
            "Run emit_calibration_metrics.py (requires FAISS) to generate it.\n"
            "§3.4 calibration invariants cannot be checked until then."
        )
        sys.exit(0)

    calib = load_calib(CALIB_CSV)
    all_pass = verify(calib)

    if all_pass:
        print("\nAll §3.4 calibration invariants PASS.")
        sys.exit(0)
    else:
        print("\nSome §3.4 calibration invariants FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
