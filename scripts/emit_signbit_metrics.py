"""Emit results/mechcontrol_metrics.csv for analysis/verify_signbit_analysis.py.

REQUIRES FAISS + the feature .npy files. DO NOT run while a large FAISS
experiment owns the machine's RAM -- it will OOM. Run it later, once the
machine is free.

What it produces (one row per (dataset, direction)):
    dataset, direction, flip_pct, j100, n25_cos
where
    flip_pct = DB-side global sign-bit flip rate * 100   (mechanism_bitflip.csv,
               vector_type=db, margin_bin=global)
    j100     = top-100 Jaccard of PMC(alpha=1) ranking vs original-IP ranking
               (mechanism_exact_control.csv, alpha=1.0,
                column jaccard100_vs_original_ranking)
    n25_cos  = mean over 5 calibration seeds of cos(estimated gap, full gap)
               at n_calib=25 (mechanism_calibration_sensitivity.csv,
                column gap_cos_to_full)

All three quantities are already computed by
scripts/research/R15_mechanism_controls.py; this script just runs R15 over the
two small settings (mscoco_clip, audiocaps_imagebind), reads the three CSVs it
writes under results/research/, and reshapes them into the flat per-(dataset,
direction) intermediate CSV that verify_signbit_analysis.py consumes.

These numbers back the §3.3 Sign-Bit Corruption Analysis prose claims:
  "PMC induces 14-18% sign-bit flips with J@100 of 0.54-0.80 ...
   n=25 gap-direction cosine >= 0.95"

Paper target values for later verification (from R15.TAB_MECHCONTROL_EXPECTED):
    mscoco/text2image : flip 16.14, J@100 .5852, n25 cos .9860
    mscoco/image2text : flip 16.00, J@100 .5422, n25 cos .9863
    audiocaps/text2audio : flip 14.20, J@100 .8016, n25 cos .9653
    audiocaps/audio2text : flip 17.75, J@100 .8036, n25 cos .9546

TODO (uncertain until run): R15 currently hardcodes audiocaps n25 cos
.9653/.9546 in TAB_MECHCONTROL_EXPECTED but those are NOT validated in its
_validate_tab_mechcontrol (only mscoco cos is checked there). Confirm the
audiocaps cos values once the calibration CSV is produced; if R15's emitted
value differs, treat R15's hardcoded expectation as the paper target and flag.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path

# (intermediate dataset key, R15 direction label) -> output direction label.
# R15 writes datasets as "mscoco"/"audiocaps" and directions as the 2img/2audio
# tokens; the reproduce script keys on (dataset, direction) with those same
# direction tokens, so we pass them through unchanged.
ROWS = [
    ("mscoco", "text2image"),
    ("mscoco", "image2text"),
    ("audiocaps", "text2audio"),
    ("audiocaps", "audio2text"),
]

N_CALIB_FOR_COS = 25
OUTPUT_FIELDNAMES = ["dataset", "direction", "flip_pct", "j100", "n25_cos"]


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "results").is_dir():
            return parent
    raise RuntimeError("Could not find project root containing results/")


PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "results"
RESEARCH_DIR = RESULTS_DIR / "research"
R15_SCRIPT = PROJECT_ROOT / "scripts" / "research" / "R15_mechanism_controls.py"
OUTPUT_CSV = RESULTS_DIR / "mechcontrol_metrics.csv"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_r15() -> None:
    """Run R15 over both small settings to (re)write the research CSVs.

    NOTE: this invokes FAISS. Only call when the machine is free.
    """
    cmd = [
        sys.executable, str(R15_SCRIPT),
        "--settings", "mscoco_clip", "audiocaps_imagebind",
    ]
    print(f"[emit_signbit_metrics] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def collect_flip_pct(bitflip_rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    return {
        (r["dataset"], r["direction"]): float(r["flip_rate"]) * 100.0
        for r in bitflip_rows
        if r["vector_type"] == "db" and r["margin_bin"] == "global"
    }


def collect_j100(exact_rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    return {
        (r["dataset"], r["direction"]): float(r["jaccard100_vs_original_ranking"])
        for r in exact_rows
        if abs(float(r["alpha"]) - 1.0) < 1e-12
    }


def collect_n25_cos(calib_rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    by_key: dict[tuple[str, str], list[float]] = {}
    for r in calib_rows:
        if int(r["n_calib"]) != N_CALIB_FOR_COS:
            continue
        by_key.setdefault((r["dataset"], r["direction"]), []).append(
            float(r["gap_cos_to_full"])
        )
    return {k: statistics.mean(v) for k, v in by_key.items() if v}


def build_records() -> list[dict[str, str]]:
    bitflip = collect_flip_pct(read_csv_rows(RESEARCH_DIR / "mechanism_bitflip.csv"))
    j100 = collect_j100(read_csv_rows(RESEARCH_DIR / "mechanism_exact_control.csv"))
    cos = collect_n25_cos(read_csv_rows(RESEARCH_DIR / "mechanism_calibration_sensitivity.csv"))

    records: list[dict[str, str]] = []
    for dataset, direction in ROWS:
        key = (dataset, direction)
        if key not in bitflip or key not in j100 or key not in cos:
            print(f"[emit_signbit_metrics][warn] missing metric for {key}; "
                  f"skipping", file=sys.stderr)
            continue
        records.append({
            "dataset": dataset,
            "direction": direction,
            "flip_pct": f"{bitflip[key]:.2f}",
            "j100": f"{j100[key]:.4f}",
            "n25_cos": f"{cos[key]:.4f}",
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
        description="Emit mechcontrol_metrics.csv (REQUIRES FAISS)."
    )
    parser.add_argument(
        "--skip-run", action="store_true",
        help="Reuse existing results/research CSVs instead of re-running R15.",
    )
    args = parser.parse_args()

    if not args.skip_run:
        run_r15()

    records = build_records()
    if not records:
        raise RuntimeError(
            "No records produced. Ensure R15 ran for both mscoco_clip and "
            "audiocaps_imagebind settings (check results/research/)."
        )
    write_output(records)
    print(f"[emit_signbit_metrics] wrote {OUTPUT_CSV} ({len(records)} rows)")


if __name__ == "__main__":
    main()
