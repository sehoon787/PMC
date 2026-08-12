"""Reproduce Table 5 (tab:multibit): multi-bit quantization (IVFPQ / OPQ) results.

This script runs NO FAISS.

Each cell reports vanilla R@100 / PMC R@100 + relative-R@100 delta% at the
multi-bit (IVFPQ / OPQ) operating point nprobe=16, rerank_k=0.

Source map (per the authoritative analyst pass):
  - All cells come from results/rerank_multibit_seed42.csv, EXCEPT
  - the OPQ cell for CLIP/MSCOCO, which comes from the 3-seed mean in
    results/pmc_opq_multiseed_clip_mscoco.csv (the rerank_multibit single-seed
    OPQ point disagrees with the paper there; the multiseed mean reproduces it).

Known reproduce notes (cells where the source disagrees with the paper number)
are tagged inline with `# REPRODUCE-MISMATCH:` and surfaced at runtime.

Outputs:
  - stdout: human-readable table + LaTeX rows (per-direction + direction-averaged)
  - results/tab6_multibit_reproduced__np16_k0.csv
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# --- Configuration ----------------------------------------------------------

DEFAULT_NPROBE = 16
DEFAULT_RERANK_K = 0

# (backbone, dataset, display_enc) row order, matching tab:multibit.
ROWS = [
    ("clip", "mscoco", "CLIP"),
    ("clip-l", "mscoco", "CL-L"),
    ("imagebind", "mscoco", "IB"),
    ("clip-l", "flickr30k", "CL-L"),
]

DATASET_LABELS = {"mscoco": "MSCOCO", "flickr30k": "Flickr"}

# Quantizer families and the (vanilla, pmc) method names in rerank_multibit CSV.
FAMILIES = {
    "IVFPQ": ("ivfpq_vanilla", "ivfpq_pmc_1.00"),
    "OPQ": ("opq_vanilla", "opq_pmc_1.00"),
}

DIRECTIONS = [("t->i", "text->image"), ("i->t", "image->text")]

# Paper values for verification: (van_r100, pmc_r100, delta_pct) per
# (enc, dataset, family, dir_label).
#
# REPRODUCE-MISMATCH (source CSV vs paper, all <= 0.01 / 1 delta point; the
# paper used a slightly different seed/protocol for these cells -- flagged at
# runtime, never silently overwritten):
#   CLIP/MSCOCO/IVFPQ/i->t : van .4933->.49  (paper .50);  delta still +30%
#   CL-L/MSCOCO/IVFPQ/t->i : delta +45% (paper +46%); van .43 pmc .63 match
#   CL-L/MSCOCO/OPQ/i->t   : van .5854->.59 (paper .60); delta +15% (paper +12%)
#   IB/MSCOCO/OPQ/t->i     : van .6843->.68 (paper .69); pmc .76 match
#   CL-L/Flickr/IVFPQ/t->i : van .5011->.50 (paper .51); pmc .5319->.53 (.54); +6% (+5%)
#   CL-L/Flickr/OPQ/i->t   : delta +4% (paper +5%); van .48 pmc .50 match
PAPER = {
    ("CLIP", "mscoco", "IVFPQ", "t->i"): (0.54, 0.63, 16),
    ("CLIP", "mscoco", "OPQ", "t->i"): (0.61, 0.66, 8),
    ("CLIP", "mscoco", "IVFPQ", "i->t"): (0.50, 0.64, 30),
    ("CLIP", "mscoco", "OPQ", "i->t"): (0.67, 0.67, -1),
    ("CL-L", "mscoco", "IVFPQ", "t->i"): (0.43, 0.63, 46),
    ("CL-L", "mscoco", "OPQ", "t->i"): (0.56, 0.67, 20),
    ("CL-L", "mscoco", "IVFPQ", "i->t"): (0.35, 0.58, 69),
    ("CL-L", "mscoco", "OPQ", "i->t"): (0.60, 0.67, 12),
    ("IB", "mscoco", "IVFPQ", "t->i"): (0.59, 0.70, 18),
    ("IB", "mscoco", "OPQ", "t->i"): (0.69, 0.76, 11),
    ("IB", "mscoco", "IVFPQ", "i->t"): (0.59, 0.68, 16),
    ("IB", "mscoco", "OPQ", "i->t"): (0.72, 0.78, 7),
    ("CL-L", "flickr30k", "IVFPQ", "t->i"): (0.51, 0.54, 5),
    ("CL-L", "flickr30k", "OPQ", "t->i"): (0.55, 0.54, -1),
    ("CL-L", "flickr30k", "IVFPQ", "i->t"): (0.43, 0.47, 9),
    ("CL-L", "flickr30k", "OPQ", "i->t"): (0.48, 0.50, 5),
}

FIELDNAMES = [
    "enc", "dataset", "family", "direction", "nprobe", "rerank_k",
    "source", "r100_van", "r100_pmc", "delta", "mismatch",
]


# --- IO helpers --------------------------------------------------------------

def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "results").is_dir():
            return parent
    raise RuntimeError("Could not find project root containing results/")


PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "results"
MULTIBIT_CSV = RESULTS_DIR / "rerank_multibit_seed42.csv"
OPQ_MULTISEED_CSV = RESULTS_DIR / "pmc_opq_multiseed_clip_mscoco.csv"


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


def delta_pct_int(base_r100: float, pmc_r100: float) -> int:
    return int(round_half_up((pmc_r100 - base_r100) / base_r100 * 100, 0))


# --- Lookups -----------------------------------------------------------------

def multibit_r100(rows: list[dict[str, str]], backbone: str, dataset: str,
                  method: str, direction: str, nprobe: int,
                  rerank_k: int) -> float:
    hits = [
        r for r in rows
        if r["backbone"] == backbone and r["dataset"] == dataset
        and r["method"] == method and r["direction"] == direction
        and int(r["nprobe"]) == nprobe and int(r["rerank_k"]) == rerank_k
    ]
    if len(hits) != 1:
        raise ValueError(
            f"Expected one multibit row for backbone={backbone} dataset={dataset} "
            f"method={method} dir={direction} np={nprobe} k={rerank_k}, "
            f"got {len(hits)}"
        )
    return float(hits[0]["r100"])


def opq_multiseed_mean_r100(rows: list[dict[str, str]], method: str,
                            direction: str, nprobe: int) -> float:
    """3-seed mean R@100 for CLIP/MSCOCO OPQ from the multiseed CSV.

    method is 'opq' (vanilla) or 'opq_pmc'; direction uses 'text->image' /
    'image->text' as in the multiseed CSV.
    """
    vals = [
        float(r["r100"])
        for r in rows
        if r["method"] == method and r["direction"] == direction
        and int(r["nprobe"]) == nprobe
    ]
    if not vals:
        raise ValueError(
            f"No OPQ multiseed rows for method={method} dir={direction} np={nprobe}"
        )
    return sum(vals) / len(vals)


# --- Build -------------------------------------------------------------------

def cell_values(multibit_rows: list[dict[str, str]],
                opq_rows: list[dict[str, str]],
                enc: str, backbone: str, dataset: str, family: str,
                direction: str, nprobe: int, rerank_k: int,
                ) -> tuple[float, float, str]:
    """Return (van_r100, pmc_r100, source_label) for one cell."""
    # Special-case: CLIP/MSCOCO OPQ comes from the multiseed mean.
    if enc == "CLIP" and dataset == "mscoco" and family == "OPQ":
        van = opq_multiseed_mean_r100(opq_rows, "opq", direction, nprobe)
        pmc = opq_multiseed_mean_r100(opq_rows, "opq_pmc", direction, nprobe)
        return van, pmc, "opq_multiseed_mean"
    van_method, pmc_method = FAMILIES[family]
    van = multibit_r100(multibit_rows, backbone, dataset, van_method, direction,
                        nprobe, rerank_k)
    pmc = multibit_r100(multibit_rows, backbone, dataset, pmc_method, direction,
                        nprobe, rerank_k)
    return van, pmc, "rerank_multibit"


def build_records(multibit_rows: list[dict[str, str]],
                  opq_rows: list[dict[str, str]],
                  nprobe: int, rerank_k: int) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for backbone, dataset, enc in ROWS:
        for family in ("IVFPQ", "OPQ"):
            for dir_label, direction in DIRECTIONS:
                van, pmc, source = cell_values(
                    multibit_rows, opq_rows, enc, backbone, dataset, family,
                    direction, nprobe, rerank_k,
                )
                van_s = fmt_recall(van)
                pmc_s = fmt_recall(pmc)
                delta = fmt_delta(van, pmc)
                mismatch = check_mismatch(
                    enc, dataset, family, dir_label, van, pmc,
                    nprobe, rerank_k,
                )
                records.append({
                    "enc": enc,
                    "dataset": DATASET_LABELS[dataset],
                    "family": family,
                    "direction": dir_label,
                    "nprobe": str(nprobe),
                    "rerank_k": str(rerank_k),
                    "source": source,
                    "r100_van": van_s,
                    "r100_pmc": pmc_s,
                    "delta": delta,
                    "mismatch": mismatch,
                    # Raw full-precision floats for averaged-LaTeX emitter only;
                    # not included in FIELDNAMES so they are not written to CSV.
                    "_r100_van_raw": van,
                    "_r100_pmc_raw": pmc,
                })
    return records


def check_mismatch(enc: str, dataset: str, family: str, dir_label: str,
                   van: float, pmc: float, nprobe: int, rerank_k: int) -> str:
    """At the paper operating point (np=16,k=0), compare to PAPER table.

    Returns a human-readable mismatch description, or "" when the cell matches.
    Only meaningful at the paper operating point; for other CLI options we
    cannot expect a match, so the check is skipped.
    """
    if nprobe != DEFAULT_NPROBE or rerank_k != DEFAULT_RERANK_K:
        return ""
    key = (enc, dataset, family, dir_label)
    expected = PAPER.get(key)
    if expected is None:
        return ""
    exp_van, exp_pmc, exp_delta = expected
    got_van = round_half_up(van, 2)
    got_pmc = round_half_up(pmc, 2)
    got_delta = delta_pct_int(van, pmc)
    issues = []
    if got_van != exp_van:
        issues.append(f"van {got_van:.2f} vs paper {exp_van:.2f}")
    if got_pmc != exp_pmc:
        issues.append(f"pmc {got_pmc:.2f} vs paper {exp_pmc:.2f}")
    if got_delta != exp_delta:
        issues.append(f"delta {got_delta:+d}% vs paper {exp_delta:+d}%")
    return "; ".join(issues)


def output_csv_path(nprobe: int, rerank_k: int) -> Path:
    return RESULTS_DIR / f"tab6_multibit_reproduced__np{nprobe}_k{rerank_k}.csv"


def write_output_csv(records: list[dict[str, str]], path: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows({k: r[k] for k in FIELDNAMES} for r in records)


# --- Reporting ---------------------------------------------------------------

def print_console_table(records: list[dict[str, str]]) -> None:
    header = (
        f"{'Enc':<5} {'Dataset':<8} {'Family':<6} {'Dir':<5} "
        f"{'van':>5} {'pmc':>5} {'delta':>6}  {'source':<18} mismatch"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        print(
            f"{r['enc']:<5} {r['dataset']:<8} {r['family']:<6} {r['direction']:<5} "
            f"{r['r100_van']:>5} {r['r100_pmc']:>5} {r['delta']:>6}  "
            f"{r['source']:<18} {r['mismatch']}"
        )


def print_latex_rows(records: list[dict[str, str]]) -> None:
    """Emit the 4 direction-averaged rows matching current paper Table 5 (tab:multibit).

    Averages the two per-direction values (t->i and i->t) already in records.
    Delta% is computed from the averaged van/pmc values, not averaged per-direction
    deltas. Where the computed average rounds differently from the paper cell, an
    AVG-LATEX-NOTE comment is printed so the discrepancy is visible.
    """
    # Paper reference for averaged cells (van, pmc, delta_pct).
    PAPER_AVG = {
        ("CLIP",  "MSCOCO",  "IVFPQ"): (0.52, 0.64, 23),
        ("CLIP",  "MSCOCO",  "OPQ"):   (0.64, 0.67,  4),
        ("CL-L",  "MSCOCO",  "IVFPQ"): (0.39, 0.61, 56),
        ("CL-L",  "MSCOCO",  "OPQ"):   (0.57, 0.67, 17),
        ("IB",    "MSCOCO",  "IVFPQ"): (0.59, 0.69, 17),
        ("IB",    "MSCOCO",  "OPQ"):   (0.70, 0.77,  9),
        ("CL-L",  "Flickr",  "IVFPQ"): (0.47, 0.50,  7),
        ("CL-L",  "Flickr",  "OPQ"):   (0.51, 0.52,  1),
    }

    # Group per-direction records by (enc, dataset, family).
    by_edf: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for r in records:
        key = (r["enc"], r["dataset"], r["family"])
        by_edf.setdefault(key, []).append(r)

    print("\n% --- auto-generated by reproduce_tab6_multibit.py (direction-averaged) ---")
    print("% Table 5 (tab:multibit): 4 rows, averaged over t->i and i->t")
    print("% columns: Dataset & Enc & IVFPQ van/PMC +D% & OPQ van/PMC +D%")

    prev_dataset = None
    for backbone, dataset, enc in ROWS:
        ds_label = DATASET_LABELS[dataset]
        if prev_dataset is not None and ds_label != prev_dataset:
            print("\\midrule")
        prev_dataset = ds_label

        ivf_recs = by_edf.get((enc, ds_label, "IVFPQ"), [])
        opq_recs = by_edf.get((enc, ds_label, "OPQ"), [])

        def avg_cell(recs: list[dict[str, str]], fam: str) -> tuple[str, str, str]:
            if len(recs) != 2:
                raise ValueError(
                    f"Expected 2 direction records for {enc}/{ds_label}/{fam}, "
                    f"got {len(recs)}"
                )
            avg_van = (recs[0]["_r100_van_raw"] + recs[1]["_r100_van_raw"]) / 2
            avg_pmc = (recs[0]["_r100_pmc_raw"] + recs[1]["_r100_pmc_raw"]) / 2
            avg_delta = delta_pct_int(avg_van, avg_pmc)
            van_s = fmt_recall(avg_van)
            pmc_s = fmt_recall(avg_pmc)
            delta_s = f"{avg_delta:+d}\\%"
            # Check against paper.
            pa = PAPER_AVG.get((enc, ds_label, fam))
            if pa is not None:
                p_van, p_pmc, p_delta = pa
                notes = []
                if round_half_up(avg_van, 2) != p_van:
                    notes.append(f"van computed {round_half_up(avg_van,2):.2f} vs paper {p_van:.2f}")
                if round_half_up(avg_pmc, 2) != p_pmc:
                    notes.append(f"pmc computed {round_half_up(avg_pmc,2):.2f} vs paper {p_pmc:.2f}")
                if avg_delta != p_delta:
                    notes.append(f"delta computed {avg_delta:+d}% vs paper {p_delta:+d}%")
                if notes:
                    print(f"# AVG-LATEX-NOTE {enc}/{ds_label}/{fam}: " + "; ".join(notes))
            return van_s, pmc_s, delta_s

        ivf_van, ivf_pmc, ivf_d = avg_cell(ivf_recs, "IVFPQ")
        opq_van, opq_pmc, opq_d = avg_cell(opq_recs, "OPQ")
        print(
            f"{ds_label:<8} & {enc:<5} & "
            f"{ivf_van}\\,/\\,\\textbf{{{ivf_pmc}}}\\,{ivf_d} & "
            f"{opq_van}\\,/\\,\\textbf{{{opq_pmc}}}\\,{opq_d} \\\\"
        )


def report_mismatches(records: list[dict[str, str]]) -> None:
    mism = [r for r in records if r["mismatch"]]
    if not mism:
        print("\nAll cells reproduce the paper values cell-exact at np=16, k=0.")
        return
    print("\nREPRODUCE-MISMATCH cells (actual vs paper):")
    for r in mism:
        print(f"  {r['enc']}/{r['dataset']}/{r['family']}/{r['direction']}: "
              f"{r['mismatch']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce tab:multibit")
    parser.add_argument("--nprobe", type=int, default=DEFAULT_NPROBE)
    parser.add_argument("--rerank-k", type=int, default=DEFAULT_RERANK_K)
    args = parser.parse_args()

    if not MULTIBIT_CSV.exists():
        raise FileNotFoundError(f"Missing source CSV: {MULTIBIT_CSV}")
    if not OPQ_MULTISEED_CSV.exists():
        raise FileNotFoundError(f"Missing source CSV: {OPQ_MULTISEED_CSV}")

    multibit_rows = read_csv_rows(MULTIBIT_CSV)
    opq_rows = read_csv_rows(OPQ_MULTISEED_CSV)
    records = build_records(multibit_rows, opq_rows, args.nprobe, args.rerank_k)

    out_path = output_csv_path(args.nprobe, args.rerank_k)
    write_output_csv(records, out_path)
    print_console_table(records)
    print_latex_rows(records)
    report_mismatches(records)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
