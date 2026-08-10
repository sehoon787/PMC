"""Reproduce tab:main (base-regime main results, no rerank).

This script runs NO FAISS or heavy experiment.  For the 6 small-dataset rows
it reads results/nprobe_sweep_pivot.csv (nlist_setting == "sqrtN") at an
nprobe chosen by the matched-budget ρ¼ rule (one per direction):

  nprobe = round_half_up(nlist / 4)

  where nlist is the sqrtN list count for that (dataset, direction) group.
  The result is snapped to the nearest swept nprobe value present in the data;
  ties break toward the larger value.  Because AudioCaps has different nlist
  for each direction (text→audio nlist=26, audio→text nlist=58), nprobe is
  computed separately per direction.

The LAION-400M row is unchanged: read from
  results/pmc_laion400m_nlist80k_seed42.csv   (forward, nprobe=256)
  results/pmc_laion400m_reverse_nlist80k_seed42.csv  (reverse, nprobe=256)

Table 2's reranked R@100 columns are reproduced by reproduce_tab2_rerank.py;
the rerank K'-sweep ablation is in reproduce_ablation_rerank.py.

Outputs:
  - stdout (markdown table)
  - results/tab2_main_reproduced.csv
"""

from __future__ import annotations

import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "results").is_dir():
            return parent
    raise RuntimeError("Could not find project root containing results/")


PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "tab2_main_reproduced.csv"

# Required source files -- checked at startup.
SOURCE_FILES = [
    RESULTS_DIR / "nprobe_sweep_pivot.csv",
    RESULTS_DIR / "pmc_laion400m_nlist80k_seed42.csv",
]

FIELDNAMES = [
    "Dataset", "Enc", "gap",
    "q_r100_van", "q_r100_ms", "q_r100_pmc",
    "q_delta_vp", "q_delta_mp",
    "db_r100_van", "db_r100_ms", "db_r100_pmc",
    "db_delta_vp", "db_delta_mp",
]

GAPS = {
    ("MSCOCO", "CLIP"): ".82",
    ("MSCOCO", "CL-L"): ".82",
    ("MSCOCO", "IB"): ".70",
    ("Flickr30K", "CL-L"): ".77",
    ("Clotho", "IB"): ".61",
    ("AudioCaps", "IB"): ".61",
    ("LAION-400M", "CLIP"): ".72",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ensure_sources_readable() -> None:
    missing = [str(path) for path in SOURCE_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source CSV files:\n" + "\n".join(missing))


def round_half_up(value: float, digits: int) -> float:
    quant = Decimal("1") if digits == 0 else Decimal("1").scaleb(-digits)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def fmt_recall(value: float, digits: int = 2) -> str:
    out = f"{round_half_up(value, digits):.{digits}f}"
    return out[1:] if out.startswith("0") else out


def fmt_delta(vanilla_r100: float, pmc_r100: float, digits: int = 2) -> str:
    """Relative percent delta computed from the DISPLAYED (rounded) recalls.

    Both recalls are first rounded to `digits` decimals (the same precision
    shown in the table), then the percent change is computed end-to-end in
    Decimal with round-half-up, so the delta is consistent with the printed
    recall values and free of float rounding artifacts.
    """
    quant = Decimal("1").scaleb(-digits)
    base = Decimal(str(vanilla_r100)).quantize(quant, rounding=ROUND_HALF_UP)
    pmc = Decimal(str(pmc_r100)).quantize(quant, rounding=ROUND_HALF_UP)
    pct = ((pmc - base) / base * Decimal(100)).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    )
    return f"{int(pct):+d}%"


def pick_one(rows: list[dict[str, str]], **filters: str) -> dict[str, str]:
    picked = [row for row in rows if all(row.get(k) == v for k, v in filters.items())]
    if len(picked) != 1:
        raise ValueError(f"Expected one row for filters={filters}, got {len(picked)}")
    return picked[0]


def rho_quarter_nprobe(
    pivot_rows: list[dict[str, str]],
    dataset: str,
    backbone: str,
    direction: str,
) -> int:
    """Return nprobe for the ρ¼ matched-budget rule (sqrtN rows only).

    Computes rho = round_half_up(nlist / 4), then snaps to the nearest
    swept nprobe present in the data for (dataset, backbone, direction).
    Ties break toward the larger value.
    """
    group = [
        r for r in pivot_rows
        if r["dataset"] == dataset
        and r["backbone"] == backbone
        and r["direction"] == direction
        and r["nlist_setting"] == "sqrtN"
    ]
    if not group:
        raise ValueError(
            f"No sqrtN rows for dataset={dataset!r} backbone={backbone!r} "
            f"direction={direction!r}"
        )
    nlist = int(group[0]["nlist"])
    rho = int(round_half_up(nlist / 4, 0))
    available = sorted(set(int(r["nprobe"]) for r in group))
    return min(available, key=lambda np: (abs(np - rho), -np))


def pick_pivot(
    rows: list[dict[str, str]],
    dataset: str,
    backbone: str,
    direction: str,
    nprobe: int,
) -> dict[str, str]:
    """Return the single sqrtN pivot row matching dataset/backbone/direction/nprobe.

    Raises ValueError if not exactly one row matches.
    """
    matched = [
        r for r in rows
        if r["dataset"] == dataset
        and r["backbone"] == backbone
        and r["direction"] == direction
        and r["nlist_setting"] == "sqrtN"
        and int(r["nprobe"]) == nprobe
    ]
    if len(matched) != 1:
        raise ValueError(
            f"Expected 1 pivot row for dataset={dataset!r} backbone={backbone!r} "
            f"direction={direction!r} nprobe={nprobe} nlist_setting=sqrtN, "
            f"got {len(matched)}"
        )
    return matched[0]


def build_row(
    dataset: str,
    enc: str,
    q_van: tuple[float, float],
    q_pmc: tuple[float, float],
    db_van: tuple[float, float] | None = None,
    db_pmc: tuple[float, float] | None = None,
    q_ms: tuple[float, float] | None = None,
    db_ms: tuple[float, float] | None = None,
    digits: int = 2,
) -> dict[str, str]:
    q_delta_vp = fmt_delta(q_van[1], q_pmc[1], digits=digits)
    q_delta_mp = fmt_delta(q_ms[1], q_pmc[1], digits=digits) if q_ms is not None else ""

    row: dict[str, str] = {
        "Dataset": dataset,
        "Enc": enc,
        "gap": GAPS[(dataset, enc)],
        "q_r100_van": fmt_recall(q_van[1], digits),
        "q_r100_ms": fmt_recall(q_ms[1], digits) if q_ms is not None else "",
        "q_r100_pmc": fmt_recall(q_pmc[1], digits),
        "q_delta_vp": q_delta_vp,
        "q_delta_mp": q_delta_mp,
        "db_r100_van": "",
        "db_r100_ms": "",
        "db_r100_pmc": "",
        "db_delta_vp": "",
        "db_delta_mp": "",
    }
    if db_van is not None and db_pmc is not None:
        db_delta_vp = fmt_delta(db_van[1], db_pmc[1], digits=digits)
        db_delta_mp = fmt_delta(db_ms[1], db_pmc[1], digits=digits) if db_ms is not None else ""
        row["db_r100_van"] = fmt_recall(db_van[1], digits)
        row["db_r100_ms"] = fmt_recall(db_ms[1], digits) if db_ms is not None else ""
        row["db_r100_pmc"] = fmt_recall(db_pmc[1], digits)
        row["db_delta_vp"] = db_delta_vp
        row["db_delta_mp"] = db_delta_mp
    return row


def _pivot_tuple(row: dict[str, str], col: str) -> tuple[float, float]:
    """Return (R10, R100) for a pivot row using the given R100 column prefix."""
    r10_col = col.replace("R100", "R10")
    return (float(row[r10_col]), float(row[col]))


def build_small_row_from_pivot(
    pivot: list[dict[str, str]],
    dataset_label: str,
    enc_label: str,
    pivot_dataset: str,
    pivot_backbone: str,
    q_direction: str,
    db_direction: str,
) -> dict[str, str]:
    """Build one table row for a small dataset using pivot lookups.

    nprobe is computed independently for each direction via the ρ¼ rule,
    so datasets where the two directions have different nlist (e.g. AudioCaps)
    will naturally pick different nprobe values.
    """
    q_nprobe = rho_quarter_nprobe(pivot, pivot_dataset, pivot_backbone, q_direction)
    db_nprobe = rho_quarter_nprobe(pivot, pivot_dataset, pivot_backbone, db_direction)

    q_row = pick_pivot(pivot, pivot_dataset, pivot_backbone, q_direction, q_nprobe)
    db_row = pick_pivot(pivot, pivot_dataset, pivot_backbone, db_direction, db_nprobe)

    return build_row(
        dataset=dataset_label,
        enc=enc_label,
        q_van=_pivot_tuple(q_row, "vanilla_R100"),
        q_pmc=_pivot_tuple(q_row, "pmc_R100"),
        db_van=_pivot_tuple(db_row, "vanilla_R100"),
        db_pmc=_pivot_tuple(db_row, "pmc_R100"),
        q_ms=_pivot_tuple(q_row, "ms_R100"),
        db_ms=_pivot_tuple(db_row, "ms_R100"),
    )


def compute_table_rows(pivot: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    # --- Small datasets (6 rows) from pivot CSV ---

    # MSCOCO -- CLIP-B/32
    rows.append(build_small_row_from_pivot(
        pivot,
        dataset_label="MSCOCO", enc_label="CLIP",
        pivot_dataset="MSCOCO", pivot_backbone="CLIP-B/32",
        q_direction="text->image", db_direction="image->text",
    ))

    # MSCOCO -- CL-L
    rows.append(build_small_row_from_pivot(
        pivot,
        dataset_label="MSCOCO", enc_label="CL-L",
        pivot_dataset="MSCOCO", pivot_backbone="CLIP-L/14",
        q_direction="text->image", db_direction="image->text",
    ))

    # MSCOCO -- IB
    rows.append(build_small_row_from_pivot(
        pivot,
        dataset_label="MSCOCO", enc_label="IB",
        pivot_dataset="MSCOCO", pivot_backbone="ImageBind",
        q_direction="text->image", db_direction="image->text",
    ))

    # Flickr30K -- CL-L
    rows.append(build_small_row_from_pivot(
        pivot,
        dataset_label="Flickr30K", enc_label="CL-L",
        pivot_dataset="Flickr30K-full", pivot_backbone="CLIP-L/14",
        q_direction="text->image", db_direction="image->text",
    ))

    # Clotho -- IB
    rows.append(build_small_row_from_pivot(
        pivot,
        dataset_label="Clotho", enc_label="IB",
        pivot_dataset="Clotho-all", pivot_backbone="ImageBind",
        q_direction="text->audio", db_direction="audio->text",
    ))

    # AudioCaps -- IB
    rows.append(build_small_row_from_pivot(
        pivot,
        dataset_label="AudioCaps", enc_label="IB",
        pivot_dataset="AudioCaps", pivot_backbone="ImageBind",
        q_direction="text->audio", db_direction="audio->text",
    ))

    # --- LAION-400M -- CLIP (unchanged; nprobe=256, 3 decimals) ---
    laion = read_csv_rows(RESULTS_DIR / "pmc_laion400m_nlist80k_seed42.csv")
    laion_van = pick_one(laion, method="vanilla_rabitq", alpha="0.0", nprobe="256")
    laion_pmc = pick_one(laion, method="pmc_1.00", alpha="1.0", nprobe="256")
    laion_ms_row = [
        r for r in laion
        if r.get("method") == "vanilla_rabitq_meanshift" and r.get("nprobe") == "256"
    ]
    laion_ms_q: tuple[float, float] | None = (
        (float(laion_ms_row[0]["r10"]), float(laion_ms_row[0]["r100"]))
        if len(laion_ms_row) == 1 else None
    )

    db_van: tuple[float, float] | None = None
    db_pmc: tuple[float, float] | None = None
    db_ms: tuple[float, float] | None = None
    laion_reverse_path = RESULTS_DIR / "pmc_laion400m_reverse_nlist80k_seed42.csv"
    if laion_reverse_path.exists():
        laion_reverse = read_csv_rows(laion_reverse_path)
        laion_rev_van = pick_one(laion_reverse, method="vanilla_rabitq", alpha="0.0", nprobe="256")
        laion_rev_pmc = pick_one(laion_reverse, method="pmc_1.00", alpha="1.0", nprobe="256")
        db_van = (float(laion_rev_van["r10"]), float(laion_rev_van["r100"]))
        db_pmc = (float(laion_rev_pmc["r10"]), float(laion_rev_pmc["r100"]))
        laion_rev_ms_row = [
            r for r in laion_reverse
            if r.get("method") == "vanilla_rabitq_meanshift" and r.get("nprobe") == "256"
        ]
        db_ms = (
            (float(laion_rev_ms_row[0]["r10"]), float(laion_rev_ms_row[0]["r100"]))
            if len(laion_rev_ms_row) == 1 else None
        )

    rows.append(
        build_row(
            "LAION-400M",
            "CLIP",
            q_van=(float(laion_van["r10"]), float(laion_van["r100"])),
            q_pmc=(float(laion_pmc["r10"]), float(laion_pmc["r100"])),
            q_ms=laion_ms_q,
            db_van=db_van,
            db_pmc=db_pmc,
            db_ms=db_ms,
            digits=3,
        )
    )

    return rows


def write_output_csv(table_rows: list[dict[str, str]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(table_rows)


def print_markdown_table(table_rows: list[dict[str, str]]) -> None:
    headers = FIELDNAMES
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in table_rows:
        print("| " + " | ".join(row.get(h, "") for h in headers) + " |")


def main() -> None:
    ensure_sources_readable()
    pivot = read_csv_rows(RESULTS_DIR / "nprobe_sweep_pivot.csv")
    table_rows = compute_table_rows(pivot)
    write_output_csv(table_rows)
    print_markdown_table(table_rows)
    print(f"\nWrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
