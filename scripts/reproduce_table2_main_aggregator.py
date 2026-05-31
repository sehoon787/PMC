"""Lightweight Table 2 (main results) reproduction/validation aggregator.

This script does not run FAISS or any heavy experiment.
It reads existing CSV artifacts, reproduces the manuscript Table 2 values,
then emits:
  - stdout (markdown table)
  - results/table2_main_reproduced.csv
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
OUTPUT_CSV = RESULTS_DIR / "table2_main_reproduced.csv"

SOURCE_FILES = [
    RESULTS_DIR / "multiseed_rabitq_summary.csv",
    RESULTS_DIR / "pmc_eval_clip-l_flickr30k_full_seed42.csv",
    RESULTS_DIR / "pmc_clotho_r1_seed42.csv",
    RESULTS_DIR / "pmc_laion400m_seed42.csv",
]

# Meanshift source files — loaded separately; missing files are silently skipped.
MS_SOURCE_FILES = [
    RESULTS_DIR / "pmc_qps_pareto_clip_mscoco_seed42.csv",        # CLIP-B/32 MSCOCO
    RESULTS_DIR / "pmc_qps_pareto_imagebind_mscoco_seed42.csv",   # IB MSCOCO
    RESULTS_DIR / "pmc_eval_clip-l_flickr30k_full_seed42.csv",    # CL-L Flickr
    RESULTS_DIR / "pmc_clotho_r1_seed42.csv",                     # Clotho
    RESULTS_DIR / "pmc_laion400m_seed42.csv",                     # LAION forward
    RESULTS_DIR / "pmc_laion400m_reverse_seed42.csv",             # LAION reverse (db-side)
]

FIELDNAMES = [
    "Dataset", "Enc", "gap",
    "q_r10_van", "q_r10_ms", "q_r10_pmc",
    "q_r100_van", "q_r100_ms", "q_r100_pmc",
    "q_delta_vp", "q_delta_mp",
    "db_r10_van", "db_r10_ms", "db_r10_pmc",
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


def ensure_sources_readable() -> dict[str, list[dict[str, str]]]:
    missing = [str(path) for path in SOURCE_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source CSV files:\n" + "\n".join(missing))
    return {path.name: read_csv_rows(path) for path in SOURCE_FILES}


def load_ms_sources() -> dict[str, list[dict[str, str]]]:
    """Load meanshift source files; silently skip missing ones."""
    result: dict[str, list[dict[str, str]]] = {}
    for path in MS_SOURCE_FILES:
        if path.exists():
            result[path.name] = read_csv_rows(path)
    return result


def round_half_up(value: float, digits: int) -> float:
    quant = Decimal("1") if digits == 0 else Decimal("1").scaleb(-digits)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def fmt_recall(value: float, digits: int = 2) -> str:
    out = f"{round_half_up(value, digits):.{digits}f}"
    return out[1:] if out.startswith("0") else out


def fmt_delta(vanilla_r100: float, pmc_r100: float) -> str:
    pct = int(round_half_up((pmc_r100 - vanilla_r100) / vanilla_r100 * 100, 0))
    return f"{pct:+d}%"


def pick_one(rows: list[dict[str, str]], **filters: str) -> dict[str, str]:
    picked = [row for row in rows if all(row.get(k) == v for k, v in filters.items())]
    if len(picked) != 1:
        raise ValueError(f"Expected one row for filters={filters}, got {len(picked)}")
    return picked[0]


def try_get_meanshift(
    rows: list[dict[str, str]],
    method_name: str,
    nprobe: str,
    direction: str | None = None,
    r10_col: str = "r10",
    r100_col: str = "r100",
) -> tuple[float, float] | None:
    """Try to extract meanshift values; return (r10, r100) or None if not found."""
    filters: dict[str, str] = {"method": method_name, "nprobe": nprobe}
    if direction is not None:
        filters["direction"] = direction
    picked = [row for row in rows if all(row.get(k) == v for k, v in filters.items())]
    if len(picked) != 1:
        return None
    return (float(picked[0][r10_col]), float(picked[0][r100_col]))


def mean_metrics(
    rows: list[dict[str, str]],
    condition: str,
    direction: str,
    method: str,
    seed: str | None = None,
) -> tuple[float, float]:
    picked = [
        row
        for row in rows
        if row["condition"] == condition
        and row["direction"] == direction
        and row["method"] == method
        and (seed is None or row["seed"] == seed)
    ]
    if not picked:
        raise ValueError(
            f"No rows for condition={condition}, direction={direction}, method={method}, seed={seed}"
        )
    r10 = sum(float(row["r10"]) for row in picked) / len(picked)
    r100 = sum(float(row["r100"]) for row in picked) / len(picked)
    return r10, r100


def _safe_mean_metrics(
    rows: list[dict[str, str]],
    condition: str,
    direction: str,
    method: str,
    seed: str | None = None,
) -> tuple[float, float] | None:
    """Like mean_metrics but returns None when no matching rows exist."""
    picked = [
        row
        for row in rows
        if row["condition"] == condition
        and row["direction"] == direction
        and row["method"] == method
        and (seed is None or row["seed"] == seed)
    ]
    if not picked:
        return None
    r10 = sum(float(row["r10"]) for row in picked) / len(picked)
    r100 = sum(float(row["r100"]) for row in picked) / len(picked)
    return r10, r100


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
    q_delta_vp = fmt_delta(q_van[1], q_pmc[1])
    q_delta_mp = fmt_delta(q_ms[1], q_pmc[1]) if q_ms is not None else ""

    row: dict[str, str] = {
        "Dataset": dataset,
        "Enc": enc,
        "gap": GAPS[(dataset, enc)],
        "q_r10_van": fmt_recall(q_van[0], digits),
        "q_r10_ms": fmt_recall(q_ms[0], digits) if q_ms is not None else "",
        "q_r10_pmc": fmt_recall(q_pmc[0], digits),
        "q_r100_van": fmt_recall(q_van[1], digits),
        "q_r100_ms": fmt_recall(q_ms[1], digits) if q_ms is not None else "",
        "q_r100_pmc": fmt_recall(q_pmc[1], digits),
        "q_delta_vp": q_delta_vp,
        "q_delta_mp": q_delta_mp,
        "db_r10_van": "",
        "db_r10_ms": "",
        "db_r10_pmc": "",
        "db_r100_van": "",
        "db_r100_ms": "",
        "db_r100_pmc": "",
        "db_delta_vp": "",
        "db_delta_mp": "",
    }
    if db_van is not None and db_pmc is not None:
        db_delta_vp = fmt_delta(db_van[1], db_pmc[1])
        db_delta_mp = fmt_delta(db_ms[1], db_pmc[1]) if db_ms is not None else ""
        row["db_r10_van"] = fmt_recall(db_van[0], digits)
        row["db_r10_ms"] = fmt_recall(db_ms[0], digits) if db_ms is not None else ""
        row["db_r10_pmc"] = fmt_recall(db_pmc[0], digits)
        row["db_r100_van"] = fmt_recall(db_van[1], digits)
        row["db_r100_ms"] = fmt_recall(db_ms[1], digits) if db_ms is not None else ""
        row["db_r100_pmc"] = fmt_recall(db_pmc[1], digits)
        row["db_delta_vp"] = db_delta_vp
        row["db_delta_mp"] = db_delta_mp
    return row


def compute_table_rows(
    source_data: dict[str, list[dict[str, str]]],
    ms_data: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    summary = source_data["multiseed_rabitq_summary.csv"]
    flickr = source_data["pmc_eval_clip-l_flickr30k_full_seed42.csv"]
    clotho = source_data["pmc_clotho_r1_seed42.csv"]
    laion = source_data["pmc_laion400m_seed42.csv"]

    clip_mscoco_ms = ms_data.get("pmc_qps_pareto_clip_mscoco_seed42.csv", [])
    flickr_ms = ms_data.get("pmc_eval_clip-l_flickr30k_full_seed42.csv", [])
    clotho_ms = ms_data.get("pmc_clotho_r1_seed42.csv", [])
    laion_ms = ms_data.get("pmc_laion400m_seed42.csv", [])

    rows: list[dict[str, str]] = []

    # MSCOCO — CLIP-B/32
    clip_ms_q = try_get_meanshift(
        clip_mscoco_ms, "vanilla_rabitq_meanshift", "16", direction="text->image"
    )
    clip_ms_db = try_get_meanshift(
        clip_mscoco_ms, "vanilla_rabitq_meanshift", "16", direction="image->text"
    )
    rows.append(
        build_row(
            "MSCOCO",
            "CLIP",
            q_van=mean_metrics(summary, "MSCOCO_CLIP", "text->image", "vanilla"),
            q_pmc=mean_metrics(summary, "MSCOCO_CLIP", "text->image", "pmc"),
            db_van=mean_metrics(summary, "MSCOCO_CLIP", "image->text", "vanilla"),
            db_pmc=mean_metrics(summary, "MSCOCO_CLIP", "image->text", "pmc"),
            q_ms=clip_ms_q,
            db_ms=clip_ms_db,
        )
    )

    # MSCOCO — CL-L (meanshift from multiseed summary)
    cl_l_ms_q = _safe_mean_metrics(summary, "MSCOCO_CLIP-L", "text->image", "meanshift")
    cl_l_ms_db = _safe_mean_metrics(summary, "MSCOCO_CLIP-L", "image->text", "meanshift")
    rows.append(
        build_row(
            "MSCOCO",
            "CL-L",
            q_van=mean_metrics(summary, "MSCOCO_CLIP-L", "text->image", "vanilla"),
            q_pmc=mean_metrics(summary, "MSCOCO_CLIP-L", "text->image", "pmc"),
            db_van=mean_metrics(summary, "MSCOCO_CLIP-L", "image->text", "vanilla"),
            db_pmc=mean_metrics(summary, "MSCOCO_CLIP-L", "image->text", "pmc"),
            q_ms=cl_l_ms_q,
            db_ms=cl_l_ms_db,
        )
    )

    # MSCOCO — IB
    ib_mscoco_ms = ms_data.get("pmc_qps_pareto_imagebind_mscoco_seed42.csv", [])
    ib_ms_q = try_get_meanshift(
        ib_mscoco_ms, "vanilla_rabitq_meanshift", "16", direction="text->image"
    )
    ib_ms_db = try_get_meanshift(
        ib_mscoco_ms, "vanilla_rabitq_meanshift", "16", direction="image->text"
    )
    rows.append(
        build_row(
            "MSCOCO",
            "IB",
            q_van=mean_metrics(summary, "MSCOCO_IB", "text->image", "vanilla"),
            q_pmc=mean_metrics(summary, "MSCOCO_IB", "text->image", "pmc"),
            db_van=mean_metrics(summary, "MSCOCO_IB", "image->text", "vanilla"),
            db_pmc=mean_metrics(summary, "MSCOCO_IB", "image->text", "pmc"),
            q_ms=ib_ms_q,
            db_ms=ib_ms_db,
        )
    )

    # Flickr30K — CL-L
    flickr_ms_q = try_get_meanshift(
        flickr_ms, "vanilla_rabitq_meanshift", "16",
        direction="text->image",
        r10_col="recall_at_10", r100_col="recall_at_100",
    )
    flickr_ms_db = try_get_meanshift(
        flickr_ms, "vanilla_rabitq_meanshift", "16",
        direction="image->text",
        r10_col="recall_at_10", r100_col="recall_at_100",
    )
    rows.append(
        build_row(
            "Flickr30K",
            "CL-L",
            q_van=(
                float(pick_one(flickr, method="vanilla_rabitq", alpha="0.0", nprobe="16", direction="text->image")["recall_at_10"]),
                float(pick_one(flickr, method="vanilla_rabitq", alpha="0.0", nprobe="16", direction="text->image")["recall_at_100"]),
            ),
            q_pmc=(
                float(pick_one(flickr, method="pmc_1.00", alpha="1.0", nprobe="16", direction="text->image")["recall_at_10"]),
                float(pick_one(flickr, method="pmc_1.00", alpha="1.0", nprobe="16", direction="text->image")["recall_at_100"]),
            ),
            db_van=(
                float(pick_one(flickr, method="vanilla_rabitq", alpha="0.0", nprobe="16", direction="image->text")["recall_at_10"]),
                float(pick_one(flickr, method="vanilla_rabitq", alpha="0.0", nprobe="16", direction="image->text")["recall_at_100"]),
            ),
            db_pmc=(
                float(pick_one(flickr, method="pmc_1.00", alpha="1.0", nprobe="16", direction="image->text")["recall_at_10"]),
                float(pick_one(flickr, method="pmc_1.00", alpha="1.0", nprobe="16", direction="image->text")["recall_at_100"]),
            ),
            q_ms=flickr_ms_q,
            db_ms=flickr_ms_db,
        )
    )

    # Clotho — IB
    clotho_ms_q = try_get_meanshift(
        clotho_ms, "vanilla_meanshift", "16", direction="text->audio"
    )
    clotho_ms_db = try_get_meanshift(
        clotho_ms, "vanilla_meanshift", "16", direction="audio->text"
    )
    rows.append(
        build_row(
            "Clotho",
            "IB",
            q_van=(
                float(pick_one(clotho, method="vanilla_rabitq", alpha="0.0", nprobe="16", direction="text->audio")["r10"]),
                float(pick_one(clotho, method="vanilla_rabitq", alpha="0.0", nprobe="16", direction="text->audio")["r100"]),
            ),
            q_pmc=(
                float(pick_one(clotho, method="pmc_1.00", alpha="1.0", nprobe="16", direction="text->audio")["r10"]),
                float(pick_one(clotho, method="pmc_1.00", alpha="1.0", nprobe="16", direction="text->audio")["r100"]),
            ),
            db_van=(
                float(pick_one(clotho, method="vanilla_rabitq", alpha="0.0", nprobe="16", direction="audio->text")["r10"]),
                float(pick_one(clotho, method="vanilla_rabitq", alpha="0.0", nprobe="16", direction="audio->text")["r100"]),
            ),
            db_pmc=(
                float(pick_one(clotho, method="pmc_1.00", alpha="1.0", nprobe="16", direction="audio->text")["r10"]),
                float(pick_one(clotho, method="pmc_1.00", alpha="1.0", nprobe="16", direction="audio->text")["r100"]),
            ),
            q_ms=clotho_ms_q,
            db_ms=clotho_ms_db,
        )
    )

    # AudioCaps — IB (meanshift from multiseed summary when available)
    ac_ms_q = _safe_mean_metrics(summary, "AudioCaps_std", "text->audio", "meanshift", seed="42")
    ac_ms_db = _safe_mean_metrics(summary, "AudioCaps_std", "audio->text", "meanshift", seed="42")
    rows.append(
        build_row(
            "AudioCaps",
            "IB",
            q_van=mean_metrics(summary, "AudioCaps_std", "text->audio", "vanilla", seed="42"),
            q_pmc=mean_metrics(summary, "AudioCaps_std", "text->audio", "pmc", seed="42"),
            db_van=mean_metrics(summary, "AudioCaps_std", "audio->text", "vanilla", seed="42"),
            db_pmc=mean_metrics(summary, "AudioCaps_std", "audio->text", "pmc", seed="42"),
            q_ms=ac_ms_q,
            db_ms=ac_ms_db,
        )
    )

    # LAION-400M — CLIP (text->image only for forward; db-side from reverse CSV if available)
    laion_van = pick_one(laion, method="vanilla_rabitq", alpha="0.0", nprobe="256")
    laion_pmc = pick_one(laion, method="pmc_1.00", alpha="1.0", nprobe="256")
    laion_ms_q = try_get_meanshift(laion_ms, "vanilla_rabitq_meanshift", "256")

    laion_reverse_path = RESULTS_DIR / "pmc_laion400m_reverse_seed42.csv"
    db_van: tuple[float, float] | None = None
    db_pmc: tuple[float, float] | None = None
    db_ms: tuple[float, float] | None = None
    if laion_reverse_path.exists():
        laion_reverse = read_csv_rows(laion_reverse_path)
        laion_rev_van = pick_one(laion_reverse, method="vanilla_rabitq", alpha="0.0", nprobe="256")
        laion_rev_pmc = pick_one(laion_reverse, method="pmc_1.00", alpha="1.0", nprobe="256")
        db_van = (float(laion_rev_van["r10"]), float(laion_rev_van["r100"]))
        db_pmc = (float(laion_rev_pmc["r10"]), float(laion_rev_pmc["r100"]))
        laion_reverse_ms = ms_data.get("pmc_laion400m_reverse_seed42.csv", [])
        db_ms = try_get_meanshift(laion_reverse_ms, "vanilla_rabitq_meanshift", "256")

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
    source_data = ensure_sources_readable()
    ms_data = load_ms_sources()
    table_rows = compute_table_rows(source_data, ms_data)
    write_output_csv(table_rows)
    print_markdown_table(table_rows)
    print(f"\nWrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
