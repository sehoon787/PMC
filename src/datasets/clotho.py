"""Clotho v2 evaluation split helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

import numpy as np

from .items import ClothoItem

CLOTHO_CAPTION_COLUMNS = tuple(f"caption_{i}" for i in range(1, 6))


def load_clotho_evaluation(
    captions_csv: Path,
    audio_dir: Optional[Path] = None,
    *,
    require_audio: bool = False,
    min_audio_bytes: int = 4096,
) -> List[ClothoItem]:
    """Load Clotho v2 evaluation captions in CSV row order.

    The standard evaluation protocol uses exactly five captions per audio clip.
    CSV row order is significant because cached 5-caption text features are
    expected to be ordered clip0_cap1..clip0_cap5, clip1_cap1..clip1_cap5, ...
    """
    captions_csv = Path(captions_csv)
    if not captions_csv.is_file():
        raise FileNotFoundError(f"Clotho captions CSV not found: {captions_csv}")

    audio_root = Path(audio_dir) if audio_dir is not None else None
    items: List[ClothoItem] = []

    with open(captions_csv, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        required = {"file_name", *CLOTHO_CAPTION_COLUMNS}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(
                f"Clotho captions CSV is missing required columns: {missing}"
            )

        for row in reader:
            file_name = row["file_name"].strip()
            captions = [row[col].strip() for col in CLOTHO_CAPTION_COLUMNS]
            if any(not caption for caption in captions):
                raise ValueError(f"Clotho row has an empty caption: {file_name}")

            audio_path = (audio_root / file_name) if audio_root else Path(file_name)
            if require_audio:
                is_too_small = (
                    audio_path.exists()
                    and audio_path.stat().st_size < min_audio_bytes
                )
                if not audio_path.exists() or is_too_small:
                    continue

            items.append(
                ClothoItem(
                    file_name=file_name,
                    audio_path=audio_path,
                    captions=captions,
                )
            )

    return items


def build_clotho_standard_ground_truth(
    n_audio: int,
    captions_per_audio: int = 5,
) -> tuple[np.ndarray, List[set[int]]]:
    """Return t->a single-GT and a->t multi-GT for Clotho standard eval."""
    if n_audio <= 0:
        raise ValueError(f"n_audio must be positive, got {n_audio}")
    if captions_per_audio <= 0:
        raise ValueError(
            f"captions_per_audio must be positive, got {captions_per_audio}"
        )

    text_to_audio = np.repeat(np.arange(n_audio), captions_per_audio)
    audio_to_text = [
        set(range(i * captions_per_audio, (i + 1) * captions_per_audio))
        for i in range(n_audio)
    ]
    return text_to_audio, audio_to_text


def validate_clotho_standard_feature_shapes(
    audio_features: np.ndarray,
    text_features: np.ndarray,
    captions_per_audio: int = 5,
) -> None:
    """Validate cached feature counts for Clotho standard 5-caption eval."""
    if audio_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError(
            "Clotho feature arrays must both be rank-2 "
            f"(got {audio_features.shape} and {text_features.shape})"
        )
    if audio_features.shape[1] != text_features.shape[1]:
        raise ValueError(
            "Clotho audio/text feature dimensions differ: "
            f"{audio_features.shape[1]} != {text_features.shape[1]}"
        )

    expected_text = audio_features.shape[0] * captions_per_audio
    if text_features.shape[0] != expected_text:
        raise ValueError(
            "Clotho text feature count must equal audio_count * captions_per_audio "
            f"({text_features.shape[0]} != {audio_features.shape[0]} * "
            f"{captions_per_audio})"
        )
