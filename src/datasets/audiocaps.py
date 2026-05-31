"""AudioCaps test split loader."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Optional

from .items import AudioCapsItem

log = logging.getLogger(__name__)

def download_audiocaps_test(
    target_dir: Path,
    metadata_csv: Optional[Path] = None,
) -> List[AudioCapsItem]:
    """Load AudioCaps test split.

    Returns list of AudioCapsItem(audiocap_id, audio_path, captions).

    Each item:
      - audio_path: Path to {audiocap_id}.wav
      - captions: list of caption strings (1+; we use captions[0] for paired eval)

    Args:
      target_dir: where audio files live
      metadata_csv: path to test.csv.  If None, raises FileNotFoundError
                    with instructions.

    Returns items sorted by audiocap_id; only items whose .wav file exists.
    """
    target_dir = Path(target_dir)

    if metadata_csv is None:
        raise FileNotFoundError(
            "AudioCaps metadata CSV path must be provided via metadata_csv=<path>.\n"
            "Expected columns: audiocap_id, youtube_id, start_time, caption\n"
            "Download from: https://huggingface.co/datasets/confit/audiocaps"
        )

    metadata_csv = Path(metadata_csv)
    if not metadata_csv.is_file():
        raise FileNotFoundError(
            f"AudioCaps metadata CSV not found: {metadata_csv}\n"
            "Expected columns: audiocap_id, youtube_id, start_time, caption\n"
            "Pass metadata_csv=<path> to specify an alternate location."
        )

    # Parse CSV: group captions by audiocap_id
    id_to_captions: dict[int, List[str]] = {}
    with open(metadata_csv, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            aid = int(row["audiocap_id"])
            caption = row["caption"].strip()
            id_to_captions.setdefault(aid, []).append(caption)

    items: List[AudioCapsItem] = []
    skipped_missing = 0
    skipped_bad = 0
    for aid in sorted(id_to_captions.keys()):
        wav_path = target_dir / f"{aid}.wav"
        if not wav_path.exists():
            skipped_missing += 1
            continue  # skip if audio file not downloaded
        # Skip files that are too short to process through ImageBind's mel-spectrogram
        # pipeline (which resamples to 16kHz and requires >= 400 samples / 25 ms).
        # Use a conservative file-size threshold: valid 10-second AudioCap clips at
        # 48kHz/16-bit stereo are ~1.9 MB; anything under 4KB is either empty or <1ms.
        if wav_path.stat().st_size < 4096:
            skipped_bad += 1
            log.warning("[audiocaps] Skipping short/empty file: %s", wav_path)
            continue
        items.append(
            AudioCapsItem(
                audiocap_id=aid,
                audio_path=wav_path,
                captions=id_to_captions[aid],
            )
        )

    log.info(
        "[audiocaps] Loaded %d items from %s (skipped %d missing, %d short/empty)",
        len(items), target_dir, skipped_missing, skipped_bad,
    )
    return items
