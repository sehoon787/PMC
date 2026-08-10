"""
22_download_audiocaps.py -- Download AudioCaps test split audio files.

Downloads:
  1. AudioCaps test.csv metadata from HuggingFace
  2. Audio clips from YouTube using yt-dlp (10s segments)

Output: data/raw/audiocaps/test.csv + data/raw/audiocaps/{audiocap_id}.wav
"""

from __future__ import annotations

import sys
from pathlib import Path

_V4_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "src").is_dir() and (parent / "config").is_dir()
)
if str(_V4_ROOT) not in sys.path:
    sys.path.insert(0, str(_V4_ROOT))

from src.runtime.config import CFG as V4_CFG
from src.datasets.downloads import (
    download_audiocaps_clips,
    download_audiocaps_metadata,
    parse_audiocaps_metadata,
)


def main():
    target_dir = V4_CFG.audiocaps_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    yt_dlp = V4_CFG.require_yt_dlp()

    # 1. Download metadata
    print("=" * 60)
    print("Step 1: Download AudioCaps test metadata")
    print("=" * 60)
    csv_path = download_audiocaps_metadata(target_dir)
    items = parse_audiocaps_metadata(csv_path)
    print(f"  {len(items)} unique clips in test split")

    # 2. Download audio clips
    print(f"\n{'='*60}")
    print("Step 2: Download audio clips from YouTube")
    print("=" * 60)

    download_audiocaps_clips(items=items, target_dir=target_dir, yt_dlp=yt_dlp, max_workers=4)

    # Summary
    total_wav = len(list(target_dir.glob("*.wav")))
    valid_wav = sum(1 for f in target_dir.glob("*.wav") if f.stat().st_size > 4096)
    print(f"\n{'='*60}")
    print(f"DONE. {valid_wav} valid wav files out of {len(items)} clips")
    print(f"  Directory: {target_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
