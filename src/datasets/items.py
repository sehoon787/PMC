"""Dataset item dataclasses shared by loaders and feature encoders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class MSCOCOItem:
    image_id: int
    image_path: Path
    captions: List[str]


@dataclass
class AudioCapsItem:
    audiocap_id: int
    audio_path: Path
    captions: List[str]


@dataclass
class ClothoItem:
    file_name: str
    audio_path: Path
    captions: List[str]
