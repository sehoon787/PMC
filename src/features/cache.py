"""Feature encoding cache helper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

import numpy as np

from src.datasets.items import AudioCapsItem, ClothoItem, MSCOCOItem
from src.encoders.clap import ClapEncoder
from src.encoders.fake import FakeEncoder
from src.encoders.imagebind import ImageBindEncoder

MODALITY_IMAGE = "image"
MODALITY_TEXT = "text"
MODALITY_AUDIO = "audio"

VALID_MODALITIES = {MODALITY_IMAGE, MODALITY_TEXT, MODALITY_AUDIO}


def encode_dataset(
    encoder: Union[FakeEncoder, ImageBindEncoder, ClapEncoder],
    items: Union[List[MSCOCOItem], List[AudioCapsItem], List[ClothoItem]],
    modality: str,
    cache_path: Path,
    force: bool = False,
) -> tuple[np.ndarray, list[int]]:
    """Encode items and cache embeddings as .npy plus row ids as .json."""
    assert modality in VALID_MODALITIES, f"Unknown modality: {modality!r}"
    if len(items) == 0:
        raise ValueError(
            f"encode_dataset received an empty item list for modality={modality!r}. "
            "Check that the dataset directory exists and contains the expected files."
        )

    npy_path = Path(str(cache_path) + ".npy")
    json_path = Path(str(cache_path) + ".json")

    if not force and npy_path.exists() and json_path.exists():
        print(f"  [encode] Loading cached {modality} embeddings from {npy_path}")
        embeddings = np.load(str(npy_path))
        with open(json_path) as fh:
            ids = json.load(fh)
        return embeddings, ids

    print(f"  [encode] Encoding {len(items)} items as {modality} ...")
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    is_audio_caps = hasattr(items[0], "audiocap_id")
    is_clotho = hasattr(items[0], "file_name") and not hasattr(items[0], "image_id")
    has_audio = is_audio_caps or is_clotho

    if modality == MODALITY_AUDIO:
        if not has_audio:
            raise ValueError(
                "modality='audio' requires AudioCapsItem or ClothoItem; "
                f"got {type(items[0]).__name__}."
            )
        paths = [item.audio_path for item in items]
        ids = [item.audiocap_id for item in items] if is_audio_caps else [item.file_name for item in items]
        embeddings = encoder.encode_audio(paths)
    elif modality == MODALITY_IMAGE:
        if has_audio:
            raise ValueError(
                "modality='image' is not valid for audio items. "
                "Use modality='audio' or modality='text'."
            )
        paths = [item.image_path for item in items]
        embeddings = encoder.encode_image(paths)
        ids = [item.image_id for item in items]
    elif modality == MODALITY_TEXT:
        captions = [item.captions[0] if item.captions else "" for item in items]
        embeddings = encoder.encode_text(captions)
        if is_audio_caps:
            ids = [item.audiocap_id for item in items]
        elif is_clotho:
            ids = [item.file_name for item in items]
        else:
            ids = [item.image_id for item in items]
    else:
        raise ValueError(f"encode_dataset: unsupported modality={modality!r}.")

    assert embeddings.ndim == 2, f"Expected 2-D embedding array, got shape {embeddings.shape}"
    assert embeddings.shape[0] == len(items), (
        f"Embedding count {embeddings.shape[0]} != item count {len(items)}"
    )

    np.save(str(npy_path), embeddings)
    with open(json_path, "w") as fh:
        json.dump(ids, fh)

    print(f"  [encode] Saved embeddings -> {npy_path} ({embeddings.nbytes / 1e6:.1f} MB)")
    return embeddings, ids
