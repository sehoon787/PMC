"""
extract_imagebind_audiocaps_hf.py -- AudioCaps ImageBind feature extraction from the
OFFICIAL HuggingFace protocol CSV (test_official_hf.csv).

Background
----------
The PMC AudioCaps experiments use the official HF AudioCaps test protocol:
  - 4411 captions  (one row per audiocap_id, each audiocap_id has 1 caption)
  - 883  unique audio clips, keyed by (youtube_id, start_time); each clip is
    referenced by ~5 distinct audiocap_ids that share the SAME audio.
  - Every audiocap_id has its own {audiocap_id}.wav on disk, and all wavs for a
    given clip are byte-identical copies.

Source CSV : data/raw/audiocaps/test_audio/test_official_hf.csv
WAV dir    : data/raw/audiocaps/test_audio/test/{audiocap_id}.wav

Outputs (float32, RAW embeddings -- NOT pre-normalized; consumers l2-normalize):
  - audiocaps_test_imagebind_text_seed42.npy          (4411, 1024)
  - audiocaps_test_imagebind_text_seed42.json         list[int] of 4411 audiocap_ids
  - audiocaps_test_imagebind_audio_seed42.npy         (4411, 1024) per-caption audio
  - audiocaps_test_imagebind_audio_seed42.json        list[int] of 4411 audiocap_ids
  - audiocaps_test_imagebind_audio_single_seed42.npy  (883, 1024) one row per clip
  - audiocaps_test_imagebind_audio_single_seed42.json list[int] representative aid/clip

Ordering
--------
Per-caption rows are sorted by audiocap_id (matches encode_dataset's MSCOCO/
AudioCaps convention of sorting by id). Per-clip (_single) rows are ordered by
the audiocap_id of the FIRST occurrence of each unique clip in audiocap_id-sorted
order (the documented safe default).

This script does NOT modify any shared loader; it builds AudioCapsItem lists
inline and reuses src.encoders.imagebind + src.features.cache.encode_dataset.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_V4_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "src").is_dir() and (parent / "config").is_dir()
)
if str(_V4_ROOT) not in sys.path:
    sys.path.insert(0, str(_V4_ROOT))

import numpy as np

from src.runtime.config import CFG as V4_CFG
from src.datasets.items import AudioCapsItem
from src.encoders.imagebind import ImageBindEncoder
from src.features.cache import encode_dataset

SEED = 42

# Official HF protocol CSV + wav directory (NOT the wrong test.csv).
# NOTE: config audiocaps_dir already points at the wav dir (.../test_audio/test
# on this machine via paths.local.yaml). The official CSV sits one level up,
# next to the test/ wav folder.
WAV_DIR = V4_CFG.audiocaps_dir
OFFICIAL_CSV = V4_CFG.audiocaps_dir.parent / "test_official_hf.csv"

MIN_WAV_BYTES = 4096  # mirror loader's short/empty guard


def load_official_rows(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"No rows parsed from {csv_path}")
    return rows


def build_caption_items(rows: List[Dict[str, str]], wav_dir: Path) -> List[AudioCapsItem]:
    """One AudioCapsItem per audiocap_id (per caption), sorted by audiocap_id."""
    # audiocap_id is unique per row in the official protocol; assert that.
    by_id: Dict[int, Dict[str, str]] = {}
    for r in rows:
        aid = int(r["audiocap_id"])
        if aid in by_id:
            raise RuntimeError(
                f"Duplicate audiocap_id {aid} in official CSV; expected 1 caption per id."
            )
        by_id[aid] = r

    items: List[AudioCapsItem] = []
    skipped_missing = 0
    skipped_short = 0
    for aid in sorted(by_id.keys()):
        wav = wav_dir / f"{aid}.wav"
        if not wav.exists():
            skipped_missing += 1
            continue
        if wav.stat().st_size < MIN_WAV_BYTES:
            skipped_short += 1
            continue
        items.append(
            AudioCapsItem(
                audiocap_id=aid,
                audio_path=wav,
                captions=[by_id[aid]["caption"].strip()],
            )
        )
    print(
        f"  caption items: {len(items)} "
        f"(skipped {skipped_missing} missing wav, {skipped_short} short/empty)"
    )
    return items


def build_clip_index(
    caption_items: List[AudioCapsItem],
    rows: List[Dict[str, str]],
) -> Tuple[List[AudioCapsItem], List[int]]:
    """Deduplicate to unique clips.

    Returns:
      single_items: list of AudioCapsItem, one per unique (youtube_id, start_time),
                    represented by the FIRST-occurring audiocap_id in
                    audiocap_id-sorted order.
      cap_to_clip:  for each caption_item (aligned by index), the index into
                    single_items of its clip -> used to assemble per-caption audio.
    """
    aid_to_clip: Dict[int, Tuple[str, str]] = {
        int(r["audiocap_id"]): (r["youtube_id"], str(r["start_time"])) for r in rows
    }

    clip_to_single_idx: Dict[Tuple[str, str], int] = {}
    single_items: List[AudioCapsItem] = []
    cap_to_clip: List[int] = []

    for item in caption_items:  # already sorted by audiocap_id
        clip = aid_to_clip[item.audiocap_id]
        if clip not in clip_to_single_idx:
            clip_to_single_idx[clip] = len(single_items)
            single_items.append(
                AudioCapsItem(
                    audiocap_id=item.audiocap_id,  # representative = first aid for clip
                    audio_path=item.audio_path,
                    captions=item.captions,
                )
            )
        cap_to_clip.append(clip_to_single_idx[clip])

    print(f"  unique clips: {len(single_items)}")
    return single_items, cap_to_clip


def save_npy_json(npy_path: Path, json_path: Path, emb: np.ndarray, ids: List[int]) -> None:
    assert emb.dtype == np.float32, f"expected float32, got {emb.dtype}"
    assert emb.shape[0] == len(ids), f"emb rows {emb.shape[0]} != ids {len(ids)}"
    np.save(str(npy_path), emb)
    with open(json_path, "w") as fh:
        json.dump([int(i) for i in ids], fh)
    print(f"  wrote {npy_path.name} {emb.shape} + {json_path.name} ({len(ids)} ids)")


def main() -> None:
    np.random.seed(SEED)
    feat_dir = V4_CFG.features_dir
    feat_dir.mkdir(parents=True, exist_ok=True)

    if not OFFICIAL_CSV.is_file():
        print(f"ERROR: official CSV not found: {OFFICIAL_CSV}")
        sys.exit(1)
    if not WAV_DIR.is_dir():
        print(f"ERROR: wav dir not found: {WAV_DIR}")
        sys.exit(1)

    print("=" * 60)
    print("Step 1: Load official HF AudioCaps protocol")
    print("=" * 60)
    print(f"  CSV : {OFFICIAL_CSV}")
    print(f"  WAVs: {WAV_DIR}")
    rows = load_official_rows(OFFICIAL_CSV)
    print(f"  CSV rows: {len(rows)}")

    caption_items = build_caption_items(rows, WAV_DIR)
    single_items, cap_to_clip = build_clip_index(caption_items, rows)

    print("\nStep 2: Initialize ImageBind encoder (CUDA, float32)")
    encoder = ImageBindEncoder(device="cuda", dtype="float32")

    # --- Text (per caption, 4411) -----------------------------------------
    print("\nStep 3: Extract text embeddings (per caption)")
    text_cache = feat_dir / f"audiocaps_test_imagebind_text_seed{SEED}"
    text_emb, text_ids = encode_dataset(
        encoder, caption_items, "text", text_cache, force=True
    )
    print(f"  text embeddings: {text_emb.shape}")

    # --- Audio single (per unique clip, 883) ------------------------------
    print("\nStep 4: Extract audio embeddings (per unique clip)")
    single_cache = feat_dir / f"audiocaps_test_imagebind_audio_single_seed{SEED}"
    single_emb, single_ids = encode_dataset(
        encoder, single_items, "audio", single_cache, force=True
    )
    print(f"  audio_single embeddings: {single_emb.shape}")

    # --- Audio full (per caption, 4411) = single clip emb repeated --------
    print("\nStep 5: Assemble per-caption audio embeddings from clip dedup")
    audio_full = single_emb[np.asarray(cap_to_clip, dtype=np.int64)].astype(np.float32)
    assert audio_full.shape[0] == len(caption_items)
    audio_ids = [it.audiocap_id for it in caption_items]
    save_npy_json(
        feat_dir / f"audiocaps_test_imagebind_audio_seed{SEED}.npy",
        feat_dir / f"audiocaps_test_imagebind_audio_seed{SEED}.json",
        audio_full,
        audio_ids,
    )

    print("\n" + "=" * 60)
    print("DONE. AudioCaps ImageBind feature files:")
    for f in sorted(feat_dir.glob("audiocaps_test_imagebind*.npy")):
        arr = np.load(str(f))
        print(f"  {f.name}: {arr.shape} {arr.dtype}")
    print("=" * 60)


if __name__ == "__main__":
    main()
