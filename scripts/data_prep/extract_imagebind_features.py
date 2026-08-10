"""
extract_imagebind_features.py -- Extract ImageBind features for PQ/OPQ experiments.

Downloads MSCOCO val2017 if needed, then extracts ImageBind (d=1024) features:
  - MSCOCO val5K: image + text embeddings
  - AudioCaps test: audio + text embeddings (if audio files available)

Output: data/features/mscoco_karpathy_val5k_imagebind_{image,text}_seed42.npy
        data/features/audiocaps_test_imagebind_{audio,text}_seed42.npy
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_V4_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "src").is_dir() and (parent / "config").is_dir()
)
if str(_V4_ROOT) not in sys.path:
    sys.path.insert(0, str(_V4_ROOT))

import numpy as np

from src.runtime.config import CFG as V4_CFG
from src.datasets.audiocaps import download_audiocaps_test
from src.datasets.mscoco import download_mscoco_val5k
from src.datasets.downloads import ensure_mscoco_val2017
from src.encoders.imagebind import ImageBindEncoder
from src.features.cache import encode_dataset

SEED = 42


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(SEED)
    feat_dir = V4_CFG.features_dir
    feat_dir.mkdir(parents=True, exist_ok=True)
    mscoco_dir = V4_CFG.raw_dir / "mscoco"

    # ------------------------------------------------------------------
    # 1. Download MSCOCO val2017
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Ensure MSCOCO val2017 data")
    print("=" * 60)
    ensure_mscoco_val2017(mscoco_dir)

    # ------------------------------------------------------------------
    # 2. Load MSCOCO items
    # ------------------------------------------------------------------
    print("\nStep 2: Load MSCOCO val5K items")
    items = download_mscoco_val5k(mscoco_dir)
    print(f"  Loaded {len(items)} MSCOCO items")

    # ------------------------------------------------------------------
    # 3. Initialize ImageBind encoder (CPU, float32)
    # ------------------------------------------------------------------
    print("\nStep 3: Initialize ImageBind encoder")
    encoder = ImageBindEncoder(device="cpu", dtype="float32")

    # ------------------------------------------------------------------
    # 4. Extract MSCOCO features
    # ------------------------------------------------------------------
    print("\nStep 4: Extract MSCOCO ImageBind features")

    img_cache = feat_dir / f"mscoco_karpathy_val5k_imagebind_image_seed{SEED}"
    txt_cache = feat_dir / f"mscoco_karpathy_val5k_imagebind_text_seed{SEED}"

    img_emb, img_ids = encode_dataset(encoder, items, "image", img_cache)
    print(f"  Image embeddings: {img_emb.shape}")

    txt_emb, txt_ids = encode_dataset(encoder, items, "text", txt_cache)
    print(f"  Text embeddings: {txt_emb.shape}")

    # ------------------------------------------------------------------
    # 5. AudioCaps (conditional)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: AudioCaps (conditional)")
    print("=" * 60)

    ac_dir = V4_CFG.audiocaps_dir
    ac_csv = V4_CFG.audiocaps_metadata_csv

    if not ac_dir.is_dir() or not ac_csv.is_file():
        print(f"  AudioCaps data not found at {ac_dir}")
        print(f"  Expected: {ac_csv} + wav files in {ac_dir}/")
        print("  SKIPPING AudioCaps feature extraction.")
    else:
        try:
            ac_items = download_audiocaps_test(ac_dir, metadata_csv=ac_csv)
            if len(ac_items) == 0:
                print("  No AudioCaps items loaded (no wav files?). Skipping.")
            else:
                print(f"  Loaded {len(ac_items)} AudioCaps items")

                aud_cache = feat_dir / f"audiocaps_test_imagebind_audio_seed{SEED}"
                atxt_cache = feat_dir / f"audiocaps_test_imagebind_text_seed{SEED}"

                aud_emb, aud_ids = encode_dataset(encoder, ac_items, "audio", aud_cache)
                print(f"  Audio embeddings: {aud_emb.shape}")

                atxt_emb, atxt_ids = encode_dataset(encoder, ac_items, "text", atxt_cache)
                print(f"  Text embeddings: {atxt_emb.shape}")
        except Exception as e:
            print(f"  AudioCaps error: {e}")
            print("  SKIPPING AudioCaps.")

    print("\n" + "=" * 60)
    print("DONE. Feature files:")
    for f in sorted(feat_dir.glob("*imagebind*.npy")):
        arr = np.load(str(f))
        print(f"  {f.name}: {arr.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
