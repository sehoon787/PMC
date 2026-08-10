"""
21_audiocaps_imagebind_only.py -- Fast AudioCaps-only ImageBind feature extraction (CUDA).

Extracts ImageBind (d=1024) features for AudioCaps test set only:
  - AudioCaps test: audio + text embeddings

Output: data/features/audiocaps_test_imagebind_audio_seed42.npy
        data/features/audiocaps_test_imagebind_audio_seed42.json  (sidecar with audiocap_ids)
        data/features/audiocaps_test_imagebind_text_seed42.npy
        data/features/audiocaps_test_imagebind_text_seed42.json   (sidecar with audiocap_ids)

Skips MSCOCO entirely. Uses CUDA for ~10-15 min runtime vs 6+ hr CPU+MSCOCO run.
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

    # ------------------------------------------------------------------
    # 1. Check AudioCaps data availability
    # ------------------------------------------------------------------
    ac_dir = V4_CFG.audiocaps_dir
    ac_csv = V4_CFG.audiocaps_metadata_csv

    if not ac_dir.is_dir():
        print(f"ERROR: AudioCaps audio directory not found: {ac_dir}")
        print("  Place wav files under that directory and re-run.")
        sys.exit(1)

    if not ac_csv.is_file():
        print(f"ERROR: AudioCaps metadata CSV not found: {ac_csv}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load AudioCaps items
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Load AudioCaps test items")
    print("=" * 60)
    ac_items = download_audiocaps_test(ac_dir, metadata_csv=ac_csv)
    if len(ac_items) == 0:
        print("ERROR: No AudioCaps items loaded (no wav files found?).")
        sys.exit(1)
    print(f"  Loaded {len(ac_items)} AudioCaps items")

    # ------------------------------------------------------------------
    # 3. Initialize ImageBind encoder (CUDA, float32)
    # ------------------------------------------------------------------
    print("\nStep 2: Initialize ImageBind encoder (CUDA)")
    encoder = ImageBindEncoder(device="cuda", dtype="float32")

    # ------------------------------------------------------------------
    # 4. Extract audio embeddings
    # ------------------------------------------------------------------
    print("\nStep 3: Extract AudioCaps audio embeddings")
    aud_cache = feat_dir / f"audiocaps_test_imagebind_audio_seed{SEED}"
    aud_emb, aud_ids = encode_dataset(encoder, ac_items, "audio", aud_cache)
    print(f"  Audio embeddings: {aud_emb.shape}")

    # ------------------------------------------------------------------
    # 5. Extract text embeddings
    # ------------------------------------------------------------------
    print("\nStep 4: Extract AudioCaps text embeddings")
    atxt_cache = feat_dir / f"audiocaps_test_imagebind_text_seed{SEED}"
    atxt_emb, atxt_ids = encode_dataset(encoder, ac_items, "text", atxt_cache)
    print(f"  Text embeddings: {atxt_emb.shape}")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DONE. AudioCaps ImageBind feature files:")
    for f in sorted(feat_dir.glob("audiocaps_test_imagebind*.npy")):
        arr = np.load(str(f))
        print(f"  {f.name}: {arr.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
