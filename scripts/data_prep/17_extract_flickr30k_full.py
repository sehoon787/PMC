"""
17_extract_flickr30k_full.py -- Extract CLIP-L features for Flickr30K full (31K).

Processes images in batches to avoid OOM on machines with limited RAM.

Produces:
  data/features/flickr30k_full_clip-l_image_seed42.npy
  data/features/flickr30k_full_clip-l_text_seed42.npy
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
from PIL import Image

from src.runtime.config import CFG
from src.datasets.flickr30k import download_flickr30k_full
from src.encoders.clip import CLIPEncoder

SEED = 42
BACKBONE = "clip-l"
BACKBONE_NAME = "clip-ViT-L-14"
IMAGE_BATCH_SIZE = 4


def encode_images_batched(encoder, paths: list, batch_size: int) -> np.ndarray:
    """Encode images in batches to avoid OOM."""
    encoder._load()
    all_embs = []
    n = len(paths)
    for i in range(0, n, batch_size):
        batch_paths = paths[i:i + batch_size]
        images = [Image.open(str(p)).convert("RGB") for p in batch_paths]
        emb = encoder._model.encode(images, convert_to_numpy=True, show_progress_bar=False)
        all_embs.append(np.asarray(emb, dtype=np.float32))
        # Free image memory
        del images
        print(f"    [{i + len(batch_paths)}/{n}]", flush=True)
    return np.vstack(all_embs)


def main() -> None:
    feat_dir = CFG.features_dir
    feat_dir.mkdir(parents=True, exist_ok=True)

    img_path = feat_dir / f"flickr30k_full_{BACKBONE}_image_seed{SEED}.npy"
    txt_path = feat_dir / f"flickr30k_full_{BACKBONE}_text_seed{SEED}.npy"

    if img_path.exists() and txt_path.exists():
        print(f"[extract] Features already exist: {img_path}, {txt_path}")
        return

    raw_dir = CFG.raw_dir / "flickr30k"
    print(f"[extract] Loading Flickr30K full from {raw_dir} ...")
    items = download_flickr30k_full(raw_dir, download_images=False)
    print(f"[extract] Loaded {len(items)} items")

    items_with_images = [it for it in items if it.image_path.exists()]
    print(f"[extract] {len(items_with_images)} items have images on disk")

    if len(items_with_images) < 30000:
        print(f"[extract] WARNING: Only {len(items_with_images)} images available (expected ~31K)")

    print(f"[extract] Initializing {BACKBONE_NAME} ...")
    encoder = CLIPEncoder(model_name=BACKBONE_NAME)

    # Encode images in batches
    if not img_path.exists():
        print(f"[extract] Encoding {len(items_with_images)} images (batch={IMAGE_BATCH_SIZE}) ...")
        paths = [it.image_path for it in items_with_images]
        img_emb = encode_images_batched(encoder, paths, IMAGE_BATCH_SIZE)
        np.save(str(img_path), img_emb)
        print(f"  image embeddings: {img_emb.shape} -> {img_path}")
        del img_emb
    else:
        print(f"[extract] Image features exist: {img_path}")

    # Encode text (lightweight, no batching needed)
    if not txt_path.exists():
        print(f"[extract] Encoding {len(items_with_images)} texts ...")
        captions = [it.captions[0] if it.captions else "" for it in items_with_images]
        txt_emb = encoder.encode_text(captions, batch_size=512)
        np.save(str(txt_path), txt_emb)
        print(f"  text embeddings: {txt_emb.shape} -> {txt_path}")
    else:
        print(f"[extract] Text features exist: {txt_path}")

    print(f"[extract] Done.")


if __name__ == "__main__":
    main()
