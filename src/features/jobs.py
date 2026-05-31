"""Reusable feature extraction jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.runtime.config import CFG
from src.datasets.flickr30k import download_flickr30k_test1k
from src.datasets.mscoco import download_mscoco_karpathy_val5k
from src.encoders.clip import CLIPEncoder
from src.features.cache import encode_dataset

BACKBONE_MAP = {
    "clip": "clip-ViT-B-32",
    "clip-l": "clip-ViT-L-14",
}

DATASET_LOADERS: dict[str, Callable] = {
    "mscoco": download_mscoco_karpathy_val5k,
    "flickr30k": download_flickr30k_test1k,
}

FEATURE_NAME_MAP = {
    "mscoco": "mscoco_karpathy_val5k",
    "flickr30k": "flickr30k_test1k",
}


@dataclass(frozen=True)
class FeatureExtractionSpec:
    backbone: str
    dataset: str
    seed: int
    force: bool = False
    raw_dir: Path | None = None
    features_dir: Path = CFG.features_dir


def resolve_clip_raw_dir(dataset: str, raw_dir: Path | None = None) -> Path:
    """Resolve the raw dataset directory used by CLIP feature extraction."""
    if raw_dir is not None:
        return Path(raw_dir)
    return CFG.raw_dir / ("mscoco_karpathy" if dataset == "mscoco" else "flickr30k")


def feature_cache_paths(
    *,
    features_dir: Path,
    dataset: str,
    backbone: str,
    seed: int,
) -> tuple[Path, Path]:
    """Return image/text cache paths for a dataset/backbone/seed."""
    dataset_prefix = FEATURE_NAME_MAP[dataset]
    img_path = features_dir / f"{dataset_prefix}_{backbone}_image_seed{seed}.npy"
    txt_path = features_dir / f"{dataset_prefix}_{backbone}_text_seed{seed}.npy"
    return img_path, txt_path


def run_clip_feature_extraction(spec: FeatureExtractionSpec) -> tuple[Path, Path]:
    """Run CLIP feature extraction and return the image/text .npy paths."""
    if spec.backbone not in BACKBONE_MAP:
        raise ValueError(f"Unsupported backbone: {spec.backbone}")
    if spec.dataset not in DATASET_LOADERS:
        raise ValueError(f"Unsupported dataset: {spec.dataset}")

    feat_dir = Path(spec.features_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = resolve_clip_raw_dir(spec.dataset, spec.raw_dir)
    backbone_name = BACKBONE_MAP[spec.backbone]
    dataset_prefix = FEATURE_NAME_MAP[spec.dataset]
    img_path, txt_path = feature_cache_paths(
        features_dir=feat_dir,
        dataset=spec.dataset,
        backbone=spec.backbone,
        seed=spec.seed,
    )

    if not spec.force and img_path.exists() and txt_path.exists():
        print("[extract] Features already exist:")
        print(f"  {img_path}")
        print(f"  {txt_path}")
        print("[extract] Use --force to re-extract.")
        return img_path, txt_path

    print(f"[extract] Loading {spec.dataset} items from {raw_dir} ...")
    loader = DATASET_LOADERS[spec.dataset]
    items = loader(raw_dir, download_images=True)
    print(f"[extract] Loaded {len(items)} items")

    print(f"[extract] Initializing {backbone_name} ...")
    encoder = CLIPEncoder(model_name=backbone_name)

    print("[extract] Encoding images ...")
    img_cache = feat_dir / f"{dataset_prefix}_{spec.backbone}_image_seed{spec.seed}"
    img_emb, _ = encode_dataset(encoder, items, "image", img_cache, force=spec.force)
    print(f"  image embeddings: {img_emb.shape}")

    print("[extract] Encoding text ...")
    txt_cache = feat_dir / f"{dataset_prefix}_{spec.backbone}_text_seed{spec.seed}"
    txt_emb, _ = encode_dataset(encoder, items, "text", txt_cache, force=spec.force)
    print(f"  text embeddings: {txt_emb.shape}")

    print(f"\n[extract] Done. Features saved to {feat_dir}/")
    print(f"  {img_path.name}  shape={img_emb.shape}")
    print(f"  {txt_path.name}  shape={txt_emb.shape}")
    return img_path, txt_path
