"""
extract_clip_features.py -- Extract CLIP features for PMC experiments.

Extracts image and text embeddings using CLIPEncoder (sentence-transformers)
and saves them as .npy files in data/features/.

Usage:
    python scripts/features/16_extract_features.py --backbone clip-l --dataset mscoco
    python scripts/features/16_extract_features.py --backbone clip --dataset mscoco
    python scripts/features/16_extract_features.py --backbone clip-l --dataset flickr30k

    # Use existing raw data from a different directory:
    python scripts/features/16_extract_features.py --backbone clip-l --dataset mscoco \
        --raw-dir /path/to/mscoco_karpathy/

Output: data/features/{dataset}_{backbone}_{modality}_seed{seed}.npy
"""

from __future__ import annotations

import argparse
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

from src.features.jobs import FeatureExtractionSpec, run_clip_feature_extraction

SEED = 42

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CLIP features for PMC")
    parser.add_argument("--backbone", choices=["clip", "clip-l"], default="clip-l")
    parser.add_argument("--dataset", choices=["mscoco", "flickr30k"], default="mscoco")
    parser.add_argument("--raw-dir", type=str, default=None,
                        help="Override raw data directory (default: data/raw/<dataset>/)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true", help="Re-extract even if cached")
    args = parser.parse_args()

    run_clip_feature_extraction(
        FeatureExtractionSpec(
            backbone=args.backbone,
            dataset=args.dataset,
            raw_dir=Path(args.raw_dir) if args.raw_dir else None,
            seed=args.seed,
            force=args.force,
        )
    )


if __name__ == "__main__":
    main()
