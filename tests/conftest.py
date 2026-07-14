"""Shared fixtures for v4 PMC tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Add v4 root to sys.path so "src.pmc" resolves to v4's src
_V4_ROOT = str(Path(__file__).resolve().parent.parent)
if _V4_ROOT not in sys.path:
    sys.path.insert(0, _V4_ROOT)


@pytest.fixture
def synth_data():
    """
    Synthetic cross-modal data with injected modality gap.

    Returns dict with keys: image_emb, text_emb, gap_true, N, d
    """
    rng = np.random.default_rng(42)
    N, d = 500, 64

    # Image embeddings: random unit vectors
    image_raw = rng.standard_normal((N, d)).astype(np.float32)
    image_emb = image_raw / np.linalg.norm(image_raw, axis=1, keepdims=True)

    # Text embeddings: shift image by a gap vector, add noise, renormalize
    gap_direction = rng.standard_normal(d).astype(np.float32)
    gap_direction /= np.linalg.norm(gap_direction)
    gap_magnitude = 0.3  # meaningful but not extreme gap
    gap_true = gap_magnitude * gap_direction

    text_raw = image_emb + gap_true[np.newaxis, :] + 0.1 * rng.standard_normal((N, d)).astype(np.float32)
    text_emb = text_raw / np.linalg.norm(text_raw, axis=1, keepdims=True)

    return {
        "image_emb": image_emb.astype(np.float32),
        "text_emb": text_emb.astype(np.float32),
        "gap_true": gap_true,
        "N": N,
        "d": d,
    }
