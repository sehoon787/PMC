"""Synthetic paired embeddings used by tests and dry-run evaluation."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def synthetic_dataset(
    n: int = 200,
    d: int = 64,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic paired (text, image) embeddings for dry-run use."""
    rng = np.random.default_rng(seed)
    text_emb = rng.standard_normal((n, d)).astype(np.float32)
    text_emb /= np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-10

    noise = rng.standard_normal((n, d)).astype(np.float32)
    image_emb = 0.7 * text_emb + 0.3 * noise
    image_emb /= np.linalg.norm(image_emb, axis=1, keepdims=True) + 1e-10
    return text_emb, image_emb
