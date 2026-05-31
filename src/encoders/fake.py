"""Deterministic fake encoder used by tests and dry-runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Union

import numpy as np

from src.runtime.config import CFG


class FakeEncoder:
    """Deterministic random encoder for tests and dry-runs."""

    embed_dim: int = 64

    def __init__(self, d: int = 64, seed: int = CFG.seed):
        self.d = d
        self.seed = seed
        self.embed_dim = d

    def _hash_seed(self, key: str) -> int:
        digest = hashlib.md5(key.encode()).hexdigest()
        return int(digest, 16) % (2**31)

    def _make_vector(self, key: str) -> np.ndarray:
        item_seed = (self.seed + self._hash_seed(key)) % (2**31)
        rng = np.random.default_rng(item_seed)
        vector = rng.standard_normal(self.d).astype(np.float32)
        vector /= np.linalg.norm(vector) + 1e-10
        return vector

    def encode_image(self, paths: List[Union[str, Path]]) -> np.ndarray:
        return np.stack([self._make_vector(f"image:{str(path)}") for path in paths])

    def encode_text(self, captions: List[str]) -> np.ndarray:
        return np.stack([self._make_vector(f"text:{caption}") for caption in captions])

    def encode_audio(self, paths: List[Union[str, Path]]) -> np.ndarray:
        return np.stack([self._make_vector(f"audio:{str(path)}") for path in paths])
