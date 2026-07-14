"""BigANN binary file readers for large-scale vector datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_fbin(path: str | Path) -> np.ndarray:
    """Read BigANN `.fbin` data as a `(n, d)` float32 array."""
    with open(path, "rb") as f:
        header = f.read(8)
        n, d = np.frombuffer(header, dtype=np.uint32)
        n, d = int(n), int(d)
        data = f.read(n * d * 4)
    return np.frombuffer(data, dtype=np.float32).reshape(n, d)
