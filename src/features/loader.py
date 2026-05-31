"""Feature loading utilities for PQ/OPQ experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.utils import l2_normalize


SEED = 42


def load_npy(path: Path) -> Optional[np.ndarray]:
    """Load a .npy file, cast to float32, and L2-normalize. Returns None if missing."""
    if not path.exists():
        print(f"[WARN] Missing: {path}", flush=True)
        return None
    arr = np.load(str(path)).astype(np.float32)
    return l2_normalize(arr)


def load_features(config: str, feat_dir: Path, seed: int = SEED) -> Optional[Dict[str, np.ndarray]]:
    """Return {'audio': ..., 'text': ...} for the requested config, or None.

    Supported configs: audiocaps_only, clotho_all, clotho_eval_only, combined.
    """
    def stem(modality: str, tag: str) -> Path:
        return feat_dir / f"{tag}_imagebind_{modality}_seed{seed}.npy"

    if config == "audiocaps_only":
        aud = load_npy(stem("audio", "audiocaps_test"))
        txt = load_npy(stem("text", "audiocaps_test"))

    elif config == "clotho_all":
        aud = load_npy(stem("audio", "clotho_all"))
        txt = load_npy(stem("text", "clotho_all"))

    elif config == "clotho_eval_only":
        aud = load_npy(stem("audio", "clotho_eval"))
        txt = load_npy(stem("text", "clotho_eval"))

    elif config == "combined":
        ac_aud = load_npy(stem("audio", "audiocaps_test"))
        ac_txt = load_npy(stem("text", "audiocaps_test"))
        cl_aud = load_npy(stem("audio", "clotho_all"))
        cl_txt = load_npy(stem("text", "clotho_all"))
        if any(x is None for x in [ac_aud, ac_txt, cl_aud, cl_txt]):
            print("[ERROR] combined config requires both audiocaps_test and clotho_all features.", flush=True)
            return None
        aud = l2_normalize(np.vstack([ac_aud, cl_aud]))
        txt = l2_normalize(np.vstack([ac_txt, cl_txt]))

    else:
        print(f"[ERROR] Unknown config: {config}", flush=True)
        return None

    if aud is None or txt is None:
        print(f"[ERROR] Feature files missing for config '{config}'.", flush=True)
        return None

    return {"audio": aud, "text": txt}
