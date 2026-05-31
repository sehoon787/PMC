"""CLIP encoder wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np


class CLIPEncoder:
    """CLIP encoder via sentence-transformers."""

    _EMBED_DIMS: dict[str, int] = {
        "clip-ViT-B-32": 512,
        "clip-ViT-L-14": 768,
    }

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self._model_name = model_name
        self._model = None
        self.embed_dim: int = self._EMBED_DIMS.get(model_name, 512)

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed.\n"
                "Install it with:\n"
                "  pip install sentence-transformers\n"
            )
        self._model = SentenceTransformer(self._model_name)

    def encode_image(
        self,
        paths: List[Union[str, Path]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        self._load()
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required for image encoding: pip install Pillow")

        images = [Image.open(str(path)).convert("RGB") for path in paths]
        kwargs = dict(convert_to_numpy=True, show_progress_bar=False)
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        emb = self._model.encode(images, **kwargs)
        return np.asarray(emb, dtype=np.float32)

    def encode_text(
        self,
        captions: List[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        self._load()
        kwargs = dict(convert_to_numpy=True, show_progress_bar=False)
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        emb = self._model.encode(captions, **kwargs)
        return np.asarray(emb, dtype=np.float32)
