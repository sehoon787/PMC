"""ImageBind encoder wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np

from src.runtime.config import CFG

try:
    from PIL import ImageFile as _PILImageFile

    _PILImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:
    pass


class ImageBindEncoder:
    """Thin wrapper around ImageBind-Huge for feature extraction."""

    embed_dim: int = 1024

    def __init__(
        self,
        device: str = CFG.device,
        dtype: str = CFG.imagebind_dtype,
    ):
        self.device = device
        self.dtype = dtype
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import imagebind.models.imagebind_model as ib
            from imagebind.models.imagebind_model import ModalityType
        except ImportError:
            raise ImportError(
                "ImageBind is not installed.\n"
                "Install it with:\n"
                "  git clone https://github.com/facebookresearch/ImageBind.git\n"
                "  cd ImageBind && pip install -e .\n"
            )

        import torch

        model = ib.imagebind_huge(pretrained=True)
        model.eval()
        torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32
        model = model.to(self.device).to(torch_dtype)
        self._model = model
        self._ModalityType = ModalityType

    def encode_image(
        self,
        paths: List[Union[str, Path]],
        batch_size: int = CFG.image_batch_size,
    ) -> np.ndarray:
        self._load()
        return self._encode_batched(paths, self._ModalityType.VISION, batch_size)

    def encode_text(
        self,
        captions: List[str],
        batch_size: int = CFG.text_batch_size,
    ) -> np.ndarray:
        self._load()
        return self._encode_batched(captions, self._ModalityType.TEXT, batch_size)

    def encode_audio(
        self,
        paths: List[Union[str, Path]],
        batch_size: int = CFG.audio_batch_size,
    ) -> np.ndarray:
        self._load()
        return self._encode_batched(paths, self._ModalityType.AUDIO, batch_size)

    def _encode_batched(self, items, modality_type, batch_size: int) -> np.ndarray:
        import torch
        from tqdm import tqdm

        try:
            import imagebind.data as ib_data
        except ImportError:
            raise ImportError("imagebind.data not found — is ImageBind installed?")

        all_embs = []
        for start in tqdm(range(0, len(items), batch_size), desc=f"Encoding {modality_type}"):
            batch = items[start : start + batch_size]
            inputs = _prepare_imagebind_inputs(batch, modality_type, self.device, ib_data)
            torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32
            inputs = {key: value.to(torch_dtype) if value.is_floating_point() else value for key, value in inputs.items()}
            with torch.no_grad():
                out = self._model(inputs)
            all_embs.append(out[modality_type].float().cpu().numpy())

        return np.vstack(all_embs).astype(np.float32)


def _prepare_imagebind_inputs(batch, modality_type, device, ib_data):
    mt_name = str(modality_type)
    if "VISION" in mt_name or "vision" in mt_name.lower():
        tensor = ib_data.load_and_transform_vision_data([str(path) for path in batch], device)
    elif "TEXT" in mt_name or "text" in mt_name.lower():
        tensor = ib_data.load_and_transform_text(batch, device)
    elif "AUDIO" in mt_name or "audio" in mt_name.lower():
        tensor = ib_data.load_and_transform_audio_data([str(path) for path in batch], device)
    else:
        raise ValueError(f"Unsupported modality type: {modality_type}")
    return {modality_type: tensor}
