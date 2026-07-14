"""CLAP encoder wrapper (laion/clap-htsat-fused via HuggingFace transformers)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np

from src.runtime.config import CFG


class ClapEncoder:
    """Thin wrapper around CLAP (laion/clap-htsat-fused) for audio-text feature extraction.

    Produces L2-normalised 512-d embeddings for both audio and text modalities.
    Uses the HuggingFace ``transformers`` ClapModel/ClapProcessor — no extra
    dependencies beyond transformers + librosa (for audio loading).
    """

    embed_dim: int = 512
    MODEL_ID: str = "laion/clap-htsat-fused"

    def __init__(
        self,
        device: str = CFG.device,
        model_id: str = MODEL_ID,
    ):
        self.device = device
        self.model_id = model_id
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError:
            raise ImportError(
                "transformers is required for CLAP.\n"
                "Install with: pip install transformers\n"
            )

        import torch

        print(f"  [CLAP] Loading model {self.model_id} ...")
        processor = ClapProcessor.from_pretrained(self.model_id)
        model = ClapModel.from_pretrained(self.model_id)
        model.eval()
        model = model.to(self.device)
        self._model = model
        self._processor = processor
        self._torch = torch
        print(f"  [CLAP] Model loaded on {self.device}  (embed_dim={self.embed_dim})")

    def encode_audio(
        self,
        paths: List[Union[str, Path]],
        batch_size: int = CFG.audio_batch_size,
        sampling_rate: int = 48000,
    ) -> np.ndarray:
        """Encode audio files to 512-d normalised embeddings."""
        self._load()
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa is required for CLAP audio loading.\n"
                "Install with: pip install librosa\n"
            )
        from tqdm import tqdm

        all_embs: list[np.ndarray] = []
        for start in tqdm(range(0, len(paths), batch_size), desc="CLAP audio"):
            batch_paths = paths[start : start + batch_size]
            waveforms = []
            for p in batch_paths:
                wav, _ = librosa.load(str(p), sr=sampling_rate, mono=True)
                waveforms.append(wav)

            inputs = self._processor(
                audios=waveforms,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with self._torch.no_grad():
                out = self._model.get_audio_features(**inputs)
            # out shape: (batch, 512) — already L2-normalised by ClapModel
            all_embs.append(out.float().cpu().numpy())

        return np.vstack(all_embs).astype(np.float32)

    def encode_text(
        self,
        captions: List[str],
        batch_size: int = CFG.text_batch_size,
    ) -> np.ndarray:
        """Encode text captions to 512-d normalised embeddings."""
        self._load()
        from tqdm import tqdm

        all_embs: list[np.ndarray] = []
        for start in tqdm(range(0, len(captions), batch_size), desc="CLAP text"):
            batch = captions[start : start + batch_size]
            inputs = self._processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with self._torch.no_grad():
                out = self._model.get_text_features(**inputs)
            all_embs.append(out.float().cpu().numpy())

        return np.vstack(all_embs).astype(np.float32)
