"""
runtime/config.py — Paths and constants for PMC cross-modal experiments.

Supports:
  - tracked defaults in config/paths.yaml
  - optional local overrides in config/paths.local.yaml
  - ad-hoc overrides via PMC_PATHS_YAML or PMC_* env vars
"""

from __future__ import annotations

import os
import shutil
import sys as _sys

if _sys.stdout.encoding and _sys.stdout.encoding.lower() != "utf-8":
    try:
        _sys.stdout.reconfigure(encoding="utf-8")
        _sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

V4_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = V4_ROOT / "config"
DEFAULT_PATHS_YAML = CONFIG_DIR / "paths.yaml"
LOCAL_PATHS_YAML = CONFIG_DIR / "paths.local.yaml"


def _resolve_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML config: {path}")
    return data


def _merge_path_settings(paths_yaml: Optional[Path] = None) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    for candidate in (DEFAULT_PATHS_YAML, LOCAL_PATHS_YAML):
        settings.update(_read_yaml_mapping(candidate))
    env_yaml = os.environ.get("PMC_PATHS_YAML")
    if env_yaml:
        settings.update(_read_yaml_mapping(Path(env_yaml).expanduser()))
    if paths_yaml is not None:
        settings.update(_read_yaml_mapping(Path(paths_yaml).expanduser()))
    return settings


def _resolve_path_value(value: Any, *, base_dir: Path) -> Optional[Path]:
    if value in (None, "", False):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _env_key(name: str) -> str:
    return f"PMC_{name.upper()}"


def _setting(
    settings: dict[str, Any],
    name: str,
    default: Any = None,
) -> Any:
    return os.environ.get(_env_key(name), settings.get(name, default))


@dataclass
class V4Config:
    seed: int = 42
    nlist: int = 64

    # Device & encoder settings (for encode.py)
    device: str = field(default_factory=_resolve_device)
    imagebind_dtype: str = "float16"
    image_batch_size: int = 8
    text_batch_size: int = 128
    audio_batch_size: int = 4

    # Config metadata
    root_dir: Path = V4_ROOT
    config_dir: Path = CONFIG_DIR
    paths_yaml: Path = DEFAULT_PATHS_YAML
    local_paths_yaml: Path = LOCAL_PATHS_YAML

    # Paths
    data_dir: Path = field(default_factory=lambda: V4_ROOT / "data")
    results_dir: Path = field(default_factory=lambda: V4_ROOT / "results")
    features_dir: Path = field(default_factory=lambda: V4_ROOT / "data" / "features")
    audiocaps_dir: Path = field(default_factory=lambda: V4_ROOT / "data" / "raw" / "audiocaps")
    audiocaps_metadata_csv: Path = field(
        default_factory=lambda: V4_ROOT / "data" / "raw" / "audiocaps" / "test.csv"
    )
    hf_home: Path = field(default_factory=lambda: V4_ROOT / ".hf-cache")
    yt_dlp_path: Optional[str] = None

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    def resolve_yt_dlp(self) -> Optional[str]:
        candidate = self.yt_dlp_path or "yt-dlp"
        if Path(candidate).expanduser().exists():
            return str(Path(candidate).expanduser())
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved

        exe_dir = Path(_sys.executable).parent
        names = [candidate]
        if not str(candidate).endswith(".exe"):
            names.append(f"{candidate}.exe")
        for name in names:
            sibling = exe_dir / name
            if sibling.exists():
                return str(sibling)
        return None

    def require_yt_dlp(self) -> str:
        resolved = self.resolve_yt_dlp()
        if resolved is not None:
            return resolved
        raise FileNotFoundError(
            "yt-dlp executable not found.\n"
            f"Set 'yt_dlp_path' in {self.paths_yaml} or config/paths.local.yaml,\n"
            "or install yt-dlp so it is available on PATH."
        )


def load_runtime_config(paths_yaml: Optional[Path] = None) -> V4Config:
    settings = _merge_path_settings(paths_yaml)

    data_dir = _resolve_path_value(_setting(settings, "data_dir", "data"), base_dir=V4_ROOT)
    results_dir = _resolve_path_value(_setting(settings, "results_dir", "results"), base_dir=V4_ROOT)
    features_dir = _resolve_path_value(
        _setting(settings, "features_dir", str(Path("data") / "features")),
        base_dir=V4_ROOT,
    )
    audiocaps_dir = _resolve_path_value(
        _setting(settings, "audiocaps_dir", str(Path("data") / "raw" / "audiocaps")),
        base_dir=V4_ROOT,
    )
    audiocaps_metadata_csv = _resolve_path_value(
        _setting(settings, "audiocaps_metadata_csv", str(Path("data") / "raw" / "audiocaps" / "test.csv")),
        base_dir=V4_ROOT,
    )
    hf_home = _resolve_path_value(
        _setting(settings, "hf_home", ".hf-cache"),
        base_dir=V4_ROOT,
    )
    yt_dlp_path = _setting(settings, "yt_dlp_path", None)

    return V4Config(
        data_dir=data_dir or (V4_ROOT / "data"),
        results_dir=results_dir or (V4_ROOT / "results"),
        features_dir=features_dir or (V4_ROOT / "data" / "features"),
        audiocaps_dir=audiocaps_dir or (V4_ROOT / "data" / "raw" / "audiocaps"),
        audiocaps_metadata_csv=audiocaps_metadata_csv or (V4_ROOT / "data" / "raw" / "audiocaps" / "test.csv"),
        hf_home=hf_home or (V4_ROOT / ".hf-cache"),
        yt_dlp_path=str(yt_dlp_path) if yt_dlp_path not in (None, "") else None,
    )


CFG = load_runtime_config()
