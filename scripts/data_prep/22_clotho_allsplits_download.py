"""
22_clotho_allsplits_download.py -- Download Clotho v2 dev/val from Zenodo and
extract ImageBind features for all splits (dev + val + eval).

Outputs:
  data/features/clotho_{dev,val,eval,all}_imagebind_{audio,text}_seed42.npy
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_V4_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "src").is_dir() and (parent / "config").is_dir()
)
if str(_V4_ROOT) not in sys.path:
    sys.path.insert(0, str(_V4_ROOT))

import numpy as np

from src.runtime.config import CFG
from src.datasets.clotho import load_clotho_evaluation
from src.encoders.imagebind import ImageBindEncoder

SEED = 42
ZENODO_BASE = "https://zenodo.org/records/4783391/files"
DOWNLOADS = [
    "clotho_audio_development.7z",
    "clotho_audio_validation.7z",
    "clotho_captions_development.csv",
    "clotho_captions_validation.csv",
]


def _progress_callback(filename: str):
    last_pct = [-1]

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        pct = int(block_num * block_size * 100 / total_size)
        pct = min(pct, 100)
        if pct // 10 != last_pct[0] // 10 or pct == 100:
            print(f"  {filename}: {pct}%")
            last_pct[0] = pct

    return reporthook


# ---------------------------------------------------------------------------

def download_clotho_files(raw_clotho_dir: Path) -> None:
    raw_clotho_dir.mkdir(parents=True, exist_ok=True)
    for filename in DOWNLOADS:
        dest = raw_clotho_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [skip] {filename} already exists ({dest.stat().st_size:,} bytes)")
            continue
        url = f"{ZENODO_BASE}/{filename}"
        print(f"  Downloading {filename} ...")
        urllib.request.urlretrieve(url, dest, reporthook=_progress_callback(filename))
        print(f"  Saved {dest.stat().st_size:,} bytes -> {dest}")


# ---------------------------------------------------------------------------
# Step 2: Extract 7z
# ---------------------------------------------------------------------------

def _find_wavs(directory: Path) -> list[Path]:
    return list(directory.rglob("*.wav"))


def extract_7z(archive: Path, extract_dir: Path) -> Path:
    """Extract archive to extract_dir; return the actual audio directory."""
    existing_wavs = _find_wavs(extract_dir)
    if existing_wavs:
        print(f"  [skip] {extract_dir.name}/ already has {len(existing_wavs)} WAV files")
        # Detect if WAVs are in a subdirectory created by 7z
        subdirs = {w.parent for w in existing_wavs}
        if len(subdirs) == 1:
            return subdirs.pop()
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {archive.name} -> {extract_dir} ...")
    try:
        import py7zr
    except ImportError:
        raise ImportError("py7zr is required: pip install py7zr")

    with py7zr.SevenZipFile(str(archive), "r") as z:
        z.extractall(path=str(extract_dir))

    wavs = _find_wavs(extract_dir)
    print(f"  Extracted {len(wavs)} WAV files")

    # Handle case where 7z creates a sub-folder
    subdirs = {w.parent for w in wavs}
    if len(subdirs) == 1:
        sub = subdirs.pop()
        if sub != extract_dir:
            print(f"  Audio files in sub-directory: {sub.name}/")
            return sub
    return extract_dir


def extract_split_features(
    split_tag: str,
    captions_csv: Path,
    audio_dir: Path,
    encoder: ImageBindEncoder,
    feat_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    aud_path = feat_dir / f"clotho_{split_tag}_imagebind_audio_seed{SEED}.npy"
    txt_path = feat_dir / f"clotho_{split_tag}_imagebind_text_seed{SEED}.npy"

    if aud_path.exists() and txt_path.exists():
        aud_emb = np.load(str(aud_path))
        txt_emb = np.load(str(txt_path))
        print(f"  [{split_tag}] Features exist: audio={aud_emb.shape}, text={txt_emb.shape}")
        return aud_emb, txt_emb

    print(f"  [{split_tag}] Loading captions from {captions_csv.name} ...")
    items = load_clotho_evaluation(
        captions_csv=captions_csv,
        audio_dir=audio_dir,
        require_audio=True,
        min_audio_bytes=4096,
    )
    print(f"  [{split_tag}] {len(items)} items with valid audio")
    if len(items) == 0:
        raise RuntimeError(f"[{split_tag}] No items found. Check audio dir: {audio_dir}")

    audio_paths = [it.audio_path for it in items]
    captions = [it.captions[0] for it in items]

    if not aud_path.exists():
        print(f"  [{split_tag}] Encoding {len(audio_paths)} audio clips (progress every 100) ...")
        aud_emb = encoder.encode_audio(audio_paths)
        np.save(str(aud_path), aud_emb)
        print(f"  [{split_tag}] audio -> {aud_path.name} {aud_emb.shape}")
    else:
        aud_emb = np.load(str(aud_path))

    if not txt_path.exists():
        print(f"  [{split_tag}] Encoding {len(captions)} captions ...")
        txt_emb = encoder.encode_text(captions)
        np.save(str(txt_path), txt_emb)
        print(f"  [{split_tag}] text  -> {txt_path.name} {txt_emb.shape}")
    else:
        txt_emb = np.load(str(txt_path))

    return aud_emb, txt_emb


def main() -> None:
    np.random.seed(SEED)
    raw_clotho_dir = CFG.raw_dir / "clotho"
    feat_dir = CFG.features_dir
    feat_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Download Clotho v2 dev/val from Zenodo ===")
    download_clotho_files(raw_clotho_dir)

    print("\n=== Step 2: Extract 7z archives ===")
    dev_audio_dir = extract_7z(
        archive=raw_clotho_dir / "clotho_audio_development.7z",
        extract_dir=raw_clotho_dir / "development",
    )
    val_audio_dir = extract_7z(
        archive=raw_clotho_dir / "clotho_audio_validation.7z",
        extract_dir=raw_clotho_dir / "validation",
    )
    # Eval split (audio + captions CSV) is assumed already present from script 21
    # or earlier manual download; only dev/val are fetched from Zenodo above.
    eval_audio_dir = raw_clotho_dir

    print("\n=== Step 3: Extract ImageBind features (CPU -- may take ~5-10 min/split) ===")
    encoder = ImageBindEncoder(device="cpu", dtype="float32")

    splits = [
        ("dev",  raw_clotho_dir / "clotho_captions_development.csv",  dev_audio_dir),
        ("val",  raw_clotho_dir / "clotho_captions_validation.csv",   val_audio_dir),
        ("eval", raw_clotho_dir / "clotho_captions_evaluation.csv",   eval_audio_dir),
    ]

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_tag, csv_path, audio_dir in splits:
        print(f"\n  --- {split_tag} ---")
        aud_emb, txt_emb = extract_split_features(
            split_tag=split_tag,
            captions_csv=csv_path,
            audio_dir=audio_dir,
            encoder=encoder,
            feat_dir=feat_dir,
        )
        arrays[split_tag] = (aud_emb, txt_emb)

    print("\n=== Step 4: Build combined clotho_all arrays ===")
    all_aud_path = feat_dir / f"clotho_all_imagebind_audio_seed{SEED}.npy"
    all_txt_path = feat_dir / f"clotho_all_imagebind_text_seed{SEED}.npy"

    combined_audio = np.vstack([arrays["dev"][0], arrays["val"][0], arrays["eval"][0]])
    combined_text  = np.vstack([arrays["dev"][1], arrays["val"][1], arrays["eval"][1]])

    np.save(str(all_aud_path), combined_audio)
    np.save(str(all_txt_path), combined_text)

    print(f"  clotho_all audio: {combined_audio.shape} -> {all_aud_path.name}")
    print(f"  clotho_all text:  {combined_text.shape}  -> {all_txt_path.name}")
    print(f"\n  Total clips: {combined_audio.shape[0]}")
    print("\nDONE. Clotho feature files:")
    for f in sorted(feat_dir.glob("clotho_*imagebind*.npy")):
        arr = np.load(str(f))
        print(f"  {f.name}: {arr.shape}")


if __name__ == "__main__":
    main()
