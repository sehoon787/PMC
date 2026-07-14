"""Download helpers for external experiment datasets."""

from __future__ import annotations

import csv
import subprocess
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MSCOCO_IMG_URL = "http://images.cocodataset.org/zips/val2017.zip"
MSCOCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
AUDIOCAPS_CSV_URL = "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/test.csv"


def _download_with_progress(url: str, dest: Path) -> None:
    """Download a URL with a simple progress display."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    print(f"  Downloading {url} ...")

    def _hook(count, block, total):
        pct = count * block * 100 / total if total > 0 else 0
        print(f"\r  {pct:.1f}% ({count * block / 1e6:.1f}/{total / 1e6:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, str(tmp), reporthook=_hook)
    print()
    tmp.rename(dest)


def ensure_mscoco_val2017(target_dir: Path) -> None:
    """Download and extract MSCOCO val2017 images + annotations if missing."""
    target_dir = Path(target_dir)
    img_dir = target_dir / "val2017"
    ann_file = target_dir / "annotations" / "captions_val2017.json"

    if img_dir.is_dir() and ann_file.is_file():
        print("[mscoco] Already present.")
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.is_dir():
        zip_path = target_dir / "val2017.zip"
        _download_with_progress(MSCOCO_IMG_URL, zip_path)
        print("  Extracting images ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        zip_path.unlink()

    if not ann_file.is_file():
        zip_path = target_dir / "annotations_trainval2017.zip"
        _download_with_progress(MSCOCO_ANN_URL, zip_path)
        print("  Extracting annotations ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if "captions_val2017" in name:
                    zf.extract(name, target_dir)
        zip_path.unlink()

    print("[mscoco] Download complete.")


def download_audiocaps_metadata(target_dir: Path) -> Path:
    """Download AudioCaps test.csv if not present."""
    csv_path = Path(target_dir) / "test.csv"
    if csv_path.exists():
        print(f"  Metadata already exists: {csv_path}")
        return csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    print("  Downloading AudioCaps test.csv ...")
    urllib.request.urlretrieve(AUDIOCAPS_CSV_URL, str(csv_path))
    print(f"  Saved: {csv_path}")
    return csv_path


def parse_audiocaps_metadata(csv_path: Path) -> list[dict]:
    """Parse AudioCaps test.csv into unique clip descriptors."""
    seen: set[int] = set()
    items: list[dict] = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            audiocap_id = int(row["audiocap_id"])
            if audiocap_id in seen:
                continue
            seen.add(audiocap_id)
            items.append(
                {
                    "audiocap_id": audiocap_id,
                    "youtube_id": row["youtube_id"],
                    "start_time": int(row["start_time"]),
                }
            )
    return items


def download_audiocaps_clip(item: dict, target_dir: Path, yt_dlp: str) -> tuple[int, bool, str]:
    """Download one 10-second AudioCaps clip from YouTube."""
    aid = item["audiocap_id"]
    wav_path = Path(target_dir) / f"{aid}.wav"
    if wav_path.exists() and wav_path.stat().st_size > 4096:
        return aid, True, "cached"

    yt_url = f"https://www.youtube.com/watch?v={item['youtube_id']}"
    start = item["start_time"]
    end = start + 10

    cmd = [
        yt_dlp,
        "--no-warnings",
        "-q",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--postprocessor-args",
        f"ffmpeg:-ss {start} -to {end} -ac 1 -ar 16000",
        "-o",
        str(wav_path),
        yt_url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if wav_path.exists() and wav_path.stat().st_size > 4096:
            return aid, True, "downloaded"
        return aid, False, result.stderr[:200] if result.stderr else "empty file"
    except subprocess.TimeoutExpired:
        return aid, False, "timeout"
    except Exception as exc:
        return aid, False, str(exc)[:200]


def download_audiocaps_clips(
    *,
    items: list[dict],
    target_dir: Path,
    yt_dlp: str,
    max_workers: int = 4,
) -> tuple[int, int]:
    """Download missing AudioCaps clips and return (success_count, failed_count)."""
    target_dir = Path(target_dir)
    existing = sum(1 for item in items if (target_dir / f"{item['audiocap_id']}.wav").exists())
    print(f"  Already downloaded: {existing}/{len(items)}")
    remaining = [
        item
        for item in items
        if not (target_dir / f"{item['audiocap_id']}.wav").exists()
        or (target_dir / f"{item['audiocap_id']}.wav").stat().st_size < 4096
    ]

    if not remaining:
        print("  All clips already downloaded!")
        return existing, 0

    print(f"  Downloading {len(remaining)} remaining clips ({max_workers} threads) ...")
    success = existing
    failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_audiocaps_clip, item, target_dir, yt_dlp): item for item in remaining}
        for i, future in enumerate(as_completed(futures), 1):
            _, ok, _ = future.result()
            if ok:
                success += 1
            else:
                failed += 1
            if i % 50 == 0 or i == len(remaining):
                print(f"  Progress: {i}/{len(remaining)} | OK: {success} | Failed: {failed}")
    return success, failed
