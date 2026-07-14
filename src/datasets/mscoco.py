"""MSCOCO dataset loaders."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import List

from .items import MSCOCOItem

log = logging.getLogger(__name__)

_MSCOCO_VAL_URL = (
    "http://images.cocodataset.org/zips/val2017.zip"
)
_MSCOCO_ANN_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)


def download_mscoco_val5k(target_dir: Path) -> List[MSCOCOItem]:
    """
    Load MSCOCO 5K val items from *target_dir*.

    If the expected files are missing, print instructions and raise FileNotFoundError.
    We do NOT auto-download (the zip files are large: ~800 MB + ~240 MB).

    Expected layout:
        target_dir/
            val2017/           <- COCO val images (5K JPEG files)
            annotations/
                captions_val2017.json

    Returns a list of MSCOCOItem sorted by image_id.
    """
    target_dir = Path(target_dir)
    img_dir = target_dir / "val2017"
    ann_file = target_dir / "annotations" / "captions_val2017.json"

    if not img_dir.is_dir() or not ann_file.is_file():
        _print_mscoco_instructions(target_dir)
        raise FileNotFoundError(
            f"MSCOCO val2017 not found in {target_dir}. "
            "See instructions printed above."
        )

    with open(ann_file, "r") as f:
        ann_data = json.load(f)

    # Build image_id -> image_path mapping
    id_to_path: dict[int, Path] = {}
    for img_info in ann_data["images"]:
        iid = img_info["id"]
        fname = img_info["file_name"]
        id_to_path[iid] = img_dir / fname

    # Group captions by image_id
    id_to_captions: dict[int, list[str]] = {}
    for ann in ann_data["annotations"]:
        iid = ann["image_id"]
        id_to_captions.setdefault(iid, []).append(ann["caption"])

    items: list[MSCOCOItem] = []
    for iid in sorted(id_to_path.keys()):
        items.append(
            MSCOCOItem(
                image_id=iid,
                image_path=id_to_path[iid],
                captions=id_to_captions.get(iid, []),
            )
        )

    return items


def _print_mscoco_instructions(target_dir: Path) -> None:
    print(
        "\n"
        "=" * 70 + "\n"
        "MSCOCO 5K val data not found.\n"
        f"Expected directory: {target_dir}\n\n"
        "Download and unpack manually:\n\n"
        f"  wget {_MSCOCO_VAL_URL}\n"
        f"  wget {_MSCOCO_ANN_URL}\n"
        f"  cd {target_dir}\n"
        "  unzip val2017.zip\n"
        "  unzip annotations_trainval2017.zip\n\n"
        "After unpacking you should have:\n"
        f"  {target_dir}/val2017/             (5000 .jpg files)\n"
        f"  {target_dir}/annotations/captions_val2017.json\n"
        "=" * 70 + "\n"
    )


# ---------------------------------------------------------------------------
# MSCOCO Karpathy val5k loader (auto-download)
# ---------------------------------------------------------------------------

_KARPATHY_ZIP_URL = (
    "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
)
_COCO_IMAGE_URL_TEMPLATE = (
    "http://images.cocodataset.org/val2014/COCO_val2014_{cocoid:012d}.jpg"
)


def download_mscoco_karpathy_val5k(
    target_dir: Path,
    download_images: bool = True,
    max_workers: int = 8,
) -> List[MSCOCOItem]:
    """
    Load the MSCOCO Karpathy test split (~5K images) with auto-download.

    Parameters
    ----------
    target_dir:       Root directory for data storage.
    download_images:  If True, download any missing JPEG images.
    max_workers:      Thread-pool workers for parallel image downloads.

    Returns
    -------
    List of MSCOCOItem sorted by image_id.  Items whose image download
    failed are still included (image_path may not exist on disk).
    """
    target_dir = Path(target_dir)
    karpathy_dir = target_dir / "karpathy"
    karpathy_dir.mkdir(parents=True, exist_ok=True)

    ann_file = karpathy_dir / "dataset_coco.json"

    # --- Step 1: Download and unzip Karpathy captions if needed ---
    if not ann_file.exists():
        zip_path = karpathy_dir / "caption_datasets.zip"
        if not zip_path.exists():
            print(f"[data] Downloading Karpathy caption split from {_KARPATHY_ZIP_URL} ...")
            urllib.request.urlretrieve(_KARPATHY_ZIP_URL, str(zip_path))
            print(f"[data] Downloaded -> {zip_path}")
        print(f"[data] Unzipping {zip_path} ...")
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(karpathy_dir))
        print(f"[data] Unzipped to {karpathy_dir}")

    # --- Step 2: Parse annotation file ---
    with open(ann_file, "r") as f:
        dataset = json.load(f)

    img_dir = target_dir / "val2014"
    img_dir.mkdir(parents=True, exist_ok=True)

    items: List[MSCOCOItem] = []
    for img in dataset["images"]:
        if img.get("split") != "test":
            continue
        cocoid: int = int(img["cocoid"])
        image_path = img_dir / img["filename"]
        captions: List[str] = [s["raw"] for s in img.get("sentences", [])]
        items.append(MSCOCOItem(image_id=cocoid, image_path=image_path, captions=captions))

    items.sort(key=lambda x: x.image_id)
    print(f"[data] Karpathy test split: {len(items)} images found in annotation.")

    # --- Step 3: Download missing images (optional) ---
    if download_images:
        missing = [it for it in items if not it.image_path.exists()]
        if missing:
            print(f"[data] Downloading {len(missing)} missing images "
                  f"(max_workers={max_workers}) ...")
            try:
                from tqdm import tqdm
                _tqdm = tqdm
            except ImportError:
                _tqdm = None

            failed: List[int] = []

            def _download_one(item: MSCOCOItem) -> bool:
                url = _COCO_IMAGE_URL_TEMPLATE.format(cocoid=item.image_id)
                try:
                    urllib.request.urlretrieve(url, str(item.image_path))
                    return True
                except Exception as exc:
                    log.warning("Failed to download image_id=%d: %s", item.image_id, exc)
                    failed.append(item.image_id)
                    return False

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_download_one, it): it for it in missing}
                iterable = concurrent.futures.as_completed(futures)
                if _tqdm is not None:
                    iterable = _tqdm(iterable, total=len(missing), desc="Downloading images")
                for _ in iterable:
                    pass

            if failed:
                print(f"[data] WARNING: {len(failed)} images failed to download "
                      f"(first 5: {failed[:5]})")
        else:
            print("[data] All images already present — skipping download.")

    return items
