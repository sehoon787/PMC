"""Flickr30K Karpathy split loaders."""

from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path
from typing import List

from .items import MSCOCOItem
from .mscoco import _KARPATHY_ZIP_URL

_FLICKR30K_HF_DATASET = "nlphuji/flickr30k"


def download_flickr30k_test1k(
    target_dir: Path,
    download_images: bool = True,
    max_workers: int = 8,
) -> List[MSCOCOItem]:
    """
    Download Karpathy split for Flickr30K, return 'test' split (1000 images).

    Strategy for images:
    1. Try HuggingFace ``datasets`` library (``nlphuji/flickr30k``).
       This dataset is gated; the user must accept the licence on HF and run
       ``huggingface-cli login`` (or set ``HUGGING_FACE_HUB_TOKEN``) first.
    2. If ``datasets`` is not installed, raise an informative ImportError.
    3. If the HF dataset raises an authentication / access error, fall back to
       "captions only" mode: the function still returns MSCOCOItems but
       ``image_path`` will point to a non-existent file.  The user must then
       place images manually at ``target_dir/flickr30k/images/<filename>``.

    The Karpathy caption JSON (``dataset_flickr30k.json``) is fetched from the
    same Stanford zip already used for MSCOCO:
        https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip

    Parameters
    ----------
    target_dir:      Root directory for data storage.
    download_images: If True, attempt to fetch images via HuggingFace.
    max_workers:     Unused (kept for API symmetry with MSCOCO loader).

    Returns
    -------
    List of MSCOCOItem (image_id = Flickr imgid, captions = 5 raw sentences).
    Items are sorted by image_id.  Image paths may not exist if download was
    skipped or failed.
    """
    target_dir = Path(target_dir)
    flickr_dir = target_dir / "flickr30k"
    img_dir = flickr_dir / "images"
    flickr_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Ensure Karpathy JSON is available ---
    karpathy_dir = target_dir / "karpathy"
    karpathy_dir.mkdir(parents=True, exist_ok=True)
    ann_file = karpathy_dir / "dataset_flickr30k.json"

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

    items: List[MSCOCOItem] = []
    for img in dataset["images"]:
        if img.get("split") != "test":
            continue
        # Flickr30K Karpathy JSON uses 'imgid' for the numeric id and
        # 'filename' for the original Flickr filename (e.g. "12345.jpg").
        imgid: int = int(img.get("imgid", img.get("cocoid", 0)))
        filename: str = img["filename"]
        image_path = img_dir / filename
        captions: List[str] = [s["raw"] for s in img.get("sentences", [])]
        items.append(MSCOCOItem(image_id=imgid, image_path=image_path, captions=captions))

    items.sort(key=lambda x: x.image_id)
    print(f"[data] Flickr30K Karpathy test split: {len(items)} images found in annotation.")

    # --- Step 3: Download images via HuggingFace (optional) ---
    if download_images:
        missing = [it for it in items if not it.image_path.exists()]
        if not missing:
            print("[data] All Flickr30K images already present — skipping download.")
        else:
            print(f"[data] {len(missing)} images missing; attempting HuggingFace download ...")
            _download_flickr30k_images_hf(missing, img_dir)
    return items


def download_flickr30k_full(
    target_dir: Path,
    download_images: bool = True,
    max_workers: int = 8,
) -> List[MSCOCOItem]:
    """
    Download Karpathy split for Flickr30K, return ALL splits (train + val + test, ~31K images).

    Uses the same Karpathy JSON and image directory as ``download_flickr30k_test1k``.
    Images are fetched via HuggingFace ``nlphuji/flickr30k`` (gated dataset; login required).

    Parameters
    ----------
    target_dir:      Root directory for data storage.
    download_images: If True, attempt to fetch images via HuggingFace.
    max_workers:     Unused (kept for API symmetry with MSCOCO loader).

    Returns
    -------
    List of MSCOCOItem (image_id = Flickr imgid, captions = 5 raw sentences).
    Items are sorted by image_id.  Image paths may not exist if download was
    skipped or failed.
    """
    target_dir = Path(target_dir)
    flickr_dir = target_dir / "flickr30k"
    img_dir = flickr_dir / "images"
    flickr_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Ensure Karpathy JSON is available ---
    karpathy_dir = target_dir / "karpathy"
    karpathy_dir.mkdir(parents=True, exist_ok=True)
    ann_file = karpathy_dir / "dataset_flickr30k.json"

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

    # --- Step 2: Parse annotation file (all splits) ---
    with open(ann_file, "r") as f:
        dataset = json.load(f)

    items: List[MSCOCOItem] = []
    for img in dataset["images"]:
        imgid: int = int(img.get("imgid", img.get("cocoid", 0)))
        filename: str = img["filename"]
        image_path = img_dir / filename
        captions: List[str] = [s["raw"] for s in img.get("sentences", [])]
        items.append(MSCOCOItem(image_id=imgid, image_path=image_path, captions=captions))

    items.sort(key=lambda x: x.image_id)
    print(f"[data] Flickr30K full: {len(items)} images found in annotation.")

    # --- Step 3: Download images via HuggingFace (optional) ---
    if download_images:
        missing = [it for it in items if not it.image_path.exists()]
        if not missing:
            print("[data] All Flickr30K images already present — skipping download.")
        else:
            print(f"[data] {len(missing)} images missing; attempting HuggingFace download ...")
            _download_flickr30k_images_hf(missing, img_dir)
    return items


def _download_flickr30k_images_hf(
    missing_items: List[MSCOCOItem],
    img_dir: Path,
) -> None:
    """
    Fetch Flickr30K images from HuggingFace ``nlphuji/flickr30k``.

    Requires:
      - ``pip install datasets``
      - HuggingFace account with licence accepted at
        https://huggingface.co/datasets/nlphuji/flickr30k
      - ``huggingface-cli login``  OR  ``HUGGING_FACE_HUB_TOKEN`` env var set.

    If either requirement is unmet, prints clear instructions and returns
    without raising so the caller can still use caption-only mode.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        print(
            "\n" + "=" * 70 + "\n"
            "Flickr30K image download requires the 'datasets' library.\n\n"
            "  pip install datasets\n\n"
            "After installing, also authenticate with HuggingFace:\n"
            "  huggingface-cli login\n"
            "and accept the dataset licence at:\n"
            f"  https://huggingface.co/datasets/{_FLICKR30K_HF_DATASET}\n"
            "\nAlternatively, place images manually at:\n"
            f"  {img_dir}/<filename>.jpg\n"
            "=" * 70 + "\n"
        )
        return

    # Build a lookup: filename -> image_path
    filename_to_path: dict[str, Path] = {
        it.image_path.name: it.image_path for it in missing_items
    }

    print(f"[data] Loading HuggingFace dataset '{_FLICKR30K_HF_DATASET}' (test split) ...")
    try:
        ds = load_dataset(_FLICKR30K_HF_DATASET, split="test", trust_remote_code=True)
    except Exception as exc:
        _msg = str(exc)
        if "gated" in _msg.lower() or "access" in _msg.lower() or "auth" in _msg.lower() or "401" in _msg:
            print(
                "\n" + "=" * 70 + "\n"
                "HuggingFace authentication / gated-access error:\n"
                f"  {exc}\n\n"
                "Steps to fix:\n"
                "  1. Accept the dataset licence at:\n"
                f"     https://huggingface.co/datasets/{_FLICKR30K_HF_DATASET}\n"
                "  2. Run:  huggingface-cli login\n"
                "     (or set HUGGING_FACE_HUB_TOKEN environment variable)\n\n"
                "Falling back to caption-only mode.\n"
                "Place images manually at:\n"
                f"  {img_dir}/<filename>.jpg\n"
                "=" * 70 + "\n"
            )
        else:
            print(f"[data] WARNING: HuggingFace load failed ({exc}); "
                  "falling back to caption-only mode.")
        return

    saved = 0
    for row in ds:
        fname: str = row.get("filename") or row.get("img_id") or ""
        if not fname:
            continue
        # HF dataset may or may not include the extension
        if not fname.endswith(".jpg"):
            fname = fname + ".jpg"
        if fname not in filename_to_path:
            continue
        out_path = filename_to_path[fname]
        if out_path.exists():
            continue
        pil_img = row.get("image")
        if pil_img is None:
            continue
        pil_img.save(str(out_path), format="JPEG")
        saved += 1

    print(f"[data] Saved {saved}/{len(missing_items)} Flickr30K images to {img_dir}")
    still_missing = sum(1 for it in missing_items if not it.image_path.exists())
    if still_missing:
        print(
            f"[data] WARNING: {still_missing} images still missing after HF download.\n"
            "       Check HuggingFace dataset column names / gated access.\n"
            f"       Manual placement: {img_dir}/<filename>.jpg"
        )
