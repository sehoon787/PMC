"""Download LAION-400M CLIP ViT-B/32 embeddings (float16, 512d).

Downloads image embeddings (410 shards, ~391 GB) and text embeddings
(up to 410 shards, ~391 GB) for bidirectional evaluation.

Source: https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/
No authentication required.

Usage
-----
  python scripts/download_laion400m.py                  # download all
  python scripts/download_laion400m.py --text-only      # text shards only
  python scripts/download_laion400m.py --img-only       # image shards only
  python scripts/download_laion400m.py --max-workers 4  # limit parallelism
  python scripts/download_laion400m.py --text-only --n-text-shards 410  # all text
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings"
N_IMG_SHARDS = 410
N_TEXT_SHARDS = 410  # full text embeddings for bidirectional eval
TARGET_DIR = Path("E:/laion400m")


def download_shard(kind: str, idx: int, target_dir: Path) -> tuple[int, bool, str]:
    """Download one shard. Returns (idx, success, message)."""
    subdir = "img_emb" if kind == "img" else "text_emb"
    filename = f"{subdir}_{idx}.npy"
    url = f"{BASE_URL}/{subdir}/{filename}"
    dest = target_dir / subdir / filename

    if dest.exists() and dest.stat().st_size > 1_000_000:
        return idx, True, "cached"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, str(tmp))
        tmp.rename(dest)
        size_mb = dest.stat().st_size / 1e6
        return idx, True, f"ok ({size_mb:.0f} MB)"
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return idx, False, str(exc)[:200]


def download_all(
    kind: str,
    n_shards: int,
    target_dir: Path,
    max_workers: int,
) -> tuple[int, int]:
    """Download shards in parallel. Returns (success, failed) counts."""
    subdir = "img_emb" if kind == "img" else "text_emb"
    shard_dir = target_dir / subdir

    # Check existing
    existing = sum(
        1 for i in range(n_shards)
        if (shard_dir / f"{subdir}_{i}.npy").exists()
        and (shard_dir / f"{subdir}_{i}.npy").stat().st_size > 1_000_000
    )
    remaining = n_shards - existing
    print(f"\n[{subdir}] {existing}/{n_shards} shards cached, {remaining} to download")

    if remaining == 0:
        return existing, 0

    t0 = time.perf_counter()
    success, failed = existing, 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for i in range(n_shards):
            dest = shard_dir / f"{subdir}_{i}.npy"
            if dest.exists() and dest.stat().st_size > 1_000_000:
                continue
            futures[pool.submit(download_shard, kind, i, target_dir)] = i

        for j, future in enumerate(as_completed(futures), 1):
            idx, ok, msg = future.result()
            if ok:
                success += 1
            else:
                failed += 1
                print(f"  FAIL shard {idx}: {msg}")

            elapsed = time.perf_counter() - t0
            if j % 5 == 0 or j == len(futures):
                speed = j / elapsed if elapsed > 0 else 0
                eta = (len(futures) - j) / speed / 60 if speed > 0 else 0
                print(
                    f"  [{subdir}] {j}/{len(futures)} done | "
                    f"OK={success} FAIL={failed} | "
                    f"ETA {eta:.0f} min"
                )

    return success, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LAION-400M embeddings")
    parser.add_argument("--img-only", action="store_true", help="Image shards only")
    parser.add_argument("--text-only", action="store_true", help="Text shards only")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel downloads")
    parser.add_argument("--target-dir", type=str, default=str(TARGET_DIR))
    parser.add_argument("--n-text-shards", type=int, default=N_TEXT_SHARDS,
                        help="Number of text shards to download (default: 410)")
    args = parser.parse_args()

    target = Path(args.target_dir)
    do_img = not args.text_only
    do_text = not args.img_only

    print("=" * 60)
    print("LAION-400M Embedding Downloader")
    print(f"  Target: {target}")
    print(f"  Workers: {args.max_workers}")
    if do_img:
        print(f"  Image shards: {N_IMG_SHARDS} (~391 GB)")
    if do_text:
        print(f"  Text shards: {args.n_text_shards} (~{args.n_text_shards} GB)")
    print("=" * 60)

    if do_text:
        s, f = download_all("text", args.n_text_shards, target, args.max_workers)
        print(f"\n[text_emb] Done: {s} OK, {f} failed")

    if do_img:
        s, f = download_all("img", N_IMG_SHARDS, target, args.max_workers)
        print(f"\n[img_emb] Done: {s} OK, {f} failed")

    print("\n[laion400m] All downloads complete.")


if __name__ == "__main__":
    main()
