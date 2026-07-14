#!/usr/bin/env python3
"""Export externally authored Figure 2 assets without redrawing.

This script treats fig_pmc_overview_source.png (or .jpg fallback) as the
source of truth and only converts it to PNG/PDF for LaTeX consumption.
"""

from pathlib import Path
import subprocess


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    src_png = out_dir / "fig_pmc_overview_source.png"
    src_jpg = out_dir / "fig_pmc_overview_source.jpg"
    src = src_png if src_png.exists() else src_jpg
    png = out_dir / "fig_pmc_overview.png"
    pdf = out_dir / "fig_pmc_overview.pdf"

    try:
        from PIL import Image

        img = Image.open(src).convert("RGB")
        img.save(png)
        img.save(pdf, "PDF", resolution=300.0)
    except ModuleNotFoundError:
        subprocess.run(["sips", "-s", "format", "png", str(src), "--out", str(png)], check=True)
        subprocess.run(["sips", "-s", "format", "pdf", str(src), "--out", str(pdf)], check=True)


if __name__ == "__main__":
    main()
