"""Reproduce Figure 1: t-SNE visualization of cross-modal embeddings.

Generates fig1_tsne_6groups.{png,pdf} showing ImageBind embeddings
from MSCOCO, Flickr30K, and Clotho before and after PMC.
"""

from pathlib import Path
import runpy

if __name__ == "__main__":
    script = Path(__file__).resolve().parents[1] / "paper" / "figures" / "fig1_tsne.py"
    if not script.exists():
        # Try current/ path
        script = Path(__file__).resolve().parents[2] / "current" / "pmc_crossmodal" / "paper" / "figures" / "fig1_tsne.py"
    runpy.run_path(str(script), run_name="__main__")
