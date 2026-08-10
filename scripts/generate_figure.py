"""Compatibility wrapper for paper figure generation.

Delegates rendering to the canonical split-figure generator at:
final/paper/figures/fig3_analysis.py
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).resolve().parents[1] / "paper" / "figures" / "fig3_analysis.py"
    runpy.run_path(str(script), run_name="__main__")
