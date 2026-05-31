"""Generate split analysis figures for the paper.

Outputs:
- fig_gap_vs_gain.(pdf|png): single-column scatter with linear trend and Pearson r.
- fig_analysis_bcd.(pdf|png): full-width (a)(b)(c) analysis panels.
- fig_combined_1x4.(pdf|png): legacy combined layout for backward compatibility.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Style
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
    }
)

C_CLIP = "#1f77b4"
C_CLIPL = "#ff7f0e"
C_IB = "#2ca02c"
C_IB_AUDIO = "#9467bd"

rabitq_data = [
    ("CLIP-L", 0.82, 0.174),
    ("CLIP-L", 0.82, 0.107),
    ("CLIP-L", 0.77, 0.149),
    ("CLIP-L", 0.77, 0.088),
    ("CLIP", 0.82, 0.109),
    ("CLIP", 0.82, 0.054),
    ("ImageBind", 0.70, 0.046),
    ("ImageBind", 0.70, 0.081),
    ("ImageBind", 0.60, 0.068),
    ("ImageBind", 0.60, 0.016),
]
ivfpq_data = [
    ("CLIP", 0.82, 0.092),
    ("CLIP", 0.82, 0.148),
    ("CLIP-L", 0.82, 0.205),
    ("CLIP-L", 0.82, 0.238),
    ("ImageBind", 0.70, 0.108),
    ("ImageBind", 0.70, 0.092),
    ("CLIP-L", 0.77, 0.031),
    ("CLIP-L", 0.77, 0.038),
]
opq_data = [
    ("CLIP", 0.82, 0.055),
    ("CLIP", 0.82, -0.001),
    ("CLIP-L", 0.82, 0.117),
    ("CLIP-L", 0.82, 0.081),
    ("ImageBind", 0.70, 0.072),
    ("ImageBind", 0.70, 0.052),
    ("CLIP-L", 0.77, -0.004),
    ("CLIP-L", 0.77, 0.019),
]

backbone_color = {
    "CLIP": C_CLIP,
    "CLIP-L": C_CLIPL,
    "ImageBind": C_IB,
}
index_cfg = {
    "RaBitQ": {"marker": "o", "s": 28},
    "IVFPQ": {"marker": "^", "s": 28},
    "OPQ": {"marker": "s", "s": 22},
}

idx_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markersize=4.2, markeredgecolor="none", label="RaBitQ"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="grey", markersize=4.2, markeredgecolor="none", label="IVFPQ"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="grey", markersize=3.5, markeredgecolor="none", label="OPQ"),
]


def plot_gap_vs_gain(ax, title_prefix=None) -> None:
    points_x = []
    points_y = []
    for idx_name, dataset in [("RaBitQ", rabitq_data), ("IVFPQ", ivfpq_data), ("OPQ", opq_data)]:
        cfg = index_cfg[idx_name]
        for bb, gap, delta in dataset:
            ax.scatter(
                gap,
                delta,
                marker=cfg["marker"],
                s=cfg["s"],
                color=backbone_color[bb],
                edgecolors="white",
                linewidths=0.3,
                zorder=3,
            )
            points_x.append(gap)
            points_y.append(delta)

    x = np.array(points_x, dtype=float)
    y = np.array(points_y, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    xline = np.linspace(0.0, 1.0, 200)
    yline = slope * xline + intercept
    r = float(np.corrcoef(x, y)[0, 1])

    ax.plot(xline, yline, color="#1a1a1a", linewidth=1.4, alpha=0.95, linestyle="-", label="Trend")
    ax.axhline(0, color="grey", linewidth=0.4, linestyle="--", zorder=0)
    ax.text(
        0.97,
        0.015,
        rf"$r={r:.2f}$",
        transform=ax.transAxes,
        fontsize=7.2,
        va="bottom",
        ha="right",
        bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.92, boxstyle="round,pad=0.16"),
    )

    ax.set_xlabel(r"Modality Gap ($L_2$ norm)", fontsize=10.2, labelpad=3)
    ax.set_ylabel(r"$\Delta$R@100 (PMC $-$ Vanilla)", fontsize=10.2, labelpad=3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.03, 0.26)
    ax.tick_params(labelsize=8.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title_prefix:
        ax.set_title(rf"({title_prefix}) Gap vs $\mathbf{{\Delta}}$R@100", fontsize=12.6, fontweight="bold", pad=6)

    bb_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_CLIPL, markersize=4.2, markeredgecolor="none", label="CLIP-L"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_CLIP, markersize=4.2, markeredgecolor="none", label="CLIP"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_IB, markersize=4.2, markeredgecolor="none", label="IB"),
    ]
    leg_bb = ax.legend(
        handles=bb_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        borderaxespad=0.0,
        fontsize=7.3,
        title="Backbone",
        title_fontsize=7.4,
        frameon=True,
        framealpha=0.82,
        edgecolor="0.8",
        borderpad=0.16,
        labelspacing=0.15,
        handletextpad=0.22,
        handlelength=1.0,
    )
    ax.add_artist(leg_bb)
    ax.legend(
        handles=idx_handles,
        loc="upper left",
        bbox_to_anchor=(0.24, 0.99),
        borderaxespad=0.0,
        fontsize=7.3,
        title="Index",
        title_fontsize=7.4,
        frameon=True,
        framealpha=0.82,
        edgecolor="0.8",
        borderpad=0.16,
        labelspacing=0.15,
        handletextpad=0.22,
        handlelength=1.0,
    )


def plot_alpha_sweep(ax, panel_label: str = "a") -> None:
    alphas = [0.0, 0.25, 0.50, 0.75, 1.00]
    clip_ti = [0.5702, 0.5529, 0.5808, 0.6092, 0.6237]
    clip_it = [0.4940, 0.5143, 0.5478, 0.5846, 0.6031]
    ib_ti = [0.6636, 0.6454, 0.6785, 0.7166, 0.7445]
    ib_it = [0.7034, 0.6846, 0.7045, 0.7265, 0.7494]
    cl_ta = [0.6558, 0.6898, 0.7227, 0.7494, 0.7550]
    cl_at = [0.6308, 0.6536, 0.6895, 0.7168, 0.7364]

    ax.plot(alphas, clip_ti, "-o", color=C_CLIP, markersize=2.8, linewidth=1.1, label="CLIP t→i")
    ax.plot(alphas, clip_it, "--s", color=C_CLIP, markersize=2.8, linewidth=1.1, label="CLIP i→t")
    ax.plot(alphas, ib_ti, "-o", color=C_IB, markersize=2.8, linewidth=1.1, label="IB t→i")
    ax.plot(alphas, ib_it, "--s", color=C_IB, markersize=2.8, linewidth=1.1, label="IB i→t")
    ax.plot(alphas, cl_ta, "-D", color=C_IB_AUDIO, markersize=2.8, linewidth=1.1, label="Clotho t→a")
    ax.plot(alphas, cl_at, "--v", color=C_IB_AUDIO, markersize=2.8, linewidth=1.1, label="Clotho a→t")

    ms_data = [
        (0.5345, "o", C_CLIP),
        (0.4909, "s", C_CLIP),
        (0.6181, "o", C_IB),
        (0.6600, "s", C_IB),
        (0.7346, "D", C_IB_AUDIO),
        (0.7152, "v", C_IB_AUDIO),
    ]
    for v, m, c in ms_data:
        ax.scatter(-0.12, v, marker=m, color=c, s=20, edgecolors="white", linewidths=0.35, zorder=4)

    ax.axvline(0.0, color="#cccccc", linewidth=0.5, linestyle="--", zorder=0)
    ax.set_xlabel(r"$\alpha$", fontsize=11.2, labelpad=4)
    ax.set_ylabel("R@100", fontsize=11.2, labelpad=4)
    ax.set_xlim(-0.2, 1.08)
    ax.set_ylim(0.44, 0.76)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=8.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(rf"({panel_label}) Sign-bit: $\mathbf{{\alpha}}$ sweep", fontsize=12.6, fontweight="bold", pad=6)
    ax.text(-0.12, 0.445, "MS", fontsize=6.3, ha="center", color="0.5")

    # 2 rows × 3 columns (column-major): each column is a color pair
    leg_lines = ax.legend(
        fontsize=7.9,
        loc="lower right",
        bbox_to_anchor=(0.995, 0.01),
        ncol=3,
        frameon=True,
        framealpha=0.82,
        edgecolor="0.8",
        handlelength=1.4,
        borderpad=0.20,
        labelspacing=0.2,
        columnspacing=0.55,
    )
    ax.add_artist(leg_lines)
    ax.legend(
        handles=idx_handles,
        loc="center right",
        bbox_to_anchor=(0.995, 0.32),
        fontsize=7.9,
        title="Index",
        title_fontsize=8.0,
        frameon=True,
        framealpha=0.82,
        edgecolor="0.8",
        borderpad=0.2,
        labelspacing=0.2,
        handletextpad=0.25,
    )


def plot_selective(ax, panel_label: str = "b") -> None:
    pct = [0, 5, 10, 20, 50, 100]
    sel_r100_ti = [0.5734, 0.6641, 0.6619, 0.6586, 0.6509, 0.6482]
    sel_r100_it = [0.5112, 0.6776, 0.6797, 0.6827, 0.6839, 0.6827]
    sel_r100_ta = [0.7930, 0.7825, 0.7806, 0.7817, 0.7776, 0.7761]
    sel_r100_at = [0.7580, 0.7888, 0.7910, 0.7936, 0.7921, 0.7928]

    ax.plot(pct, sel_r100_ti, "-o", color=C_CLIP, markersize=2.8, linewidth=1.1, label="COCO t→i")
    ax.plot(pct, sel_r100_it, "--s", color=C_CLIP, markersize=2.8, linewidth=1.1, label="COCO i→t")
    ax.plot(pct, sel_r100_ta, "-D", color=C_IB_AUDIO, markersize=2.8, linewidth=1.1, label="Clotho t→a")
    ax.plot(pct, sel_r100_at, "--v", color=C_IB_AUDIO, markersize=2.8, linewidth=1.1, label="Clotho a→t")

    ax.set_xlabel(r"Top-$P\,\%$ dims", fontsize=11.2, labelpad=4)
    ax.set_ylabel("R@100", fontsize=11.2, labelpad=4)
    ax.set_xlim(-3, 105)
    ax.set_ylim(0.49, 0.81)
    ax.set_xticks([0, 5, 10, 20, 50, 100])
    ax.tick_params(labelsize=8.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(rf"({panel_label}) Selective PMC (top-$\mathbf{{P}}\,\%$)", fontsize=12.6, fontweight="bold", pad=6)

    ax.legend(
        fontsize=8.2,
        loc="lower right",
        frameon=True,
        framealpha=0.82,
        edgecolor="0.8",
        handlelength=1.5,
        borderpad=0.2,
        labelspacing=0.2,
    )


def plot_qps(ax, panel_label: str = "c") -> None:
    vanilla_r100 = [0.4083, 0.5262, 0.5788, 0.5869, 0.5701, 0.5433, 0.522]
    vanilla_qps = [21712, 11692, 5901, 2855, 1536, 808, 388]
    pmc_r100 = [0.366, 0.4837, 0.5649, 0.6067, 0.6232, 0.6214, 0.6116]
    pmc_qps = [17304, 10136, 4997, 2485, 1337, 661, 287]
    ms_r100 = [0.3613, 0.469, 0.5182, 0.5353, 0.5344, 0.5227, 0.5138]
    ms_qps = [19672, 11474, 5542, 3059, 1642, 821, 278]

    ax.plot(vanilla_qps, vanilla_r100, "-o", color="0.4", markersize=2.8, linewidth=1.1, label="Vanilla")
    ax.plot(pmc_qps, pmc_r100, "-s", color=C_CLIP, markersize=2.8, linewidth=1.1, label="PMC")
    ax.plot(ms_qps, ms_r100, "--^", color="#ff7f0e", markersize=2.8, linewidth=1.1, label="Meanshift")

    ax.set_xscale("log")
    ax.set_xlabel("QPS", fontsize=11.2, labelpad=4)
    ax.set_ylabel("R@100", fontsize=11.2, labelpad=4)
    ax.set_xlim(200, 30000)
    ax.set_ylim(0.34, 0.65)
    ax.tick_params(labelsize=8.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        fontsize=8.2,
        loc="lower left",
        frameon=True,
        framealpha=0.82,
        edgecolor="0.8",
        handlelength=1.5,
        borderpad=0.2,
        labelspacing=0.2,
    )
    ax.set_title(rf"({panel_label}) R@100–QPS Pareto", fontsize=12.6, fontweight="bold", pad=6)

    ax.annotate(
        "$n_p{=}16$",
        xy=(1337, 0.6232),
        fontsize=9.0,
        color=C_CLIP,
        xytext=(2450, 0.631),
        arrowprops=dict(arrowstyle="->", color=C_CLIP, lw=0.5),
    )


def save_split_figures(outdir: Path) -> None:
    fig_gap, ax_gap = plt.subplots(1, 1, figsize=(3.20, 2.10))
    plot_gap_vs_gain(ax_gap)
    fig_gap.tight_layout(pad=0.28)
    fig_gap.savefig(str(outdir / "fig_gap_vs_gain.pdf"), bbox_inches="tight")
    fig_gap.savefig(str(outdir / "fig_gap_vs_gain.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_gap)

    fig_bcd, axes_bcd = plt.subplots(1, 3, figsize=(12.0, 3.10))
    plot_alpha_sweep(axes_bcd[0], panel_label="a")
    plot_selective(axes_bcd[1], panel_label="b")
    plot_qps(axes_bcd[2], panel_label="c")
    fig_bcd.tight_layout(pad=0.45, w_pad=0.95)
    fig_bcd.savefig(str(outdir / "fig_analysis_bcd.pdf"), bbox_inches="tight")
    fig_bcd.savefig(str(outdir / "fig_analysis_bcd.png"), dpi=300, bbox_inches="tight")
    fig_bcd.savefig(str(outdir / "fig3_analysis.pdf"), bbox_inches="tight")
    fig_bcd.savefig(str(outdir / "fig3_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_bcd)


def save_legacy_combined(outdir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13.8, 3.12))
    plot_gap_vs_gain(axes[0], title_prefix="a")
    plot_alpha_sweep(axes[1], panel_label="b")
    plot_selective(axes[2], panel_label="c")
    plot_qps(axes[3], panel_label="d")
    fig.tight_layout(pad=0.58, w_pad=1.12)
    fig.savefig(str(outdir / "fig_combined_1x4.pdf"), bbox_inches="tight")
    fig.savefig(str(outdir / "fig_combined_1x4.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    outdir = Path(__file__).resolve().parent
    save_split_figures(outdir)
    save_legacy_combined(outdir)
    print("Saved fig_gap_vs_gain, fig_analysis_bcd, and fig_combined_1x4 (pdf/png)")
