"""
fig1_tsne_6groups.py
Figure 1 (6-group variant, 3 datasets) — t-SNE visualization of cross-modal embeddings.

Groups (ImageBind 1024d):
  - MSCOCO Image     (dark blue   ● circle)   /  MSCOCO Text     (light blue   ▲ triangle)
  - Flickr30K Image  (dark red    ● circle)   /  Flickr30K Text  (light red    ▲ triangle)
  - Clotho Audio     (dark green  ● circle)   /  Clotho Text     (light green  ▲ triangle)

Design:
  - dataset = color family, modality = dark/light shade + marker
  - smooth scipy gaussian_kde contour fills with percentile threshold
  - centroid ★, gap arrows in BOTH panels colored to match dataset dark shade
  - no in-figure legend (encoding described in caption)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
import sklearn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEAT_DIR = os.environ.get("PMC_FEATURES_DIR", "data/features").rstrip("/") + "/"
OUT_DIR  = os.environ.get("PMC_FIGURES_DIR", os.path.dirname(os.path.abspath(__file__))).rstrip("/") + "/"

MAX_PER_GROUP    = 2000
TSNE_PERPLEXITY  = 30
TSNE_SEED        = 42
KDE_ALPHA        = 0.2
KDE_BW           = 0.3
KDE_LEVELS       = 4
SCATTER_ALPHA    = 0.45
SCATTER_S        = 22
CENTROID_S       = 280
DPI              = 300

# dataset = color family; DB (image/audio) = dark shade, Query (text) = light shade
COLORS = {
    "mscoco_image":    "#1565C0",  # dark blue
    "mscoco_text":     "#90CAF9",  # light blue
    "flickr_image":    "#C62828",  # dark red
    "flickr_text":     "#EF9A9A",  # light red/pink
    "clotho_audio":    "#2E7D32",  # dark green
    "clotho_text":     "#A5D6A7",  # light green
}

LABELS = {
    "mscoco_image":    "MSCOCO Img ●",
    "mscoco_text":     "MSCOCO Txt ▲",
    "flickr_image":    "Flickr30K Img ●",
    "flickr_text":     "Flickr30K Txt ▲",
    "clotho_audio":    "Clotho Aud ●",
    "clotho_text":     "Clotho Txt ▲",
}

# marker per modality role: DB=circle, Query=triangle
MARKERS = {
    "mscoco_image":    "o",
    "mscoco_text":     "^",
    "flickr_image":    "o",
    "flickr_text":     "^",
    "clotho_audio":    "o",
    "clotho_text":     "^",
}

# Pairs: (db_key, query_key)
PAIRS = [
    ("mscoco_image",    "mscoco_text"),
    ("flickr_image",    "flickr_text"),
    ("clotho_audio",    "clotho_text"),
]

# gap arrow color = dark shade of that dataset
ARROW_COLORS = {
    ("mscoco_image",    "mscoco_text"):    "#1565C0",
    ("flickr_image",    "flickr_text"):    "#C62828",
    ("clotho_audio",    "clotho_text"):    "#2E7D32",
}

# Ordered for legend: DB then Query per dataset
KEYS_ORDERED = [
    "mscoco_image", "mscoco_text",
    "flickr_image", "flickr_text",
    "clotho_audio", "clotho_text",
]

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       10,
    "axes.titlesize":  20,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 13,
    "figure.dpi":      DPI,
    "axes.linewidth":  0.8,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-8)).astype(np.float32)


def subsample(arr: np.ndarray, max_n: int, seed: int = 42) -> np.ndarray:
    if len(arr) <= max_n:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), max_n, replace=False)
    return arr[idx]


def plot_kde_contour(ax, x: np.ndarray, y: np.ndarray, color: str,
                     alpha: float = KDE_ALPHA, levels: int = KDE_LEVELS,
                     bw_method: float = KDE_BW) -> None:
    """Plot smooth Gaussian KDE density contours with percentile threshold clipping."""
    try:
        kde = gaussian_kde(np.vstack([x, y]), bw_method=bw_method)
        margin = 3
        xmin, xmax = x.min() - margin, x.max() + margin
        ymin, ymax = y.min() - margin, y.max() + margin
        xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = kde(positions).reshape(xx.shape)
        # Clip outermost contour so it doesn't look boxy
        pos_vals = density[density > 0]
        if len(pos_vals) > 0:
            threshold = np.percentile(pos_vals, 15)
            density[density < threshold] = 0
        ax.contourf(xx, yy, density, levels=levels,
                    colors=[color], alpha=alpha)
        ax.contour(xx, yy, density, levels=levels,
                   colors=[color], alpha=alpha + 0.15, linewidths=0.5)
    except Exception:
        pass  # skip if too few points


def draw_panel(ax, embeddings_2d: dict, title: str,
               gap_norms_orig=None) -> None:
    """Draw one t-SNE panel with scatter, KDE, centroids, and gap arrows."""
    ax.set_facecolor("white")
    ax.set_title(title, fontweight="bold", fontsize=20, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    centroids = {}
    for key in KEYS_ORDERED:
        pts = embeddings_2d[key]
        color  = COLORS[key]
        marker = MARKERS[key]

        # smooth KDE contour background
        if len(pts) >= 20:
            plot_kde_contour(ax, pts[:, 0], pts[:, 1], color)

        # scatter with modality-specific marker
        ax.scatter(pts[:, 0], pts[:, 1],
                   s=SCATTER_S, c=color, marker=marker,
                   alpha=SCATTER_ALPHA, linewidths=0, zorder=3)

        # centroid star — computed from subsampled 2D points
        c = pts.mean(axis=0)
        centroids[key] = c
        ax.scatter(*c, s=CENTROID_S, marker="*", c=color,
                   edgecolors="black", linewidths=1.5, zorder=5)

    # Gap arrows between paired centroids (drawn in BOTH panels)
    # Vertical offsets to prevent label overlap
    label_offsets = [(-14, "bottom"), (14, "top"), (-28, "bottom")]
    for idx, (db_key, q_key) in enumerate(PAIRS):
        c_db = centroids[db_key]
        c_q  = centroids[q_key]
        arrow_color = ARROW_COLORS[(db_key, q_key)]

        ax.annotate(
            "",
            xy=c_q, xytext=c_db,
            arrowprops=dict(
                arrowstyle="<->",
                color=arrow_color,
                lw=1.2,
                linestyle="dashed",
                mutation_scale=12,
            ),
            zorder=4,
        )

        # Use original 1024d gap norm if available, else t-SNE distance
        if gap_norms_orig and (db_key, q_key) in gap_norms_orig:
            label = f"‖g‖={gap_norms_orig[(db_key, q_key)]:.2f}"
        else:
            label = f"‖g‖={np.linalg.norm(c_q - c_db):.2f}"

        mid = (c_db + c_q) / 2
        y_off, va = label_offsets[idx]
        ax.text(
            mid[0], mid[1] + y_off,
            label,
            fontsize=13,
            fontweight="bold",
            ha="center", va=va,
            color=arrow_color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.88),
            zorder=6,
        )


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading features …")
raw = {
    "mscoco_image":    np.load(FEAT_DIR + "mscoco_karpathy_val5k_imagebind_image_seed42.npy"),
    "mscoco_text":     np.load(FEAT_DIR + "mscoco_karpathy_val5k_imagebind_text_seed42.npy"),
    "flickr_image":    np.load(FEAT_DIR + "flickr30k_test1k_imagebind_image_seed42.npy"),
    "flickr_text":     np.load(FEAT_DIR + "flickr30k_test1k_imagebind_text_seed42.npy"),
    "clotho_audio":    np.load(FEAT_DIR + "clotho_eval_imagebind_audio_seed42.npy"),
    "clotho_text":     np.load(FEAT_DIR + "clotho_eval_imagebind_text_seed42.npy"),
}
for k, v in raw.items():
    print(f"  {k}: {v.shape}")

raw = {k: l2_normalize(v) for k, v in raw.items()}

# ---------------------------------------------------------------------------
# PMC correction (α=1, DB-side only)
# Compute gap from FULL data before any subsampling
# ---------------------------------------------------------------------------
print("\nComputing PMC corrections …")
pmc = {k: v.copy() for k, v in raw.items()}
gap_norms_orig = {}  # store original 1024d gap norms for labels

for db_key, q_key in PAIRS:
    g = raw[q_key].mean(axis=0) - raw[db_key].mean(axis=0)
    gap_norm = np.linalg.norm(g)
    gap_norms_orig[(db_key, q_key)] = gap_norm
    print(f"  Gap magnitude ({db_key} ↔ {q_key}): ‖g‖ = {gap_norm:.4f}")
    pmc[db_key] = l2_normalize(raw[db_key] + g[np.newaxis, :])
    # query unchanged at α=1

# Compute post-PMC gap norms in 1024d space
gap_norms_pmc = {}
for db_key, q_key in PAIRS:
    g_post = pmc[q_key].mean(axis=0) - pmc[db_key].mean(axis=0)
    gap_post = np.linalg.norm(g_post)
    gap_norms_pmc[(db_key, q_key)] = gap_post
    print(f"  Post-PMC gap ({db_key} ↔ {q_key}): ‖g‖ = {gap_post:.4f}")

# ---------------------------------------------------------------------------
# Subsample for t-SNE (centroids computed from full data above)
# ---------------------------------------------------------------------------
raw_sub = {k: subsample(raw[k], MAX_PER_GROUP) for k in KEYS_ORDERED}
pmc_sub = {k: subsample(pmc[k], MAX_PER_GROUP) for k in KEYS_ORDERED}


def stack_ordered(d: dict) -> np.ndarray:
    return np.vstack([d[k] for k in KEYS_ORDERED])


raw_stack = stack_ordered(raw_sub)
pmc_stack = stack_ordered(pmc_sub)

# ---------------------------------------------------------------------------
# t-SNE — separate fit per panel
# ---------------------------------------------------------------------------
_tsne_iter_kw = (
    "max_iter"
    if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5)
    else "n_iter"
)

print("\nRunning t-SNE for panel (a) …")
tsne_a = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY,
              random_state=TSNE_SEED, **{_tsne_iter_kw: 1000}, init="pca")
emb_a = tsne_a.fit_transform(raw_stack)

print("Running t-SNE for panel (b) …")
tsne_b = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY,
              random_state=TSNE_SEED, **{_tsne_iter_kw: 1000}, init="pca")
emb_b = tsne_b.fit_transform(pmc_stack)


def split_embeddings(emb_2d: np.ndarray, sub_dict: dict) -> dict:
    result = {}
    offset = 0
    for k in KEYS_ORDERED:
        n = len(sub_dict[k])
        result[k] = emb_2d[offset: offset + n]
        offset += n
    return result


emb_a_dict = split_embeddings(emb_a, raw_sub)
emb_b_dict = split_embeddings(emb_b, pmc_sub)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
print("\nRendering figure …")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.patch.set_facecolor("white")

draw_panel(axes[0], emb_a_dict, "(a) Original", gap_norms_orig=gap_norms_orig)
draw_panel(axes[1], emb_b_dict, "(b) After PMC (α=1)", gap_norms_orig=gap_norms_pmc)

fig.subplots_adjust(top=0.90, bottom=0.03, left=0.02, right=0.98, wspace=0.08)

out_png = OUT_DIR + "fig1_tsne_6groups.png"
out_pdf = OUT_DIR + "fig1_tsne_6groups.pdf"
fig.savefig(out_png, dpi=DPI, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
print(f"\nSaved:\n  {out_png}\n  {out_pdf}")
