"""Generate the CIKM 2026 poster for PMC (csp4014) as an editable PPTX.

v2 — Korea University / IE LAB visual language, matching the lab's own deck:
white ground, KU-crimson serif headings (#8B0029 sampled from the lab deck),
thin crimson rules, IE-LAB blue (#0D4DA3) as the single accent for PMC,
gray for baselines. No colored panel fills, no icon chips — academic and
quiet, legible at one metre.

A0 portrait (33.11 x 46.81 in); rescale once CIKM fixes the poster spec.
Every element is a native, editable PowerPoint shape.

Usage:  python make_poster.py
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent

CRIMSON = RGBColor(0x8B, 0x00, 0x29)   # KU deck title colour (sampled)
BLUE = RGBColor(0x0D, 0x4D, 0xA3)      # IE LAB logo blue -> the PMC accent
INK = RGBColor(0x1F, 0x1F, 0x1F)
GRAY = RGBColor(0x6F, 0x6F, 0x6F)
HAIR = RGBColor(0xC8, 0xC8, 0xC8)
CARD = RGBColor(0xF6, 0xF5, 0xF3)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SERIF = "Times New Roman"

PAGE_W, PAGE_H = Inches(33.11), Inches(46.81)

# Native pixel sizes of the embedded assets, for exact aspect-ratio placement.
ASPECT = {
    "asset_fig1.png": 2400 / 1193,
    "asset_fig3b.png": 1188 / 929,
    "asset_fig3c.png": 1206 / 929,
}


def solid(shape, color, line_color=None, line_w=None):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    if line_color is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line_color
        shape.line.width = line_w or Pt(1.0)
    shape.shadow.inherit = False


def rule(slide, x, y, w, weight=Pt(2.4), color=CRIMSON):
    r = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, weight)
    solid(r, color)
    return r


def text(slide, x, y, w, h, paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         line_spacing=1.08, space_after=0):
    """paras: list of paragraphs; each a list of (text, size, bold, color[, italic])."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        p.space_after = Pt(space_after)
        for run in para:
            t, size, bold, color = run[:4]
            italic = run[4] if len(run) > 4 else False
            # "_{q}" marks a subscript, so mu_q reads as the paper sets it.
            for chunk, is_sub in _split_subscripts(t):
                r = p.add_run()
                r.text = chunk
                r.font.size = Pt(size * 0.72 if is_sub else size)
                r.font.bold = bold
                r.font.italic = italic
                r.font.color.rgb = color
                r.font.name = SERIF
                if is_sub:
                    r.font._rPr.set("baseline", "-25000")
    return tb


def _split_subscripts(t):
    """Yield (text, is_subscript) pairs, splitting on the _{...} marker."""
    out, i = [], 0
    while True:
        j = t.find("_{", i)
        if j < 0:
            if t[i:]:
                out.append((t[i:], False))
            return out
        k = t.find("}", j)
        if t[i:j]:
            out.append((t[i:j], False))
        out.append((t[j + 2:k], True))
        i = k + 1


def picture(slide, name, x, y, w):
    """Place an asset at width w and return the height it occupies."""
    h = Emu(int(w / ASPECT[name]))
    slide.shapes.add_picture(str(HERE / name), x, y, w, h)
    return h


def fraction(slide, x, y, w, num, den, size, color=INK):
    """Stacked fraction with a rule sized to the wider of the two lines."""
    h = Inches(size / 72 * 1.35)
    # Times averages ~0.5 em per glyph; pad the longer line slightly.
    bar_w = min(w, Inches(max(len(num), len(den)) * size / 72 * 0.56))
    bx = x + Emu(int((w - bar_w) / 2))
    text(slide, x, y, w, h, [[(num, size, True, color)]], align=PP_ALIGN.CENTER)
    bar_y = y + h + Inches(0.05)
    rule(slide, bx, bar_y, bar_w, Pt(2.4), color)
    text(slide, x, bar_y + Inches(0.11), w, h,
         [[(den, size, True, color)]], align=PP_ALIGN.CENTER)
    return h * 2 + Inches(0.28)


def heading(slide, x, y, w, num, label):
    text(slide, x, y, w, Inches(1.1),
         [[(f"{num}.  {label}", 42, True, CRIMSON)]], line_spacing=1.05)
    rule(slide, x, y + Inches(1.15), w, Pt(2.0), CRIMSON)
    return y + Inches(1.75)


def hbar(slide, x, y, w_full, frac, h, color, label, value):
    lane = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w_full, h)
    solid(lane, WHITE, HAIR, Pt(1.0))
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, Emu(int(w_full * frac)), h)
    solid(bar, color)
    text(slide, x + Inches(0.2), y, w_full - Inches(0.35), h,
         [[(label, 24, False, WHITE if frac > 0.45 else INK)]],
         anchor=MSO_ANCHOR.MIDDLE)
    text(slide, x + w_full + Inches(0.18), y, Inches(1.8), h,
         [[(value, 26, True, INK)]], anchor=MSO_ANCHOR.MIDDLE)


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = PAGE_W, PAGE_H
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, PAGE_W, PAGE_H), WHITE)

    M = Inches(1.05)
    CW = PAGE_W - 2 * M

    # ================= Header =================
    # Logos: KU crest is 157x49 px, IE LAB 177x71 px. Widths are derived from the
    # set heights so the two never collide, and the QR sits under the pair.
    KU_H, IE_H = Inches(1.30), Inches(1.30)
    KU_W = Emu(int(KU_H * 157 / 49))
    IE_W = Emu(int(IE_H * 177 / 71))
    LOGO_R = PAGE_W - M
    slide.shapes.add_picture(str(HERE / "asset_ielab.png"),
                             LOGO_R - IE_W, Inches(1.00), IE_W, IE_H)
    slide.shapes.add_picture(str(HERE / "asset_ku.jpg"),
                             LOGO_R - IE_W - Inches(0.55) - KU_W, Inches(1.00), KU_W, KU_H)
    QR = Inches(2.3)
    slide.shapes.add_picture(str(HERE / "asset_qr.png"),
                             LOGO_R - QR, Inches(2.75), QR, QR)
    text(slide, LOGO_R - Inches(6.5), Inches(5.15), Inches(6.5), Inches(0.5),
         [[("Paper + code:  github.com/sehoon787/PMC", 18, False, GRAY)]],
         align=PP_ALIGN.RIGHT)

    TW = CW - Inches(9.6)
    text(slide, M, Inches(1.0), TW, Inches(2.9),
         [[("PMC: Build-Time Per-Modality Centroid Correction", 62, True, CRIMSON)],
          [("for Cross-Modal Binary-Quantized Retrieval", 62, True, CRIMSON)]],
         line_spacing=1.1)
    text(slide, M, Inches(3.95), TW, Inches(0.8),
         [[("Se Hoon Kim,   Jun Hyung Lee,   Soonyoung Jung*", 34, False, INK)]])
    text(slide, M, Inches(4.78), TW, Inches(0.7),
         [[("Department of Computer Science and Engineering, Korea University   ·   "
            "Intelligence Engineering Lab", 25, False, GRAY)]])
    rule(slide, M, Inches(5.85), CW, Pt(3.0), CRIMSON)

    # ================= Row 1 =================
    R1_Y = Inches(6.45)
    R1_H = Inches(22.6)
    GUT = Inches(1.0)
    W1 = Emu(int(CW * 0.30))
    W2 = Emu(int(CW * 0.335))
    W3 = CW - W1 - W2 - 2 * GUT
    X1, X2, X3 = M, M + W1 + GUT, M + W1 + GUT + W2 + GUT
    for xd in (X2 - Emu(int(GUT / 2)), X3 - Emu(int(GUT / 2))):
        solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, xd, R1_Y, Pt(1.0), R1_H), HAIR)

    # ---- 1. The problem ----
    y = heading(slide, X1, R1_Y, W1, "1", "The problem")
    text(slide, X1, y, W1, Inches(1.8),
         [[("Cross-modal queries and database vectors do not share a centroid.",
            32, True, INK)]], line_spacing=1.15)
    y += Inches(2.25)
    y += picture(slide, "asset_fig1.png", X1, y, W1) + Inches(0.35)
    text(slide, X1, y, W1, Inches(1.0),
         [[("t-SNE (ImageBind): modality clusters separate in (a); after PMC they "
            "overlap in (b).", 20, False, GRAY, True)]], line_spacing=1.15)
    y += Inches(1.35)
    text(slide, X1, y, W1, Inches(0.9),
         [[("One gap, two failures", 36, True, CRIMSON)]])
    y += Inches(1.15)
    for head_t, body_t, gap in [
            ("IVF routing", "queries are steered into the wrong inverted lists",
             Inches(1.25)),
            ("Binary codes", "sign(x − c) has zero error margin — the boundary "
             "passes exactly through the centroid, so the offset flips bits",
             Inches(2.35))]:
        solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, X1, y + Inches(0.16),
                                     Inches(0.26), Inches(0.26)), CRIMSON)
        text(slide, X1 + Inches(0.5), y, W1 - Inches(0.5), Inches(2.2),
             [[(head_t + "  —  ", 26, True, INK), (body_t, 26, False, INK)]],
             line_spacing=1.18)
        y += gap
    y += Inches(0.2)
    solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, X1, y, W1, Inches(2.7)),
          CARD, HAIR, Pt(1.0))
    text(slide, X1 + Inches(0.45), y + Inches(0.35), W1 - Inches(0.9), Inches(2.1),
         [[("Oracle test — MSCOCO, CLIP-L, RaBitQ", 23, True, INK)],
          [("same-modality  (‖g‖ = 0)       R@100 = 0.71", 25, False, INK)],
          [("cross-modal  (‖g‖ = .82)       R@100 = 0.54", 25, True, CRIMSON)]],
         line_spacing=1.4)
    y += Inches(3.0)
    text(slide, X1, y, W1, Inches(2.4),
         [[("The damage is structured. ", 24, True, INK),
           ("The top 10% of dimensions carry ≈90% of ‖g‖² on CLIP-L, and flip "
            "risk grows with |g_{i}|:   E[F] ≈ α Σ_{i} p_{i}(0)·|g_{i}|.", 24, False, INK)]],
         line_spacing=1.28)

    # ---- 2. Method ----
    y = heading(slide, X2, R1_Y, W2, "2", "PMC — correct at the source")
    STEP_H = Inches(3.45)
    for i, (h1, h2) in enumerate([
            ("Calibrate", "estimate the gap from a small paired sample"),
            ("Build  (offline)", "shift every database vector, then build the index"),
            ("Serve  (online)", "search the corrected index — the query is untouched")]):
        solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, X2, y, Pt(5.0), STEP_H), BLUE)
        text(slide, X2 + Inches(0.5), y - Inches(0.06), W2 - Inches(0.55), Inches(0.95),
             [[(f"{i + 1}    ", 38, True, BLUE), (h1, 38, True, INK)]])
        text(slide, X2 + Inches(0.5), y + Inches(0.95), W2 - Inches(0.55), Inches(0.85),
             [[(h2, 25, False, GRAY)]], line_spacing=1.15)
        eq_y = y + Inches(1.85)
        eq_x, eq_w = X2 + Inches(0.5), W2 - Inches(1.0)
        if i == 0:      # g = mu_q - mu_x   (Sec. 3.1)
            text(slide, eq_x, eq_y + Inches(0.25), eq_w, Inches(1.0),
                 [[("g  =  μ_{q} − μ_{x}", 32, True, INK),
                   ("        (25 samples suffice)", 24, False, GRAY)]])
        elif i == 1:    # x' = (x + alpha g) / ||x + alpha g||   (Eq. 3)
            text(slide, eq_x, eq_y + Inches(0.32), Inches(1.5), Inches(0.9),
                 [[("x′  =", 32, True, INK)]])
            fraction(slide, eq_x + Inches(1.45), eq_y - Inches(0.05),
                     Inches(4.6), "x + α g", "‖x + α g‖", 30)
        else:           # q' = q at alpha = 1
            text(slide, eq_x, eq_y + Inches(0.25), eq_w, Inches(1.0),
                 [[("q′  =  q", 32, True, INK),
                   ("        (at α = 1)", 24, False, GRAY)]])
        y += STEP_H
        if i < 2:
            ar = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW,
                                        X2 + Emu(int(W2 / 2)) - Inches(0.6),
                                        y + Inches(0.12), Inches(1.2), Inches(0.85))
            solid(ar, BLUE)
            y += Inches(1.15)

    y += Inches(0.5)
    solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, X2, y, W2, Inches(2.05)), BLUE)
    text(slide, X2 + Inches(0.5), y + Inches(0.32), W2 - Inches(1.0), Inches(1.5),
         [[("At α = 1 the correction is absorbed into the index:", 29, True, WHITE)],
          [("zero query-time transform, zero extra memory.", 29, True, WHITE)]],
         line_spacing=1.3)
    y += Inches(2.55)
    text(slide, X2, y, W2, Inches(0.6),
         [[("The query side carries the complement:", 25, False, INK)]])
    y += Inches(0.75)
    text(slide, X2 + Inches(0.15), y + Inches(0.3), Inches(1.5), Inches(0.9),
         [[("q′  =", 28, True, INK)]])
    y += fraction(slide, X2 + Inches(1.45), y, Inches(6.2),
                  "q − (1 − α) g", "‖q − (1 − α) g‖", 26)
    text(slide, X2, y, W2, Inches(2.9),
         [[("α = 0 is query-only mean shift: it leaves IVF centroids and quantized "
            "codes misaligned, and can lower recall.", 25, False, INK)],
          [("α = 1 makes q′ = q and rewrites the centroid where it plays both of "
            "its roles — routing pivot and code boundary.", 25, True, INK)],
          [("The α-sweep confirms α = 1 best or near-best in every setting.",
            25, False, BLUE)]],
         line_spacing=1.28, space_after=12)

    # ---- 3. Results ----
    y = heading(slide, X3, R1_Y, W3, "3", "Results")
    text(slide, X3, y - Inches(0.15), W3, Inches(2.0),
         [[("407 M", 92, True, CRIMSON), ("   vectors", 32, False, GRAY)]],
         line_spacing=0.95)
    y += Inches(1.6)
    # Scale context, so the low absolute recall below reads as the difficulty of
    # the task rather than a weakness of the method.
    text(slide, X3, y, W3, Inches(0.6),
         [[("LAION-400M  ·  29.3 GB at 72 B/vec  ·  28× compressed",
            24, True, GRAY)]])
    y += Inches(0.8)
    text(slide, X3, y, W3, Inches(0.6),
         [[("CLIP · R@100, no reranking", 23, True, INK)]])
    y += Inches(0.75)
    bw = W3 - Inches(2.0)
    hbar(slide, X3, y, bw, 0.108 / 0.143, Inches(0.8), GRAY, "Vanilla", ".108")
    y += Inches(0.98)
    hbar(slide, X3, y, bw, 1.0, Inches(0.8), BLUE, "PMC   (+32%)", ".143")
    y += Inches(1.12)
    text(slide, X3, y, W3, Inches(0.6),
         [[("with exact reranking K′ = 400:   .198 → ", 25, False, INK),
           (".277   (+40%)", 25, True, BLUE)]])
    y += Inches(1.1)

    # R@100 for all 14 configurations, ordered by modality gap. The ordering is
    # the argument: as ||g|| falls down the table, so does the gain. Values are
    # Table 2's, verbatim from results/tab2_main_reproduced.csv.
    text(slide, X3, y, W3, Inches(0.75),
         [[("Every configuration, ordered by modality gap", 26, True, INK)]])
    y += Inches(0.85)
    # Column widths are derived from W3 so the name column cannot be squeezed out.
    COL_G, COL_N = Inches(0.85), Inches(3.0)
    COL_D = Emu(int((W3 - COL_G - COL_N) / 2))   # each direction block
    text(slide, X3, y, COL_G, Inches(0.6), [[("‖g‖", 22, True, GRAY)]])
    text(slide, X3 + COL_G, y, COL_N, Inches(0.6),
         [[("Dataset · Enc.", 22, True, GRAY)]])
    for i, h in enumerate(("q→db", "db→q")):
        text(slide, X3 + COL_G + COL_N + i * COL_D, y, COL_D, Inches(0.6),
             [[(h, 22, True, GRAY)]], align=PP_ALIGN.RIGHT)
    y += Inches(0.68)
    rule(slide, X3, y, W3, Pt(1.6), INK)
    y += Inches(0.16)
    TABLE = [
        (".82", "MSCOCO CL-L", ".55", ".65", "+18%", ".47", ".63", "+34%"),
        (".82", "MSCOCO CLIP", ".58", ".63", "+9%", ".50", ".60", "+20%"),
        (".77", "Flickr30K CL-L", ".41", ".48", "+17%", ".33", ".48", "+45%"),
        (".72", "LAION-400M CLIP", ".108", ".143", "+32%", ".069", ".073", "+6%"),
        (".70", "MSCOCO IB", ".67", ".75", "+12%", ".71", ".75", "+6%"),
        (".61", "Clotho IB", ".72", ".73", "+1%", ".62", ".69", "+11%"),
        (".61", "AudioCaps IB", ".75", ".78", "+4%", ".83", ".83", "+0%"),
    ]
    for g, name, qv, qp, qd, dv, dp, dd in TABLE:
        text(slide, X3, y + Inches(0.06), COL_G, Inches(0.62),
             [[(g, 24, True, CRIMSON)]])
        text(slide, X3 + COL_G, y + Inches(0.06), COL_N, Inches(0.62),
             [[(name, 22, False, INK)]])
        for i, (v0, v1, d) in enumerate(((qv, qp, qd), (dv, dp, dd))):
            text(slide, X3 + COL_G + COL_N + i * COL_D, y, COL_D, Inches(0.72),
                 [[(f"{v0}→", 20, False, GRAY), (v1, 24, True, BLUE),
                   (f" {d}", 20, True, BLUE)]], align=PP_ALIGN.RIGHT)
        y += Inches(0.82)
        rule(slide, X3, y - Inches(0.06), W3, Pt(0.8), HAIR)
    y += Inches(0.3)
    text(slide, X3, y, W3, Inches(1.6),
         [[("The gain tracks the gap. ", 24, True, CRIMSON),
           ("R@100 improved or matched in all 16 BQ configurations; best in all "
            "14 at R@10.", 24, False, INK)]],
         line_spacing=1.3)
    y += Inches(1.55)
    # Size the QPS panel to the room left in the column and centre it. At full
    # column width it stood 7.2 in tall and ran under the row-2 cards.
    CAP_H = Inches(0.95)
    fig_h = (R1_Y + R1_H) - y - CAP_H - Inches(0.2)
    fig_w = Emu(int(fig_h * ASPECT["asset_fig3c.png"]))
    if fig_w > W3:
        fig_w, fig_h = W3, Emu(int(W3 / ASPECT["asset_fig3c.png"]))
    slide.shapes.add_picture(str(HERE / "asset_fig3c.png"),
                             X3 + Emu(int((W3 - fig_w) / 2)), y, fig_w, fig_h)
    y += fig_h + Inches(0.12)
    text(slide, X3, y, W3, CAP_H,
         [[("QPS tracks Vanilla at every n_{probe} — PMC adds no per-query work.",
            21, False, GRAY, True)]], line_spacing=1.15, align=PP_ALIGN.CENTER)

    # ================= Row 2 =================
    R2_Y = R1_Y + R1_H + Inches(0.85)
    rule(slide, M, R2_Y - Inches(0.5), CW, Pt(1.2), HAIR)
    R2_H = PAGE_H - R2_Y - Inches(1.75)
    W4 = Emu(int(CW * 0.365))
    W5 = Emu(int(CW * 0.285))
    W6 = CW - W4 - W5 - 2 * GUT
    X4, X5, X6 = M, M + W4 + GUT, M + W4 + GUT + W5 + GUT
    for xd in (X5 - Emu(int(GUT / 2)), X6 - Emu(int(GUT / 2))):
        solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, xd, R2_Y, Pt(1.0), R2_H), HAIR)

    # ---- 4. Mechanism ----
    y = heading(slide, X4, R2_Y, W4, "4", "Why it works")
    y += picture(slide, "asset_fig3b.png", X4, y, W4) + Inches(0.55)
    text(slide, X4, y, W4, Inches(0.95),
         [[("Correcting 5% of dimensions can already be enough.", 34, True, BLUE)]],
         line_spacing=1.15)
    y += Inches(1.45)
    text(slide, X4, y, W4, Inches(2.4),
         [[("CLIP’s gap energy is concentrated (top 10% ≈ 86–92%), so selective PMC "
            "on the highest-|g_{i}| dimensions recovers peak recall. ImageBind’s "
            "diffuse gap (≈72%) needs the full vector — exactly what the flip-risk "
            "analysis predicts.", 26, False, INK)]], line_spacing=1.32)

    # ---- 5. Ablation ----
    y = heading(slide, X5, R2_Y, W5, "5", "Where to correct")
    text(slide, X5, y - Inches(0.2), W5, Inches(0.6),
         [[("MSCOCO · CLIP-B/32 · IVF-RaBitQ · R@100 (t→i)", 21, False, GRAY)]])
    y += Inches(0.6)
    abw = W5 - Inches(2.0)
    for label, v, col, val in [("Vanilla", 0.578, GRAY, ".578"),
                               ("Query-only", 0.541, GRAY, ".541"),
                               ("Both sides", 0.599, GRAY, ".599"),
                               ("DB-only  (PMC)", 0.637, BLUE, ".637")]:
        hbar(slide, X5, y, abw, (v - 0.45) / (0.637 - 0.45), Inches(0.95), col,
             label, val)
        y += Inches(1.25)
    y += Inches(0.65)
    text(slide, X5, y, W5, Inches(3.0),
         [[("Shifting both sides re-displaces the routing pivot.", 26, False, INK)],
          [("Correct the centroid where it controls both routing and code "
            "formation.", 29, True, CRIMSON)],
          [("Random, shuffled, sign-flipped and un-normalized controls all trail "
            "DB-PMC.", 24, False, GRAY)]],
         line_spacing=1.3, space_after=12)

    # ---- Takeaway ----
    solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, X6, R2_Y, W6, R2_H),
          CARD, HAIR, Pt(1.0))
    rule(slide, X6, R2_Y, W6, Pt(4.0), CRIMSON)
    tx, tw = X6 + Inches(0.6), W6 - Inches(1.2)
    text(slide, tx, R2_Y + Inches(0.6), tw, Inches(0.9),
         [[("Takeaway", 38, True, CRIMSON)]])
    text(slide, tx, R2_Y + Inches(1.9), tw, Inches(4.2),
         [[("A one-time, database-side centroid correction repairs IVF routing and "
            "binary quantization together — without changing the serving path.",
            33, True, INK)]], line_spacing=1.3)
    y = R2_Y + Inches(6.5)
    for t in ["Recall gains scale with the gap ‖g‖",
              "Zero query-time transform",
              "Zero additional index memory",
              "Validated at 407M-vector scale"]:
        solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, tx, y + Inches(0.15),
                                     Inches(0.28), Inches(0.28)), BLUE)
        text(slide, tx + Inches(0.58), y, tw - Inches(0.6), Inches(0.75),
             [[(t, 29, False, INK)]], anchor=MSO_ANCHOR.MIDDLE)
        y += Inches(1.25)

    # ================= Footer =================
    rule(slide, M, PAGE_H - Inches(1.4), CW, Pt(2.0), CRIMSON)
    text(slide, M, PAGE_H - Inches(1.08), CW, Inches(0.6),
         [[("CIKM ’26 · November 07–11, 2026 · Rome, Italy          "
            "DOI 10.1145/3799682.3840007          CC-BY 4.0          "
            "Supported by the NRF of Korea (MSIT), No. 00359638", 19, False, GRAY)]],
         align=PP_ALIGN.CENTER)

    out = HERE / "PMC_CIKM2026_poster.pptx"
    prs.save(str(out))
    print(f"saved {out}")


if __name__ == "__main__":
    build()
