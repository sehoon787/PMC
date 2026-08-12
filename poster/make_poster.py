"""Generate the CIKM 2026 poster for PMC (csp4014) as an editable PPTX.

A0 portrait (33.11 x 46.81 in), scale-to-fit any final size the conference
announces. Every box, bar and label is a native PowerPoint shape.

Layout (per the approved design):
  header 12%  - title / authors / QR
  row 1       - (1) Problem 28% | (2) Method 40% | (3) Main results 32%
  row 2       - (4) Why it works + selective PMC | DB-vs-Query ablation | (5) Takeaway
Palette: navy structure, orange = gap/problem, green = PMC, gray = vanilla.

Usage:  python make_poster.py   (writes PMC_CIKM2026_poster.pptx next to this file)
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent

NAVY = RGBColor(0x1A, 0x27, 0x44)
ORANGE = RGBColor(0xE8, 0x72, 0x0C)
GREEN = RGBColor(0x1F, 0x7A, 0x4D)
GRAY = RGBColor(0x8A, 0x8F, 0x98)
LIGHT = RGBColor(0xF2, 0xF4, 0xF7)
ORANGE_BG = RGBColor(0xFD, 0xEF, 0xE2)
GREEN_BG = RGBColor(0xE7, 0xF3, 0xEC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

PAGE_W, PAGE_H = Inches(33.11), Inches(46.81)


def solid(shape, color, line_color=None, line_w=None):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    if line_color is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line_color
        shape.line.width = line_w or Pt(1.5)
    shape.shadow.inherit = False


def text(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         line_spacing=1.0, space_after=0):
    """runs: list of paragraphs; each paragraph a list of (text, size, bold, color)."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        p.space_after = Pt(space_after)
        for t, size, bold, color in para:
            r = p.add_run()
            r.text = t
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.color.rgb = color
            r.font.name = "Helvetica Neue"
    return tb


def panel(slide, x, y, w, h, fill=WHITE, line=RGBColor(0xD8, 0xDD, 0xE4)):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    s.adjustments[0] = 0.025
    solid(s, fill, line, Pt(2.0))
    return s


def section_head(slide, x, y, w, label, color=NAVY, size=38):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y + Pt(6), Inches(0.28), Inches(0.75))
    solid(bar, color)
    text(slide, x + Inches(0.45), y, w - Inches(0.45), Inches(0.95),
         [[(label, size, True, color)]], line_spacing=1.02)


def hbar(slide, x, y, w_full, frac, h, color, label, value, vsize=22):
    """Horizontal bar with left label and right value."""
    lane = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w_full, h)
    solid(lane, LIGHT)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, Emu(int(w_full * frac)), h)
    solid(bar, color)
    text(slide, x + Inches(0.12), y, w_full - Inches(0.2), h,
         [[(label, vsize, False, WHITE if frac > 0.35 else NAVY)]],
         anchor=MSO_ANCHOR.MIDDLE)
    text(slide, x + w_full + Inches(0.12), y, Inches(1.5), h,
         [[(value, vsize, True, NAVY)]], anchor=MSO_ANCHOR.MIDDLE)


def build():
    prs = Presentation()
    prs.slide_width = PAGE_W
    prs.slide_height = PAGE_H
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, PAGE_W, PAGE_H)
    solid(bg, RGBColor(0xFA, 0xFB, 0xFC))

    M = Inches(0.85)                     # page margin
    CW = PAGE_W - 2 * M                  # content width

    # ---------------- Header ----------------
    head = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, PAGE_W, Inches(5.75))
    solid(head, NAVY)
    text(slide, M, Inches(0.65), CW - Inches(4.4), Inches(2.7),
         [[("PMC: Build-Time Per-Modality Centroid Correction", 74, True, WHITE)],
          [("for Cross-Modal Binary-Quantized Retrieval", 74, True, WHITE)]],
         line_spacing=1.02)
    text(slide, M, Inches(3.7), CW - Inches(4.4), Inches(0.9),
         [[("Se Hoon Kim   ·   Jun Hyung Lee   ·   Soonyoung Jung        ", 40, False, WHITE),
           ("Korea University", 40, True, RGBColor(0xBF, 0xD3, 0xF0))]])
    text(slide, M, Inches(4.6), CW - Inches(4.4), Inches(0.8),
         [[("CIKM 2026 · Rome, Italy       ", 28, False, RGBColor(0x9D, 0xAC, 0xC6)),
           ("One gap. Two failures. One build-time fix.", 30, True, ORANGE)]])
    # QR
    qr_bg = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                   PAGE_W - M - Inches(3.5), Inches(0.7),
                                   Inches(3.5), Inches(4.2))
    qr_bg.adjustments[0] = 0.06
    solid(qr_bg, WHITE)
    slide.shapes.add_picture(str(HERE / "asset_qr.png"),
                             PAGE_W - M - Inches(3.28), Inches(0.9),
                             Inches(3.06), Inches(3.06))
    text(slide, PAGE_W - M - Inches(3.5), Inches(4.0), Inches(3.5), Inches(0.8),
         [[("Paper + Code", 30, True, NAVY)],
          [("github.com/sehoon787/PMC", 16, False, GRAY)]],
         align=PP_ALIGN.CENTER)

    # ---------------- Row 1 geometry ----------------
    R1_Y = Inches(6.3)
    R1_H = Inches(24.6)
    GUT = Inches(0.55)
    W1 = Emu(int(CW * 0.27))             # problem
    W2 = Emu(int(CW * 0.395))            # method
    W3 = CW - W1 - W2 - 2 * GUT          # results
    X1, X2, X3 = M, M + W1 + GUT, M + W1 + GUT + W2 + GUT

    # ============ (1) PROBLEM ============
    panel(slide, X1, R1_Y, W1, R1_H, fill=ORANGE_BG,
          line=RGBColor(0xEC, 0xC8, 0xA6))
    px = X1 + Inches(0.55)
    pw = W1 - Inches(1.1)
    section_head(slide, px, R1_Y + Inches(0.5), pw,
                 "① Why BQ fails cross-modally", ORANGE)
    text(slide, px, R1_Y + Inches(2.15), pw, Inches(1.7),
         [[("Queries and database vectors do not share the same centroid.",
            40, True, NAVY)]], line_spacing=1.05)
    slide.shapes.add_picture(str(HERE / "asset_fig1.png"),
                             px, R1_Y + Inches(4.45), pw)
    text(slide, px, R1_Y + Inches(10.35), pw, Inches(0.8),
         [[("t-SNE (ImageBind): separated modality clusters (a) overlap after PMC (b).",
            22, False, NAVY)]])
    text(slide, px, R1_Y + Inches(11.55), pw, Inches(1.0),
         [[("One gap, two failures.", 44, True, ORANGE)]])
    for i, (t1, t2) in enumerate([
            ("IVF Routing", "queries probe the wrong inverted lists"),
            ("Binary Codes", "sign(x−c) flips: the boundary passes through c with zero error margin")]):
        by = R1_Y + Inches(12.85 + i * 2.5)
        b = panel(slide, px, by, pw, Inches(2.2), fill=WHITE,
                  line=ORANGE)
        text(slide, px + Inches(0.4), by + Inches(0.3), pw - Inches(0.8), Inches(1.7),
             [[(t1, 34, True, ORANGE)], [(t2, 24, False, NAVY)]], line_spacing=1.1)
    text(slide, px, R1_Y + Inches(18.4), pw, Inches(2.4),
         [[("Oracle test (MSCOCO · CLIP-L · RaBitQ)", 24, True, NAVY)],
          [("same-modality ‖g‖=0 :  R@100 = 0.71", 26, False, NAVY)],
          [("cross-modal ‖g‖=.82 :  R@100 = 0.54", 26, True, ORANGE)]],
         line_spacing=1.35)
    text(slide, px, R1_Y + Inches(21.3), pw, Inches(3.0),
         [[("Structured, not random:", 26, True, NAVY)],
          [("top 10% of dimensions carry ≈90% of ‖g‖² (CLIP-L). "
            "Flip risk grows with |gᵢ| — E[F] ≈ αΣᵢ pᵢ(0)|gᵢ|.", 24, False, NAVY)]],
         line_spacing=1.2)

    # ============ (2) METHOD ============
    panel(slide, X2, R1_Y, W2, R1_H, fill=GREEN_BG,
          line=RGBColor(0xB9, 0xDC, 0xC9))
    mx = X2 + Inches(0.6)
    mw = W2 - Inches(1.2)
    section_head(slide, mx, R1_Y + Inches(0.5), mw,
                 "② PMC fixes the boundary at the source", GREEN)

    steps = [
        ("CALIBRATE", "estimate the gap from a small paired sample",
         "g = μq − μx        (25 samples suffice)"),
        ("OFFLINE BUILD", "shift every database vector, then build the BQ index",
         "x′ = (x + g) / ‖x + g‖   →   IVF-RaBitQ / BBQ / BinaryFlat"),
        ("ONLINE SERVING", "search the corrected index — the query is untouched",
         "q′ = q        zero query-time transform"),
    ]
    sy = R1_Y + Inches(2.2)
    for i, (h1, h2, eq) in enumerate(steps):
        card = panel(slide, mx, sy, mw, Inches(4.15), fill=WHITE, line=GREEN)
        num = slide.shapes.add_shape(MSO_SHAPE.OVAL, mx + Inches(0.35),
                                     sy + Inches(0.5), Inches(1.15), Inches(1.15))
        solid(num, GREEN)
        text(slide, mx + Inches(0.35), sy + Inches(0.52), Inches(1.15), Inches(0.95),
             [[(str(i + 1), 44, True, WHITE)]], align=PP_ALIGN.CENTER)
        text(slide, mx + Inches(1.85), sy + Inches(0.45), mw - Inches(2.2), Inches(3.5),
             [[(h1, 38, True, GREEN)],
              [(h2, 25, False, NAVY)],
              [(eq, 30, True, NAVY)]], line_spacing=1.35)
        sy += Inches(4.15)
        if i < 2:
            ar = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW,
                                        mx + mw / 2 - Inches(0.45), sy + Inches(0.12),
                                        Inches(0.9), Inches(0.75))
            solid(ar, GREEN)
            sy += Inches(1.0)

    band = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, mx,
                                  sy + Inches(0.25), mw, Inches(1.5))
    band.adjustments[0] = 0.12
    solid(band, GREEN)
    text(slide, mx, sy + Inches(0.42), mw, Inches(1.2),
         [[("α = 1  →  no query-time transformation, no extra index memory",
            34, True, WHITE)]], align=PP_ALIGN.CENTER)

    gy = sy + Inches(2.2)
    text(slide, mx, gy, mw, Inches(2.6),
         [[("General form:  x′ = (x + αg)/‖x + αg‖ ,   α ∈ [0,1]", 26, False, NAVY)],
          [("α = 0 is query-only mean shift — it leaves IVF centroids and codes "
            "misaligned. α = 1 rewrites the centroid where it is BOTH the routing "
            "pivot and the code boundary.", 25, False, NAVY)],
          [("The α-sweep confirms α = 1 best or near-best in all settings.",
            25, True, GREEN)]],
         line_spacing=1.25, space_after=8)

    # ============ (3) RESULTS ============
    panel(slide, X3, R1_Y, W3, R1_H, fill=WHITE)
    rx = X3 + Inches(0.55)
    rw = W3 - Inches(1.1)
    section_head(slide, rx, R1_Y + Inches(0.5), rw, "③ Main results", NAVY)

    text(slide, rx, R1_Y + Inches(1.75), rw, Inches(2.3),
         [[("407M", 110, True, GREEN)],
          [("vectors · LAION-400M · CLIP", 28, False, NAVY)]],
         line_spacing=0.95)
    # LAION bars
    by = R1_Y + Inches(4.35)
    text(slide, rx, by, rw, Inches(0.6),
         [[("R@100, no reranking (n_list=80K, n_probe=256)", 24, True, NAVY)]])
    bw = rw - Inches(1.7)
    hbar(slide, rx, by + Inches(0.75), bw, 0.108 / 0.143, Inches(0.72), GRAY,
         "Vanilla", ".108", 24)
    hbar(slide, rx, by + Inches(1.65), bw, 1.0, Inches(0.72), GREEN,
         "PMC  (+32%)", ".143", 24)
    text(slide, rx, by + Inches(2.6), rw, Inches(0.55),
         [[("with exact reranking K′=400:  .198 → .277  (+40%)", 26, True, GREEN)]])

    # three result cards
    cards = [
        ("MSCOCO · CLIP-L · i→t", ".47", ".63", "+34%"),
        ("Flickr30K · CLIP-L · i→t", ".33", ".48", "+45%"),
        ("MSCOCO · ImageBind · t→i", ".67", ".75", "+12%"),
    ]
    cy = by + Inches(3.5)
    for label, v0, v1, d in cards:
        c = panel(slide, rx, cy, rw, Inches(1.9), fill=LIGHT, line=RGBColor(0xD8, 0xDD, 0xE4))
        text(slide, rx + Inches(0.35), cy + Inches(0.22), rw - Inches(0.7), Inches(0.6),
             [[(label, 25, True, NAVY)]])
        text(slide, rx + Inches(0.35), cy + Inches(0.85), rw - Inches(0.7), Inches(0.95),
             [[(f"{v0}  →  ", 40, False, GRAY), (v1, 44, True, GREEN),
               (f"    {d}", 34, True, GREEN)]])
        cy += Inches(2.15)

    text(slide, rx, cy + Inches(0.1), rw, Inches(0.6),
         [[("5 benchmarks · 3 encoders · 4 BQ index types", 24, True, NAVY)]])
    text(slide, rx, cy + Inches(0.75), rw, Inches(1.3),
         [[("R@100 improved or matched in all 16 BQ configs; ", 24, False, NAVY),
           ("best in all 14 configs at R@10.", 24, True, GREEN)]],
         line_spacing=1.2)
    # QPS panel
    slide.shapes.add_picture(str(HERE / "asset_fig3c.png"),
                             rx, cy + Inches(2.05), rw)
    text(slide, rx, cy + Inches(2.05) + Emu(int(rw * 929 / 1191)) + Inches(0.1),
         rw, Inches(0.7),
         [[("QPS tracks Vanilla at every n_probe — PMC adds no per-query work.",
            23, False, NAVY)]])

    # ---------------- Row 2 ----------------
    R2_Y = R1_Y + R1_H + Inches(0.55)
    R2_H = PAGE_H - R2_Y - Inches(1.6)
    W4 = Emu(int(CW * 0.40))
    W5 = Emu(int(CW * 0.315))
    W6 = CW - W4 - W5 - 2 * GUT
    X4, X5, X6 = M, M + W4 + GUT, M + W4 + GUT + W5 + GUT

    # (4) why it works — selective PMC
    panel(slide, X4, R2_Y, W4, R2_H, fill=WHITE)
    wx = X4 + Inches(0.55)
    ww = W4 - Inches(1.1)
    section_head(slide, wx, R2_Y + Inches(0.45), ww,
                 "④ Why it works: concentrated gap", NAVY)
    slide.shapes.add_picture(str(HERE / "asset_fig3b.png"),
                             wx, R2_Y + Inches(1.95), ww)
    fig_h = Emu(int(ww * 929 / 1188))
    text(slide, wx, R2_Y + Inches(2.2) + fig_h, ww, Inches(1.1),
         [[("Only 5% of dimensions can be enough.", 40, True, GREEN)]])
    text(slide, wx, R2_Y + Inches(3.3) + fig_h, ww, Inches(2.2),
         [[("CLIP's gap energy is highly concentrated: correcting only the "
            "highest-|gᵢ| dimensions recovers peak recall (top 10% ≈ 86–92% of "
            "energy). ImageBind's diffuse gap (≈72%) needs the full vector — "
            "exactly what the flip-risk analysis predicts.", 25, False, NAVY)]],
         line_spacing=1.25)

    # (5) ablation
    panel(slide, X5, R2_Y, W5, R2_H, fill=WHITE)
    ax = X5 + Inches(0.55)
    aw = W5 - Inches(1.1)
    section_head(slide, ax, R2_Y + Inches(0.45), aw,
                 "Where should we correct?", NAVY)
    text(slide, ax, R2_Y + Inches(1.55), aw, Inches(0.6),
         [[("MSCOCO · CLIP-B/32 · IVF-RaBitQ · R@100 (t→i)", 24, False, NAVY)]])
    rows = [("Vanilla", 0.578, GRAY, ".578"),
            ("Query-only", 0.541, GRAY, ".541"),
            ("Both sides", 0.599, GRAY, ".599"),
            ("DB-only  ★ PMC", 0.637, GREEN, ".637")]
    aby = R2_Y + Inches(2.3)
    abw = aw - Inches(1.6)
    for label, v, col, val in rows:
        hbar(slide, ax, aby, abw, (v - 0.45) / (0.637 - 0.45), Inches(0.78),
             col, label, val, 24)
        aby += Inches(1.0)
    text(slide, ax, aby + Inches(0.25), aw, Inches(2.6),
         [[("Shifting both sides re-displaces the routing pivot.", 25, False, NAVY)],
          [("Correct the centroid where it controls both routing and code formation.",
            30, True, GREEN)],
          [("Random / shuffled / sign-flipped / un-normalized controls all trail "
            "DB-PMC — the direction of g matters.", 23, False, GRAY)]],
         line_spacing=1.25, space_after=10)

    # (6) takeaway
    panel(slide, X6, R2_Y, W6, R2_H, fill=NAVY, line=NAVY)
    tx = X6 + Inches(0.6)
    tw = W6 - Inches(1.2)
    text(slide, tx, R2_Y + Inches(0.55), tw, Inches(0.9),
         [[("⑤ PMC in one sentence", 40, True, WHITE)]])
    text(slide, tx, R2_Y + Inches(1.85), tw, Inches(3.2),
         [[("A one-time DB-side centroid correction repairs both IVF routing and "
            "binary quantization — without changing the serving path.",
            34, True, RGBColor(0xDD, 0xE7, 0xF5))]], line_spacing=1.2)
    marks = [("↑", "Recall gains scale with ‖g‖"),
             ("0", "query-time transform"),
             ("0", "additional index memory"),
             ("✓", "validated at 407M-vector scale")]
    my = R2_Y + Inches(5.2)
    for sym, t in marks:
        chip = slide.shapes.add_shape(MSO_SHAPE.OVAL, tx, my, Inches(1.0), Inches(1.0))
        solid(chip, GREEN)
        text(slide, tx, my + Inches(0.12), Inches(1.0), Inches(0.8),
             [[(sym, 36, True, WHITE)]], align=PP_ALIGN.CENTER)
        text(slide, tx + Inches(1.35), my + Inches(0.08), tw - Inches(1.4), Inches(0.9),
             [[(t, 28, True, WHITE)]], anchor=MSO_ANCHOR.MIDDLE)
        my += Inches(1.3)

    # ---------------- Footer ----------------
    text(slide, M, PAGE_H - Inches(1.15), CW, Inches(0.7),
         [[("CIKM ’26 · November 07–11, 2026 · Rome, Italy      "
            "DOI 10.1145/3799682.3840007      CC-BY 4.0      "
            "Supported by NRF Korea (MSIT) No. 00359638", 20, False, GRAY)]],
         align=PP_ALIGN.CENTER)

    out = HERE / "PMC_CIKM2026_poster.pptx"
    prs.save(str(out))
    print(f"saved {out}")


if __name__ == "__main__":
    build()
