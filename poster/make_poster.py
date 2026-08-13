"""Generate the CIKM 2026 poster for PMC (csp4014) as an editable PPTX.

v3 — dense and self-contained, in the style of the CIKM / ICLR / ECCV posters
the authors referenced: an explicit Background, Contributions, Setup,
Conclusion and full explanatory prose, so a visitor can read the whole
argument without the presenter standing there.

Visual language follows the lab's own deck: white ground, KU-crimson serif
headings (#8B0029, sampled), hairline rules, IE-LAB blue (#0D4DA3) as the
single accent meaning PMC, gray for baselines. The official CIKM '26 mark
sits in the footer band with the conference details.

Layout is a flow model: each column owns a top-down cursor and every block
reports the height it consumed, so a block can never silently overrun its
neighbour. build() prints per-column fill against the available height.

A0 portrait (33.11 x 46.81 in); rescale once CIKM publishes the poster spec.

Usage:  python make_poster.py
"""

import re
from pathlib import Path

from PIL import ImageFont

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent

CRIMSON = RGBColor(0x8B, 0x00, 0x29)
BLUE = RGBColor(0x0D, 0x4D, 0xA3)
INK = RGBColor(0x1F, 0x1F, 0x1F)
GRAY = RGBColor(0x6F, 0x6F, 0x6F)
HAIR = RGBColor(0xC8, 0xC8, 0xC8)
CARD = RGBColor(0xF6, 0xF5, 0xF3)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SERIF = "Times New Roman"

PAGE_W, PAGE_H = Inches(33.11), Inches(46.81)
MARGIN = Inches(0.85)
HEADER_H = Inches(5.25)
FOOTER_H = Inches(2.05)

ASPECT = {                        # native pixel aspect of each embedded asset
    "asset_fig1.png": 2400 / 1193,
    "asset_fig3b.png": 1188 / 929,
    "asset_fig3c.png": 1206 / 929,
    "asset_ku.jpg": 157 / 49,
    "asset_ielab.png": 177 / 71,
    "asset_cikm.png": 250 / 193,
}

# Line breaking is measured with the real Times metrics (PIL) rather than
# estimated, so the height a block reserves matches what PowerPoint sets and
# no block can overlap its neighbour.
LINE_FACTOR = 1.20      # PowerPoint sets leading at 1.2x the font size
_TIMES = {False: "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
          True: "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"}
_MPX = 64               # metric render size; widths scale linearly from it
_FONTS = {}


def solid(shape, color, line_color=None, line_w=None):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    if line_color is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line_color
        shape.line.width = line_w or Pt(1.0)
    shape.shadow.inherit = False


def _width_pt(text, size, bold):
    if bold not in _FONTS:
        _FONTS[bold] = ImageFont.truetype(_TIMES[bold], _MPX)
    return _FONTS[bold].getlength(text) * size / _MPX


def wrap_lines(runs, width_pt):
    """Greedy word-wrap over mixed-format runs; returns each line's max size."""
    lines, cur_w, cur_max = [], 0.0, 0.0
    for text, size, bold in runs:
        for tok in re.findall(r"\s+|\S+", text):
            w = _width_pt(tok, size, bold)
            if tok.isspace():
                if cur_w:
                    cur_w += w
                continue
            if cur_w and cur_w + w > width_pt:
                lines.append(cur_max)
                cur_w, cur_max = w, size
            else:
                cur_w += w
                cur_max = max(cur_max, size)
    if cur_max:
        lines.append(cur_max)
    return lines or [runs[0][1]]


def add_runs(p, text, size, bold, color, italic=False):
    """Emit runs, rendering `_{...}` as a true subscript."""
    parts = re.split(r"_\{([^}]*)\}", text)
    for i, part in enumerate(parts):
        if not part:
            continue
        r = p.add_run()
        r.text = part
        r.font.size = Pt(size * 0.72) if i % 2 else Pt(size)
        r.font.bold, r.font.italic, r.font.name = bold, italic, SERIF
        r.font.color.rgb = color
        if i % 2:
            r.font._rPr.set("baseline", "-25000")


def plain(text):
    return re.sub(r"_\{([^}]*)\}", r"\1", text)


def _runs_of(para, size, bold, color, italic):
    return para if isinstance(para[0], tuple) else [(para, size, bold, color, italic)]


def cell(slide, runs, x, y, w, h, align=PP_ALIGN.LEFT, wrap=False):
    """One line of mixed runs, vertically centred — used by tables and bars."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    for t, sz, bd, cl in runs:
        add_runs(p, t, sz, bd, cl)
    return tb


class Col:
    """A poster column with a top-down cursor."""

    def __init__(self, slide, x, w, top, bottom):
        self.s, self.x, self.w, self.y, self.bottom = slide, x, w, top, bottom

    def gap(self, inches):
        self.y += Inches(inches)

    def rule(self, weight=Pt(1.0), color=HAIR, pad=0.10):
        self.y += Inches(pad)
        solid(self.s.shapes.add_shape(MSO_SHAPE.RECTANGLE, self.x, self.y, self.w, weight),
              color)
        self.y += weight + Inches(pad)

    def text(self, paras, size=21, color=INK, bold=False, italic=False,
             align=PP_ALIGN.LEFT, spacing=1.22, after=5, x=None, w=None):
        x = self.x if x is None else x
        w = self.w if w is None else w
        width_pt = w / 12700.0
        total_pt = 0.0
        for p in paras:
            runs = [(plain(r[0]), r[1] if len(r) > 1 else size,
                     r[2] if len(r) > 2 else bold)
                    for r in _runs_of(p, size, bold, color, italic)]
            total_pt += sum(ls * spacing * LINE_FACTOR
                            for ls in wrap_lines(runs, width_pt)) + after
        h = Inches(total_pt / 72.0)
        tb = self.s.shapes.add_textbox(x, self.y, w, h)
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        for i, para in enumerate(paras):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            p.line_spacing = spacing
            p.space_after = Pt(after)
            for run in _runs_of(para, size, bold, color, italic):
                add_runs(p, run[0],
                         run[1] if len(run) > 1 else size,
                         run[2] if len(run) > 2 else bold,
                         run[3] if len(run) > 3 else color,
                         run[4] if len(run) > 4 else italic)
        self.y += h
        return h

    def heading(self, num, title):
        self.gap(0.12)
        self.text([[(f"{num}.  {title}", 31, True, CRIMSON)]], spacing=1.05, after=0)
        self.rule(Pt(2.0), CRIMSON, pad=0.06)

    def bullets(self, items, size=21, marker=CRIMSON, gap_in=0.09):
        for it in items:
            solid(self.s.shapes.add_shape(MSO_SHAPE.RECTANGLE, self.x + Inches(0.03),
                                          self.y + Inches(0.12), Inches(0.16), Inches(0.16)),
                  marker)
            self.text([it], size=size, x=self.x + Inches(0.38),
                      w=self.w - Inches(0.38), spacing=1.20, after=0)
            self.gap(gap_in)

    def figure(self, name, frac=1.0):
        w = Emu(int(self.w * frac))
        h = Emu(int(w / ASPECT[name]))
        self.s.shapes.add_picture(str(HERE / name),
                                  self.x + Emu(int((self.w - w) / 2)), self.y, w, h)
        self.y += h

    def _boxed(self, paras, size, fill, pad, color, bold, line):
        # The box is created before the text so it sits behind it in z-order;
        # its height is corrected once the text has advanced the cursor.
        start = self.y
        box = self.s.shapes.add_shape(MSO_SHAPE.RECTANGLE, self.x, start, self.w,
                                      Inches(1))
        solid(box, fill, line, Pt(1.0) if line else None)
        self.gap(pad)
        self.text(paras, size=size, color=color, bold=bold, x=self.x + Inches(pad),
                  w=self.w - Inches(2 * pad), spacing=1.28, after=4)
        self.gap(pad)
        box.height = self.y - start

    def card(self, paras, size=21, pad=0.28):
        self._boxed(paras, size, CARD, pad, INK, False, HAIR)

    def band(self, paras, size=24, pad=0.26):
        self._boxed(paras, size, BLUE, pad, WHITE, True, None)

    def hbar(self, frac, color, label, value, h_in=0.60, val_w=1.45):
        h = Inches(h_in)
        lane_w = self.w - Inches(val_w)
        solid(self.s.shapes.add_shape(MSO_SHAPE.RECTANGLE, self.x, self.y, lane_w, h),
              WHITE, HAIR, Pt(1.0))
        solid(self.s.shapes.add_shape(MSO_SHAPE.RECTANGLE, self.x, self.y,
                                      Emu(int(lane_w * frac)), h), color)
        cell(self.s, [(label, 21, False, WHITE if frac > 0.45 else INK)],
             self.x + Inches(0.16), self.y, lane_w - Inches(0.3), h)
        cell(self.s, [(value, 21, True, INK)],
             self.x + lane_w + Inches(0.12), self.y, Inches(val_w - 0.12), h)
        self.y += h + Inches(0.09)


ABSTRACT = (
    "Approximate nearest neighbor search is a core operator in large-scale retrieval, where"
    " vector quantization is widely used to reduce memory cost. Binary quantization methods"
    " encode each dimension as a single bit, but they assume queries and database vectors s"
    "hare a common centroid. This assumption fails in cross-modal retrieval, where each mod"
    "ality clusters around a different centroid, producing a modality gap that corrupts the"
    " centroid's dual role as inverted-file routing pivot and sign-bit decision boundary. W"
    "e propose Per-Modality Centroid Correction (PMC), which shifts database vectors toward"
    " the query centroid at build time, rewriting routing and code boundaries at the source"
    " rather than merely adjusting queries at serving time. A selective variant confirms th"
    "at concentrated gap energy in a few dimensions drives most recall loss. PMC adds no in"
    "dex memory or serving overhead; at α=1, the correction is fully absorbed into the inde"
    "x with zero query-time cost. Experiments on five cross-modal benchmarks, three encoder"
    "s, and four binary-quantized index types show gains over uncorrected and query-shifted"
    " baselines that scale with modality-gap magnitude, up to a 400M-vector deployment.")


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = PAGE_W, PAGE_H
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, PAGE_W, PAGE_H), WHITE)
    M, CW = MARGIN, PAGE_W - 2 * MARGIN

    # ============================= header =============================
    KU_H = IE_H = Inches(1.15)
    KU_W = Emu(int(KU_H * ASPECT["asset_ku.jpg"]))
    IE_W = Emu(int(IE_H * ASPECT["asset_ielab.png"]))
    R = PAGE_W - M
    slide.shapes.add_picture(str(HERE / "asset_ielab.png"), R - IE_W, Inches(0.65), IE_W, IE_H)
    slide.shapes.add_picture(str(HERE / "asset_ku.jpg"),
                             R - IE_W - Inches(0.5) - KU_W, Inches(0.65), KU_W, KU_H)
    QR = Inches(2.05)
    slide.shapes.add_picture(str(HERE / "asset_qr.png"), R - QR, Inches(2.20), QR, QR)
    cell(slide, [("Paper + code:  github.com/sehoon787/PMC", 17, False, GRAY)],
         R - Inches(6.5), Inches(4.35), Inches(6.5), Inches(0.45), PP_ALIGN.RIGHT)

    head = Col(slide, M, CW - Inches(9.0), Inches(0.72), Inches(5.1))
    head.text([[("PMC: Build-Time Per-Modality Centroid Correction", 55, True, CRIMSON)],
               [("for Cross-Modal Binary-Quantized Retrieval", 55, True, CRIMSON)]],
              spacing=1.10, after=0)
    head.gap(0.26)
    head.text([[("Se Hoon Kim,   Jun Hyung Lee,   Soonyoung Jung*", 30, False, INK)]],
              after=0)
    head.gap(0.08)
    head.text([[("Department of Computer Science and Engineering, Korea University  ·  "
                 "Intelligence Engineering Lab", 22, False, GRAY)]], after=0)
    solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, M, HEADER_H, CW, Pt(3.0)), CRIMSON)

    # ============================= abstract ===========================
    ab = Col(slide, M, CW, HEADER_H + Inches(0.42), PAGE_H)
    ab.card([[("Abstract.  ", 23, True, CRIMSON), (ABSTRACT, 23, False, INK)]],
            size=23, pad=0.26)

    # ============================= columns ============================
    TOP = ab.y + Inches(0.42)
    BOT = PAGE_H - FOOTER_H - Inches(0.30)
    GUT = Inches(0.70)
    W = Emu(int((CW - 2 * GUT) / 3))
    c1 = Col(slide, M, W, TOP, BOT)
    c2 = Col(slide, M + W + GUT, W, TOP, BOT)
    c3 = Col(slide, M + 2 * (W + GUT), W, TOP, BOT)
    for cx in (c2.x - Emu(int(GUT / 2)), c3.x - Emu(int(GUT / 2))):
        solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, cx, TOP, Pt(1.0), BOT - TOP), HAIR)

    # ------------------------------ column 1 --------------------------
    c1.heading("1", "Background")
    c1.text([[("CLIP and ImageBind map images, text and audio into one shared vector "
               "space, so a text query can retrieve an image directly. At deployment "
               "the index, not the model, is the bottleneck: 407M float32 vectors of "
               "512 dimensions occupy 833 GB.", 21, False, INK)]])
    c1.gap(0.06)
    c1.bullets([
        [("Vector quantization ", 21, True, INK),
         ("compresses each vector to a few bytes so the index fits in memory.",
          21, False, INK)],
        [("Binary quantization (BQ) ", 21, True, INK),
         ("is the extreme point — one sign bit per dimension, 32× smaller. RaBitQ "
          "bounds its distortion; Lucene ships it as BBQ.", 21, False, INK)],
        [("Every BQ method encodes ", 21, False, INK), ("sign(x − c)", 21, True, BLUE),
         (" against a centroid c — also the pivot routing a query to inverted lists.",
          21, False, INK)],
    ])

    c1.heading("2", "The problem: the modality gap")
    c1.text([[("BQ assumes queries and database vectors share that centroid. In "
               "cross-modal retrieval they do not — each modality clusters around its "
               "own centre, an offset known as the ", 21, False, INK),
              ("modality gap", 21, True, CRIMSON),
              (".  g = μ_{q} − μ_{x} is stable and large: ‖g‖ = .61–.82 on our benchmarks.",
               21, False, INK)]])
    c1.gap(0.10)
    c1.figure("asset_fig1.png", 0.80)
    c1.gap(0.06)
    c1.text([[("t-SNE of ImageBind embeddings: (a) each modality forms its own "
               "cluster; (b) after PMC the pairs overlap.",
               18, False, GRAY, True)]], size=18, spacing=1.15, after=0)
    c1.gap(0.20)
    c1.text([[("One gap, two failures", 29, True, CRIMSON)]], after=0)
    c1.gap(0.16)
    c1.bullets([
        [("IVF routing. ", 21, True, INK),
         ("The shifted centroid steers queries into the wrong inverted lists, so the "
          "true neighbours are never scanned.", 21, False, INK)],
        [("Binary codes. ", 21, True, INK),
         ("With one bit the decision boundary passes exactly through c with zero "
          "error margin, so any systematic offset flips the bits nearest it.",
          21, False, INK)],
    ])
    c1.gap(0.04)
    c1.card([[("Oracle test — MSCOCO, CLIP-L, RaBitQ", 21, True, INK)],
             [("same-modality  (‖g‖ = 0)           R@100 = 0.71", 22, False, INK)],
             [("cross-modal  (‖g‖ = .82)           R@100 = 0.54", 22, True, CRIMSON)],
             [("Removing the gap alone recovers 17 points: the loss is the gap, not "
               "the quantizer.", 19, False, GRAY, True)]])
    c1.gap(0.16)
    c1.text([[("The corruption is structured, not random. ", 21, True, INK),
              ("On CLIP-L the top 10% of dimensions carry ≈90% of ‖g‖², and a "
               "first-order expansion gives the expected number of flipped bits as a "
               "density-weighted ℓ₁ norm of the gap, E[F] ≈ α Σᵢ pᵢ(0)·|gᵢ|. Flip "
               "risk therefore grows with |gᵢ|, so a few dimensions dominate both the "
               "damage and the repair — and Hamming distances are biased "
               "systematically rather than noisily.", 21, False, INK)]])

    c1.heading("3", "Every binary index has the same wound")
    c1.text([[("R@100, Vanilla → PMC, at one operating point. Rotation (BBQ-style) "
               "spreads the concentrated gap energy across all bits, so RotatedBinary "
               "collapses without correction — and gains most from it.",
               20, False, INK)]], size=20)
    c1.gap(0.12)
    T1_W = Emu(int((c1.w - Inches(2.35)) / 2))
    for t, xx, ww, al in (("Method", c1.x, Inches(2.35), PP_ALIGN.LEFT),
                          ("MSCOCO t→i", c1.x + Inches(2.35), T1_W, PP_ALIGN.RIGHT),
                          ("AudioCaps a→t", c1.x + Inches(2.35) + T1_W, T1_W,
                           PP_ALIGN.RIGHT)):
        cell(slide, [(t, 19, True, GRAY)], xx, c1.y, ww, Inches(0.46), al)
    c1.y += Inches(0.48)
    c1.rule(Pt(1.5), INK, pad=0.02)
    for m, a, b, c_, d in (("BinaryFlat", ".51", ".57", ".51", ".64"),
                           ("BinaryIVF", ".51", ".57", ".50", ".63"),
                           ("RotatedBinary", ".29", ".58", ".63", ".69"),
                           ("RaBitQ", ".58", ".64", ".67", ".75")):
        rh = Inches(0.56)
        cell(slide, [(m, 20, False, INK)], c1.x, c1.y, Inches(2.35), rh)
        cell(slide, [(f"{a} → ", 19, False, GRAY), (b, 21, True, BLUE)],
             c1.x + Inches(2.35), c1.y, T1_W, rh, PP_ALIGN.RIGHT)
        cell(slide, [(f"{c_} → ", 19, False, GRAY), (d, 21, True, BLUE)],
             c1.x + Inches(2.35) + T1_W, c1.y, T1_W, rh, PP_ALIGN.RIGHT)
        c1.y += rh
        solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, c1.x, c1.y, c1.w, Pt(0.8)), HAIR)
    c1.gap(0.18)
    c1.text([[("PMC improves or matches R@100 in all 16 method × direction "
               "configurations. The wound is in the centroid, not in any one codec, "
               "so the same one-line fix applies to every sign(·−c) method and to "
               "schemes such as LSH.", 20, False, INK)]], size=20)

    c1.heading("4", "How PMC differs from prior work")
    c1.bullets([
        [("Graph-side methods ", 20, True, INK),
         ("(OOD-DiskANN, RoarGraph, DEG) rewire connectivity for distribution shift, "
          "but assume uncompressed vectors.", 20, False, INK)],
        [("Drift adaptation ", 20, True, INK),
         ("(DeDrift, TCR) retrains or adapts at test time — an online cost PMC does "
          "not pay.", 20, False, INK)],
        [("Gap-aware calibration ", 20, True, INK),
         ("rescales scores for mixed-modality ranking, but never enters code "
          "formation and still shifts the query at serving time.", 20, False, INK)],
    ], size=20)
    c1.gap(0.02)
    c1.text([[("No prior work addresses how the modality gap degrades ", 20, False, INK),
              ("compressed", 20, True, CRIMSON),
              (" ANN indexes, where centroid misalignment corrupts routing and codes "
               "at once.", 20, False, INK)]], size=20)

    c1.heading("5", "Contributions")
    c1.bullets([
        [("We show the modality gap corrupts IVF routing and binary codes together, "
          "with recall loss scaling with gap magnitude.", 21, False, INK)],
        [("We find the corruption concentrates in a few high-|gᵢ| dimensions; "
          "correcting only those recovers most of the loss.", 21, False, INK)],
        [("We propose PMC, a zero-cost build-time correction at α = 1; the ablation "
          "confirms database-side over query-side correction.", 21, False, INK)],
    ], marker=BLUE)

    # ------------------------------ column 2 --------------------------
    c2.heading("6", "PMC — correct at the source")
    for i, (h1, h2, eq, note) in enumerate([
            ("Calibrate", "estimate the gap from a small paired sample",
             "g  =  μ_{q} − μ_{x}", "25 samples suffice"),
            ("Build  (offline)", "shift every database vector, renormalize, then build "
             "the index exactly as before", "x′  =  (x + α g) / ‖x + α g‖",
             "IVF-RaBitQ · BBQ · BinaryFlat"),
            ("Serve  (online)", "search the corrected index — the query is untouched",
             "q′  =  q", "at α = 1")]):
        top = c2.y
        c2.gap(0.05)
        c2.text([[(f"{i + 1}    ", 29, True, BLUE), (h1, 29, True, INK)]],
                x=c2.x + Inches(0.38), w=c2.w - Inches(0.42), after=2)
        c2.text([[(h2, 20, False, GRAY)]], size=20, x=c2.x + Inches(0.38),
                w=c2.w - Inches(0.42), spacing=1.18, after=2)
        c2.text([[(eq, 25, True, INK), ("      " + note, 18, False, GRAY)]],
                x=c2.x + Inches(0.38), w=c2.w - Inches(0.42), after=0)
        c2.gap(0.09)
        solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, c2.x, top, Pt(4.5), c2.y - top),
              BLUE)
        if i < 2:
            solid(slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW,
                                         c2.x + Emu(int(c2.w / 2)) - Inches(0.42),
                                         c2.y + Inches(0.04), Inches(0.84), Inches(0.55)),
                  BLUE)
            c2.gap(0.72)
    c2.gap(0.10)
    c2.band([[("At α = 1 the correction is absorbed into the index:", 24, True, WHITE)],
             [("zero query-time transform, zero extra index memory.", 24, True, WHITE)]])
    c2.gap(0.16)
    c2.text([[("The query side carries the complement, q′ = (q − (1−α) g) / "
               "‖q − (1−α) g‖. α = 0 is query-only mean shift: it leaves the IVF "
               "centroids and the quantized codes misaligned and can lower recall. "
               "α = 1 makes q′ = q and rewrites the centroid where it plays both of "
               "its roles — routing pivot and code boundary. The α-sweep confirms "
               "α = 1 is best or near-best in every setting, and per-vector "
               "renormalization keeps the unit-norm structure the index expects.",
               21, False, INK)]])

    c2.heading("7", "Why it works: the gap is concentrated")
    c2.figure("asset_fig3b.png", 1.0)
    c2.gap(0.10)
    c2.text([[("Correcting 5% of dimensions can already be enough.", 25, True, BLUE)]],
            after=0)
    c2.gap(0.14)
    c2.text([[("Restricting the shift to the top-P% of dimensions by |gᵢ| isolates the "
               "mechanism. On CLIP, whose gap energy is concentrated (top 10% carries "
               "86–92%), P = 5% already reaches peak recall. ImageBind's diffuse gap "
               "(≈72%) needs the full vector — exactly what the flip-risk analysis "
               "predicts. This is the evidence that PMC repairs a structured failure "
               "rather than perturbing the codes at random.", 21, False, INK)]])

    c2.heading("8", "Beyond one bit")
    c2.text([[("Multi-bit quantizers absorb some displacement through wider error "
               "margins, so BQ is where the wound is deepest — but the same "
               "correction still helps. R@100, Vanilla → PMC, both directions "
               "averaged.", 20, False, INK)]], size=20)
    c2.gap(0.12)
    MB_W = Emu(int((c2.w - Inches(2.9)) / 2))
    for t, xx, ww, al in (("Dataset · Enc.", c2.x, Inches(2.9), PP_ALIGN.LEFT),
                          ("IVFPQ", c2.x + Inches(2.9), MB_W, PP_ALIGN.RIGHT),
                          ("OPQ", c2.x + Inches(2.9) + MB_W, MB_W, PP_ALIGN.RIGHT)):
        cell(slide, [(t, 19, True, GRAY)], xx, c2.y, ww, Inches(0.46), al)
    c2.y += Inches(0.48)
    c2.rule(Pt(1.5), INK, pad=0.02)
    for name, a, b, ad, c_, d, dd in (
            ("MSCOCO CLIP", ".52", ".64", "+23%", ".64", ".67", "+4%"),
            ("MSCOCO CL-L", ".39", ".61", "+56%", ".57", ".67", "+17%"),
            ("MSCOCO IB", ".59", ".69", "+17%", ".70", ".77", "+9%"),
            ("Flickr30K CL-L", ".47", ".50", "+7%", ".51", ".52", "+1%")):
        rh = Inches(0.56)
        cell(slide, [(name, 20, False, INK)], c2.x, c2.y, Inches(2.9), rh)
        cell(slide, [(f"{a}→", 18, False, GRAY), (b, 21, True, BLUE),
                     (f" {ad}", 18, True, BLUE)],
             c2.x + Inches(2.9), c2.y, MB_W, rh, PP_ALIGN.RIGHT)
        cell(slide, [(f"{c_}→", 18, False, GRAY), (d, 21, True, BLUE),
                     (f" {dd}", 18, True, BLUE)],
             c2.x + Inches(2.9) + MB_W, c2.y, MB_W, rh, PP_ALIGN.RIGHT)
        c2.y += rh
        solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, c2.x, c2.y, c2.w, Pt(0.8)), HAIR)
    c2.gap(0.18)
    c2.text([[("OPQ's learned rotation partly absorbs the shift, giving smaller but "
               "complementary gains; IVFPQ lacks this and benefits more. Audio is "
               "omitted here — its database is too small to train a codebook.",
               20, False, INK)]], size=20)

    c2.heading("9", "Experimental setup")
    c2.text([[("Five benchmarks — MSCOCO 5K, Flickr30K (31K), Clotho v2 (5,926), "
               "AudioCaps (884 clips / 4,415 captions) and LAION-400M (407M vectors) — "
               "three frozen encoders (CLIP-B/32, CLIP-L/14, ImageBind) and four BQ "
               "index types (BinaryFlat, BinaryIVF, BBQ-style rotated, RaBitQ). FAISS "
               "IVFRaBitQFastScan at 72–136 B/vec; n_list = ⌈√n⌉ and n_probe ≈ "
               "n_list/4, matched across methods. R@100 against exact inner-product "
               "ground truth on the original embeddings; single-thread CPU.",
               20, False, INK)]], size=20)

    # ------------------------------ column 3 --------------------------
    c3.heading("10", "Results")
    c3.gap(0.02)
    c3.text([[("407 M", 58, True, CRIMSON), ("  vectors", 24, False, GRAY)]],
            spacing=0.95, after=0)
    c3.gap(0.04)
    c3.text([[("LAION-400M  ·  29.3 GB of codes at 72 B/vec  ·  28× compressed",
               20, True, GRAY)]], after=0)
    c3.gap(0.20)
    c3.text([[("R@100, no reranking (n_list = 80K, n_probe = 256)", 20, True, INK)]],
            after=0)
    c3.gap(0.14)
    c3.hbar(0.108 / 0.143, GRAY, "Vanilla", ".108")
    c3.hbar(1.0, BLUE, "PMC   (+32%)", ".143")
    c3.gap(0.02)
    c3.text([[("With exact reranking at K′ = 400 the lead carries over: ", 20, False, INK),
              (".198 → .277  (+40%)", 20, True, BLUE)]], size=20, after=0)
    c3.gap(0.22)

    c3.text([[("Every configuration, ordered by modality gap", 23, True, INK)]], after=0)
    c3.gap(0.16)
    COL_G, COL_N = Inches(0.76), Inches(2.45)
    COL_D = Emu(int((c3.w - COL_G - COL_N) / 2))
    for t, xx, ww, al in (("‖g‖", c3.x, COL_G, PP_ALIGN.LEFT),
                          ("Dataset · Enc.", c3.x + COL_G, COL_N, PP_ALIGN.LEFT),
                          ("q→db", c3.x + COL_G + COL_N, COL_D, PP_ALIGN.RIGHT),
                          ("db→q", c3.x + COL_G + COL_N + COL_D, COL_D, PP_ALIGN.RIGHT)):
        cell(slide, [(t, 19, True, GRAY)], xx, c3.y, ww, Inches(0.46), al)
    c3.y += Inches(0.48)
    c3.rule(Pt(1.5), INK, pad=0.02)
    for g, name, qv, qp, qd, dv, dp, dd in [
            (".82", "MSCOCO CL-L", ".55", ".65", "+18%", ".47", ".63", "+34%"),
            (".82", "MSCOCO CLIP", ".58", ".63", "+9%", ".50", ".60", "+20%"),
            (".77", "Flickr30K CL-L", ".41", ".48", "+17%", ".33", ".48", "+45%"),
            (".72", "LAION-400M CLIP", ".108", ".143", "+32%", ".069", ".073", "+6%"),
            (".70", "MSCOCO IB", ".67", ".75", "+12%", ".71", ".75", "+6%"),
            (".61", "Clotho IB", ".72", ".73", "+1%", ".62", ".69", "+11%"),
            (".61", "AudioCaps IB", ".75", ".78", "+4%", ".83", ".83", "+0%")]:
        rh = Inches(0.58)
        cell(slide, [(g, 21, True, CRIMSON)], c3.x, c3.y, COL_G, rh)
        cell(slide, [(name, 20, False, INK)], c3.x + COL_G, c3.y, COL_N, rh)
        cell(slide, [(f"{qv}→", 18, False, GRAY), (qp, 21, True, BLUE),
                     (f" {qd}", 18, True, BLUE)],
             c3.x + COL_G + COL_N, c3.y, COL_D, rh, PP_ALIGN.RIGHT)
        cell(slide, [(f"{dv}→", 18, False, GRAY), (dp, 21, True, BLUE),
                     (f" {dd}", 18, True, BLUE)],
             c3.x + COL_G + COL_N + COL_D, c3.y, COL_D, rh, PP_ALIGN.RIGHT)
        c3.y += rh
        solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, c3.x, c3.y, c3.w, Pt(0.8)), HAIR)
    c3.gap(0.20)
    c3.text([[("The gain tracks the gap. ", 21, True, CRIMSON),
              ("R@100 improves or matches in all 16 BQ configurations and PMC is best "
               "in all 14 at R@10, so the benefit is not confined to deep result "
               "lists. The rows with the smallest gap gain least, exactly as the "
               "flip-risk analysis predicts.", 21, False, INK)]])

    c3.heading("11", "Robustness under exact reranking")
    c3.text([[("Exact rescoring can only recover what the first stage retrieved, so "
               "Vanilla stays capped by what its corrupted codes admit. On MSCOCO "
               "t→i, PMC reaches .62/.80/.90/.93 at K′ = 100/200/400/500 against "
               "Vanilla's .53/.69/.81/.85 — R@100 ≥ 0.90 at a smaller budget.",
               20, False, INK)]], size=20)
    c3.gap(0.06)
    c3.card([[("One regime favours uncorrected codes.", 20, True, CRIMSON)],
             [("At 400M scale in the reverse direction, rescoring lets Vanilla "
               "overtake PMC for K′ ≥ 200: LAION probes only 0.32% of its lists, so "
               "PMC's flatter candidate pool covers fewer true positives — a "
               "scan-depth effect, not a limit of the correction.",
               19, False, INK)]],
            size=19, pad=0.24)

    c3.heading("12", "Where to correct")
    c3.text([[("MSCOCO · CLIP-B/32 · IVF-RaBitQ · R@100 (t→i)", 19, False, GRAY)]],
            after=0)
    c3.gap(0.14)
    for label, v, col, val in (("Vanilla", 0.578, GRAY, ".578"),
                               ("Query-only", 0.541, GRAY, ".541"),
                               ("Both sides", 0.599, GRAY, ".599"),
                               ("DB-only  (PMC)", 0.637, BLUE, ".637")):
        c3.hbar((v - 0.45) / (0.637 - 0.45), col, label, val)
    c3.gap(0.04)
    c3.text([[("Shifting both sides re-displaces the routing pivot. ", 21, False, INK),
              ("Correct the centroid where it controls both routing and code "
               "formation.", 21, True, CRIMSON)]])
    c3.gap(0.02)
    c3.text([[("Direction controls (t→i): a random direction of the same norm gives "
               ".562, a shuffled gap .552, a sign-flipped gap .514, an un-normalized "
               "shift .555 — all below Vanilla's .578, far below PMC's .637. The "
               "direction of g is what matters, not perturbing the codes.",
               19, False, GRAY)]], size=19, spacing=1.20)

    c3.heading("13", "Throughput and conclusion")
    c3.figure("asset_fig3c.png", 0.90)
    c3.gap(0.06)
    c3.text([[("QPS tracks Vanilla at every n_probe — PMC adds no per-query work.",
               18, False, GRAY, True)]], size=18, spacing=1.15, after=0)
    c3.gap(0.16)
    c3.card([[("A one-time, database-side centroid correction repairs IVF routing and "
               "binary quantization together — without changing the serving path.",
               23, True, INK)],
             [("↑ Recall, scaling with ‖g‖   ·   0 query-time transform   ·   0 extra "
               "index memory   ·   validated at 407M-vector scale", 20, False, BLUE)],
             [("Drift-adaptive α, graph-based ANN and billion-scale deployment remain "
               "future work.", 19, False, GRAY)]], pad=0.24)

    # ============================= footer =============================
    fy = PAGE_H - FOOTER_H
    solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, fy, PAGE_W, FOOTER_H), CARD)
    solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, fy, PAGE_W, Pt(3.0)), CRIMSON)
    LOGO_H = Inches(1.30)
    LOGO_W = Emu(int(LOGO_H * ASPECT["asset_cikm.png"]))
    slide.shapes.add_picture(str(HERE / "asset_cikm.png"), M, fy + Inches(0.36),
                             LOGO_W, LOGO_H)
    ft = Col(slide, M + LOGO_W + Inches(0.65), CW - LOGO_W - Inches(0.65),
             fy + Inches(0.50), PAGE_H)
    ft.text([[("The 35th ACM International Conference on Information and Knowledge "
               "Management  ·  November 07–11, 2026  ·  Rome, Italy", 21, True, INK)]],
            after=2)
    ft.text([[("DOI 10.1145/3799682.3840007   ·   CC-BY 4.0   ·   Supported by the "
               "National Research Foundation of Korea (MSIT), No. 00359638   ·   "
               "Code and reproduction package: github.com/sehoon787/PMC",
               18, False, GRAY)]], size=18, after=0)

    for n, c in (("1", c1), ("2", c2), ("3", c3)):
        used, room = (c.y - TOP) / 914400.0, (BOT - TOP) / 914400.0
        print(f"  col {n}: {used:5.2f} / {room:5.2f} in  ({used / room * 100:4.1f}%)"
              f"  {'OK' if c.y <= BOT else 'OVERFLOW'}")

    out = HERE / "PMC_CIKM2026_poster.pptx"
    prs.save(str(out))
    print(f"saved {out}")


if __name__ == "__main__":
    build()
