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
    "asset_fig2ov.png": 1537 / 1024,
    "asset_fig3b.png": 1188 / 929,
    "asset_fig3c.png": 1206 / 929,
    "asset_gapgain.png": 989 / 657,
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
        self.gap(0.16)
        self.text([[(f"{num}.  {title}", 35, True, CRIMSON)]], spacing=1.05, after=0)
        self.rule(Pt(2.5), CRIMSON, pad=0.07)

    def bullets(self, items, size=21, marker=CRIMSON, gap_in=0.09):
        for it in items:
            solid(self.s.shapes.add_shape(MSO_SHAPE.RECTANGLE, self.x + Inches(0.03),
                                          self.y + Inches(0.15), Inches(0.18), Inches(0.18)),
                  marker)
            self.text([it], size=size, x=self.x + Inches(0.44),
                      w=self.w - Inches(0.44), spacing=1.20, after=0)
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

    def hbar(self, frac, color, label, value, h_in=0.68, val_w=1.70):
        h = Inches(h_in)
        lane_w = self.w - Inches(val_w)
        solid(self.s.shapes.add_shape(MSO_SHAPE.RECTANGLE, self.x, self.y, lane_w, h),
              WHITE, HAIR, Pt(1.0))
        solid(self.s.shapes.add_shape(MSO_SHAPE.RECTANGLE, self.x, self.y,
                                      Emu(int(lane_w * frac)), h), color)
        cell(self.s, [(label, 24, False, WHITE if frac > 0.45 else INK)],
             self.x + Inches(0.18), self.y, lane_w - Inches(0.3), h)
        cell(self.s, [(value, 24, True, INK)],
             self.x + lane_w + Inches(0.14), self.y, Inches(val_w - 0.14), h)
        self.y += h + Inches(0.10)


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
    head.gap(0.06)
    head.text([[("{sehoon787, junicus, jsy}@korea.ac.kr", 20, False, GRAY, True)]], after=0)
    solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, M, HEADER_H, CW, Pt(3.0)), CRIMSON)

    # ========================= TL;DR band =============================
    ab = Col(slide, M, CW, HEADER_H + Inches(0.42), PAGE_H)
    ab.band([[("The modality gap breaks binary-quantized indexes.  PMC repairs it "
               "at build time — for free.", 32, True, WHITE)],
             [("Shift each database vector toward the query centroid before "
               "indexing: zero query-time transform, zero extra memory, recall "
               "gains that scale with the gap — validated at 407M vectors.",
               25, False, WHITE)]], pad=0.34)

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
    c1.bullets([
        [("One shared space (CLIP, ImageBind) — but 407M float vectors = ",
          26, False, INK), ("833 GB.", 26, True, INK)],
        [("Binary quantization:", 26, True, INK),
         (" 1 bit per dimension, 32× smaller (RaBitQ, BBQ).", 26, False, INK)],
        [("Every BQ method encodes ", 26, False, INK), ("sign(x − c)", 26, True, BLUE),
         (" — the centroid c also routes IVF queries.", 26, False, INK)],
    ], size=26, gap_in=0.22)

    c1.heading("2", "The problem: the modality gap")
    c1.text([[("Queries and database vectors do ", 26, False, INK),
              ("not", 26, True, CRIMSON),
              (" share that centroid:", 26, False, INK)]], size=26)
    c1.gap(0.12)
    c1.text([[("g  =  μ_{q} − μ_{x}", 36, True, INK)]],
            align=PP_ALIGN.CENTER, after=0)
    c1.text([[("stable and large:  ‖g‖ = .61–.82", 21, False, GRAY, True)]],
            size=21, align=PP_ALIGN.CENTER, after=0)
    c1.gap(0.20)
    c1.figure("asset_fig1.png", 1.0)
    c1.gap(0.10)
    c1.text([[("Each modality forms its own cluster; after PMC the pairs overlap.",
               21, False, GRAY, True)]], size=21, spacing=1.15, after=0)
    c1.gap(0.30)
    c1.text([[("One gap, two failures", 30, True, CRIMSON)]], after=0)
    c1.gap(0.16)
    c1.bullets([
        [("IVF routing", 26, True, INK),
         ("  —  queries land in the wrong lists.", 26, False, INK)],
        [("Sign bits", 26, True, INK),
         ("  —  zero-margin boundary at c; the offset flips nearby bits:",
          26, False, INK)],
    ], size=26, gap_in=0.16)
    c1.gap(0.12)
    c1.text([[("E[F]  ≈  α Σᵢ pᵢ(0) · |gᵢ|", 34, True, INK)]],
            align=PP_ALIGN.CENTER, after=0)
    c1.text([[("expected bit flips — top 10% of dims carry ≈90% of ‖g‖²",
               21, False, GRAY, True)]], size=21,
            align=PP_ALIGN.CENTER, spacing=1.15, after=0)
    c1.gap(0.26)
    c1.card([[("Oracle test — MSCOCO, CLIP-L, RaBitQ", 24, True, INK)],
             [("same-modality  (‖g‖ = 0)         R@100 = 0.71", 25, False, INK)],
             [("cross-modal  (‖g‖ = .82)         R@100 = 0.54", 25, True, CRIMSON)],
             [("The loss is the gap, not the quantizer.", 21, False, GRAY, True)]],
            pad=0.30)

    c1.heading("3", "Every binary index has the same wound")
    c1.text([[("R@100, Vanilla → PMC", 21, False, GRAY)]], size=21, after=0)
    c1.gap(0.14)
    T1_W = Emu(int((c1.w - Inches(2.9)) / 2))
    for t, xx, ww, al in (("Method", c1.x, Inches(2.9), PP_ALIGN.LEFT),
                          ("MSCOCO t→i", c1.x + Inches(2.9), T1_W, PP_ALIGN.RIGHT),
                          ("AudioCaps a→t", c1.x + Inches(2.9) + T1_W, T1_W,
                           PP_ALIGN.RIGHT)):
        cell(slide, [(t, 22, True, GRAY)], xx, c1.y, ww, Inches(0.52), al)
    c1.y += Inches(0.54)
    c1.rule(Pt(1.5), INK, pad=0.02)
    for m, a, b, c_, d in (("BinaryFlat", ".51", ".57", ".51", ".64"),
                           ("BinaryIVF", ".51", ".57", ".50", ".63"),
                           ("RotatedBinary", ".29", ".58", ".63", ".69"),
                           ("RaBitQ", ".58", ".64", ".67", ".75")):
        rh = Inches(0.66)
        cell(slide, [(m, 24, False, INK)], c1.x, c1.y, Inches(2.9), rh)
        cell(slide, [(f"{a} → ", 22, False, GRAY), (b, 25, True, BLUE)],
             c1.x + Inches(2.9), c1.y, T1_W, rh, PP_ALIGN.RIGHT)
        cell(slide, [(f"{c_} → ", 22, False, GRAY), (d, 25, True, BLUE)],
             c1.x + Inches(2.9) + T1_W, c1.y, T1_W, rh, PP_ALIGN.RIGHT)
        c1.y += rh
        solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, c1.x, c1.y, c1.w, Pt(0.8)), HAIR)
    c1.gap(0.20)
    c1.text([[("Improves or matches in all 16 configs. ", 26, True, INK),
              ("The wound is the centroid, not the codec.", 26, False, INK)]],
            size=26)

    c1.heading("4", "Contributions")
    c1.bullets([
        [("The gap corrupts routing and codes ", 26, False, INK),
         ("together", 26, True, INK), ("; loss scales with ‖g‖.", 26, False, INK)],
        [("A few high-|gᵢ| dimensions carry most of the damage.", 26, False, INK)],
        [("PMC:", 26, True, INK),
         (" zero-cost, build-time, database-side correction.", 26, False, INK)],
    ], marker=BLUE, size=26, gap_in=0.22)
    c1.gap(0.30)
    c1.card([[("Setup — ", 21, True, INK),
              ("5 benchmarks (MSCOCO, Flickr30K, Clotho, AudioCaps, LAION-400M) · "
               "3 frozen encoders (CLIP-B/32, CLIP-L/14, ImageBind) · 4 BQ index "
               "types · FAISS, 72–136 B/vec · n_list = ⌈√n⌉ · R@100 vs exact ground "
               "truth · single-thread CPU.", 21, False, GRAY)]], size=21, pad=0.26)

    # ------------------------------ column 2 --------------------------
    c2.heading("5", "PMC — correct at the source")
    c2.figure("asset_fig2ov.png", 1.0)
    c2.gap(0.10)
    c2.text([[("Calibrate g from ~25 paired samples · shift the database offline · "
               "serve with the query untouched.", 21, False, GRAY, True)]],
            size=21, spacing=1.15, after=0)
    c2.gap(0.26)
    c2.card([[("Build  (offline)", 22, True, GRAY)],
             [("x′  =  (x + α g)  /  ‖x + α g‖", 32, True, INK)],
             [("", 10, False, INK)],
             [("Serve  (online)", 22, True, GRAY)],
             [("q′  =  (q − (1−α) g)  /  ‖q − (1−α) g‖", 32, True, INK)],
             [("→   q′ = q   at α = 1", 27, True, BLUE)]], pad=0.32)
    c2.gap(0.18)
    c2.band([[("At α = 1 the correction is absorbed into the index:", 26, True, WHITE)],
             [("zero query-time transform, zero extra index memory.", 26, True, WHITE)]])
    c2.gap(0.20)
    c2.bullets([
        [("α = 0", 26, True, INK),
         (" (query-only mean shift) leaves codes misaligned — recall can drop.",
          26, False, INK)],
        [("α = 1", 26, True, INK),
         (" rewrites the centroid in both its roles; best in the α-sweep.",
          26, False, INK)],
    ], size=26, gap_in=0.20)

    c2.heading("6", "Why it works: the gap is concentrated")
    c2.figure("asset_fig3b.png", 0.94)
    c2.gap(0.14)
    c2.text([[("Correcting 5% of dimensions can already be enough.", 28, True, BLUE)]],
            after=0)
    c2.gap(0.12)
    c2.text([[("CLIP's concentrated gap peaks at P = 5%; ImageBind's diffuse gap "
               "needs all dims — just as E[F] predicts.", 26, False, INK)]], size=26)

    c2.heading("7", "Beyond one bit")
    c2.text([[("R@100, Vanilla → PMC, both directions averaged", 21, False, GRAY)]],
            size=21, after=0)
    c2.gap(0.14)
    MB_W = Emu(int((c2.w - Inches(3.4)) / 2))
    for t, xx, ww, al in (("Dataset · Enc.", c2.x, Inches(3.4), PP_ALIGN.LEFT),
                          ("IVFPQ", c2.x + Inches(3.4), MB_W, PP_ALIGN.RIGHT),
                          ("OPQ", c2.x + Inches(3.4) + MB_W, MB_W, PP_ALIGN.RIGHT)):
        cell(slide, [(t, 22, True, GRAY)], xx, c2.y, ww, Inches(0.52), al)
    c2.y += Inches(0.54)
    c2.rule(Pt(1.5), INK, pad=0.02)
    for name, a, b, ad, c_, d, dd in (
            ("MSCOCO CLIP", ".52", ".64", "+23%", ".64", ".67", "+4%"),
            ("MSCOCO CL-L", ".39", ".61", "+56%", ".57", ".67", "+17%"),
            ("MSCOCO IB", ".59", ".69", "+17%", ".70", ".77", "+9%"),
            ("Flickr30K CL-L", ".47", ".50", "+7%", ".51", ".52", "+1%")):
        rh = Inches(0.66)
        cell(slide, [(name, 24, False, INK)], c2.x, c2.y, Inches(3.4), rh)
        cell(slide, [(f"{a}→", 21, False, GRAY), (b, 25, True, BLUE),
                     (f" {ad}", 21, True, BLUE)],
             c2.x + Inches(3.4), c2.y, MB_W, rh, PP_ALIGN.RIGHT)
        cell(slide, [(f"{c_}→", 21, False, GRAY), (d, 25, True, BLUE),
                     (f" {dd}", 21, True, BLUE)],
             c2.x + Inches(3.4) + MB_W, c2.y, MB_W, rh, PP_ALIGN.RIGHT)
        c2.y += rh
        solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, c2.x, c2.y, c2.w, Pt(0.8)), HAIR)
    c2.gap(0.20)
    c2.text([[("Multi-bit absorbs part of the shift — BQ is the deepest wound, but "
               "PMC still helps.", 26, False, INK)]], size=26)

    # ------------------------------ column 3 --------------------------
    c3.heading("8", "Results")
    c3.gap(0.02)
    c3.text([[("407 M", 64, True, CRIMSON), ("  vectors", 26, False, GRAY)]],
            spacing=0.95, after=0)
    c3.gap(0.06)
    c3.text([[("LAION-400M · 29.3 GB of codes · 28× compressed", 22, True, GRAY)]],
            after=0)
    c3.gap(0.22)
    c3.hbar(0.108 / 0.143, GRAY, "Vanilla", ".108")
    c3.hbar(1.0, BLUE, "PMC   (+32%)", ".143")
    c3.text([[("R@100, no reranking · with exact reranking: ", 21, False, GRAY),
              (".198 → .277 (+40%)", 21, True, BLUE)]], size=21, after=0)
    c3.gap(0.30)

    c3.text([[("Every configuration, ordered by ‖g‖", 27, True, INK)]], after=0)
    c3.gap(0.18)
    COL_G, COL_N = Inches(0.90), Inches(2.95)
    COL_D = Emu(int((c3.w - COL_G - COL_N) / 2))
    for t, xx, ww, al in (("‖g‖", c3.x, COL_G, PP_ALIGN.LEFT),
                          ("Dataset · Enc.", c3.x + COL_G, COL_N, PP_ALIGN.LEFT),
                          ("q→db", c3.x + COL_G + COL_N, COL_D, PP_ALIGN.RIGHT),
                          ("db→q", c3.x + COL_G + COL_N + COL_D, COL_D, PP_ALIGN.RIGHT)):
        cell(slide, [(t, 22, True, GRAY)], xx, c3.y, ww, Inches(0.52), al)
    c3.y += Inches(0.54)
    c3.rule(Pt(1.5), INK, pad=0.02)
    for g, name, qv, qp, qd, dv, dp, dd in [
            (".82", "MSCOCO CL-L", ".55", ".65", "+18%", ".47", ".63", "+34%"),
            (".82", "MSCOCO CLIP", ".58", ".63", "+9%", ".50", ".60", "+20%"),
            (".77", "Flickr30K CL-L", ".41", ".48", "+17%", ".33", ".48", "+45%"),
            (".72", "LAION-400M CLIP", ".108", ".143", "+32%", ".069", ".073", "+6%"),
            (".70", "MSCOCO IB", ".67", ".75", "+12%", ".71", ".75", "+6%"),
            (".61", "Clotho IB", ".72", ".73", "+1%", ".62", ".69", "+11%"),
            (".61", "AudioCaps IB", ".75", ".78", "+4%", ".83", ".83", "+0%")]:
        rh = Inches(0.66)
        cell(slide, [(g, 24, True, CRIMSON)], c3.x, c3.y, COL_G, rh)
        cell(slide, [(name, 23, False, INK)], c3.x + COL_G, c3.y, COL_N, rh)
        cell(slide, [(f"{qv}→", 20, False, GRAY), (qp, 24, True, BLUE),
                     (f" {qd}", 20, True, BLUE)],
             c3.x + COL_G + COL_N, c3.y, COL_D, rh, PP_ALIGN.RIGHT)
        cell(slide, [(f"{dv}→", 20, False, GRAY), (dp, 24, True, BLUE),
                     (f" {dd}", 20, True, BLUE)],
             c3.x + COL_G + COL_N + COL_D, c3.y, COL_D, rh, PP_ALIGN.RIGHT)
        c3.y += rh
        solid(slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, c3.x, c3.y, c3.w, Pt(0.8)), HAIR)
    c3.gap(0.28)
    c3.figure("asset_gapgain.png", 0.60)
    c3.gap(0.12)
    c3.text([[("The gain tracks the gap ", 27, True, CRIMSON),
              ("— smallest gaps gain least, as E[F] predicts.", 27, False, INK)]],
            size=27)

    c3.heading("9", "Where to correct")
    c3.text([[("MSCOCO · CLIP-B/32 · IVF-RaBitQ · R@100 (t→i)", 21, False, GRAY)]],
            size=21, after=0)
    c3.gap(0.16)
    for label, v, col, val in (("Vanilla", 0.578, GRAY, ".578"),
                               ("Query-only", 0.541, GRAY, ".541"),
                               ("Both sides", 0.599, GRAY, ".599"),
                               ("DB-only  (PMC)", 0.637, BLUE, ".637")):
        c3.hbar((v - 0.45) / (0.637 - 0.45), col, label, val)
    c3.gap(0.08)
    c3.text([[("Correct the centroid where it controls both routing and code "
               "formation.", 26, True, CRIMSON)]], size=26)

    c3.heading("10", "Throughput and conclusion")
    c3.figure("asset_fig3c.png", 0.74)
    c3.gap(0.08)
    c3.text([[("QPS tracks Vanilla at every n_probe — no per-query work.",
               21, False, GRAY, True)]], size=21, spacing=1.15, after=0)
    c3.gap(0.20)
    c3.card([[("One build-time correction repairs IVF routing and binary "
               "quantization together.", 26, True, INK)],
             [("↑ Recall scaling with ‖g‖  ·  0 query-time cost  ·  0 extra memory "
               " ·  407M-vector scale", 22, False, BLUE)]], pad=0.30)

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
