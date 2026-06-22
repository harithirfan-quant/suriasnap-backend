"""
Render a representative rooftop PV-array design image (Arka-360-style satellite
layout) with Pillow — no SVG renderer, no new dependency.

The same PNG is used two ways:
  - sent as a WhatsApp image message (app/conversations/orchestrator.py)
  - embedded into the PDF report (app/services/report_generator.py)

It is a SCHEMATIC, representative layout (clearly labelled), driven by the real
panel count / orientation — not a survey-grade engineering drawing.
"""

from __future__ import annotations

import io
import math

from PIL import Image, ImageDraw, ImageFont

# 2× the 480×300 web mock for crisp output; +banner room baked into H
W, H = 960, 600
BANNER_H = 66

# satellite-scene palette (hardcoded — this is a physical scene, not theme UI)
SAT_TOP    = (27, 42, 58)
SAT_BOT    = (36, 59, 79)
PLOT       = (46, 69, 89)
ROOF       = (51, 64, 77)
ROOF_EDGE  = (90, 107, 122)
PANEL      = (38, 86, 138)
PANEL_LINE = (63, 124, 184)
PANEL_EDGE = (14, 36, 56)
PROP       = (124, 255, 216)
SUN        = (253, 184, 19)
BANNER_BG  = (15, 26, 38)
INK        = (226, 236, 243)
MUTE       = (150, 170, 185)

try:
    _RESAMPLE = Image.Resampling.BICUBIC
except AttributeError:                       # Pillow < 9.1
    _RESAMPLE = Image.BICUBIC


def _font(size: int, bold: bool = False) -> ImageFont.FontFont:
    suffix = "-Bold" if bold else ""
    for path in (
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf",
        f"DejaVuSans{suffix}.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)   # Pillow ≥ 10
    except TypeError:
        return ImageFont.load_default()


def render_design_png(num_panels: int, orientation: str, system_kwp: float,
                      specific_yield: int) -> bytes:
    num_panels = max(1, int(num_panels))
    scene_h = H - BANNER_H

    img = Image.new("RGB", (W, H), SAT_BOT)
    draw = ImageDraw.Draw(img)

    # vertical gradient ground (horizontal lines = fast)
    for y in range(scene_h):
        t = y / scene_h
        draw.line(
            [(0, y), (W, y)],
            fill=tuple(int(a + (b - a) * t) for a, b in zip(SAT_TOP, SAT_BOT)),
        )

    # faint neighbouring plots
    for x0, y0, w, h in [(40, 60, 180, 120), (760, 76, 156, 140),
                         (56, 360, 220, 110), (720, 360, 184, 96)]:
        draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=8, fill=PLOT)

    # roof + panel array on a slightly-rotated transparent layer
    cols = min(7, num_panels)
    rows = math.ceil(num_panels / cols)
    pw, ph, gap = 54, 32, 5
    arr_w = cols * pw + (cols - 1) * gap
    arr_h = rows * ph + (rows - 1) * gap
    ox = (W - arr_w) / 2
    oy = (scene_h - arr_h) / 2

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    ld.rounded_rectangle([ox - 88, oy - 80, ox + arr_w + 88, oy + arr_h + 80],
                         radius=16, outline=PROP + (110,), width=2)
    ld.rounded_rectangle([ox - 60, oy - 52, ox + arr_w + 60, oy + arr_h + 52],
                         radius=12, fill=ROOF + (255,), outline=ROOF_EDGE + (255,), width=3)
    for i in range(num_panels):
        x = ox + (i % cols) * (pw + gap)
        y = oy + (i // cols) * (ph + gap)
        ld.rounded_rectangle([x, y, x + pw, y + ph], radius=3,
                             fill=PANEL + (255,), outline=PANEL_EDGE + (255,), width=1)
        ld.line([(x + pw / 3, y), (x + pw / 3, y + ph)], fill=PANEL_LINE + (255,), width=1)
        ld.line([(x + 2 * pw / 3, y), (x + 2 * pw / 3, y + ph)], fill=PANEL_LINE + (255,), width=1)
        ld.line([(x, y + ph / 2), (x + pw, y + ph / 2)], fill=PANEL_LINE + (255,), width=1)
    layer = layer.rotate(-7, resample=_RESAMPLE, center=(W / 2, scene_h / 2))
    img.paste(layer, (0, 0), layer)

    # sun (top-right) + north arrow (top-left)
    sx, sy = W - 76, 72
    draw.ellipse([sx - 22, sy - 22, sx + 22, sy + 22], fill=SUN)
    for i in range(8):
        a = i * math.pi / 4
        draw.line([(sx + math.cos(a) * 28, sy + math.sin(a) * 28),
                   (sx + math.cos(a) * 38, sy + math.sin(a) * 38)], fill=SUN, width=3)
    nx, ny = 64, 70
    draw.polygon([(nx, ny - 24), (nx + 10, ny + 8), (nx, ny), (nx - 10, ny + 8)], fill=PROP)
    draw.text((nx, ny + 22), "N", fill=PROP, font=_font(20, bold=True), anchor="mm")

    # bottom banner: title + representative tag + specs
    draw.rectangle([0, H - BANNER_H, W, H], fill=BANNER_BG)
    draw.text((24, H - BANNER_H + 13), "Professional Design Preview",
              fill=INK, font=_font(23, bold=True))
    draw.text((24, H - BANNER_H + 43), "Representative layout · powered by Arka 360",
              fill=MUTE, font=_font(13))
    specs = (f"{num_panels} x 400W   ·   {system_kwp} kWp   ·   "
             f"{orientation}-facing   ·   ~{specific_yield:,} kWh/kWp/yr")
    draw.text((W - 24, H - BANNER_H // 2), specs, fill=INK, font=_font(16), anchor="rm")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
