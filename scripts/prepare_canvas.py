#!/usr/bin/env python3
"""Prepare the 1280x720 whiteboard canvas for the image-only pipeline.

Detects the actual content bounding box of the question image (trims white
margins), selects the best layout mode, then composes a white board with the
question image on the appropriate zone.  The composed canvas is the render
background — every downstream coordinate is canvas-native.

Layout modes:
  two_column    — question left (40–560 px), solution right (610–1200 px)
  top_bottom    — question top (40–720 wide, 50–340 px), solution below
  question_first — question fills the left 60% (wide images); solution is right 40%

Output:
    output/canvas.png   — composed board (render background)
    output/layout.json  — zone metadata
"""

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

CANVAS_W, CANVAS_H = 1280, 720
SEP_COLOR = (205, 208, 215)

# Zone templates (x1, y1, x2, y2)
_ZONES = {
    "two_column": {
        "question_zone": (40,  50,  560, 650),
        "solution_zone": (610, 70, 1200, 650),
    },
    "top_bottom": {
        "question_zone": (40,  40,  1240, 340),
        "solution_zone": (40, 360, 1240, 680),
    },
    "question_first": {
        "question_zone": (30,  50,  720, 660),
        "solution_zone": (740, 60, 1240, 660),
    },
}

DEFAULT_MODE = "two_column"


def _content_bbox(img: Image.Image, threshold: int = 245) -> tuple:
    """Return (x1, y1, x2, y2) of non-white content in the image.

    Anything brighter than `threshold` on all channels is treated as margin."""
    arr = np.array(img.convert("RGB"))
    mask = np.any(arr < threshold, axis=2)          # True where there is content
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return (0, 0, img.width, img.height)        # no white margin found
    pad = 4                                          # tiny padding so content isn't clipped
    return (max(0, int(cols[0]) - pad), max(0, int(rows[0]) - pad),
            min(img.width, int(cols[-1]) + pad), min(img.height, int(rows[-1]) + pad))


def _choose_layout(content_w: int, content_h: int) -> str:
    """Pick the best layout mode for the cropped question content."""
    aspect = content_w / max(content_h, 1)
    # Wide question (landscape-ish) → top-bottom splits nicely
    if aspect > 2.0:
        return "top_bottom"
    # Very tall question that would be tiny in a 520px column → question_first
    if content_h > 580 and aspect < 0.9:
        return "question_first"
    return "two_column"


def _readability_status(scale: float) -> str:
    if scale >= 0.9:
        return "good"
    if scale >= 0.7:
        return "acceptable"
    return "poor"


def prepare_canvas(image_path: str,
                   canvas_out: str = "output/canvas.png",
                   layout_out: str = "output/layout.json",
                   layout_mode: str | None = None) -> dict:
    """Compose the whiteboard and write layout.json. Returns the layout dict."""
    img = Image.open(image_path).convert("RGB")

    # 1. Detect and crop white margins
    cx1, cy1, cx2, cy2 = _content_bbox(img)
    content_w, content_h = cx2 - cx1, cy2 - cy1
    cropped = img.crop((cx1, cy1, cx2, cy2))

    # 2. Choose layout
    mode = layout_mode or _choose_layout(content_w, content_h)
    zones = _ZONES.get(mode, _ZONES[DEFAULT_MODE])
    qx1, qy1, qx2, qy2 = zones["question_zone"]
    max_w, max_h = qx2 - qx1, qy2 - qy1

    # 3. Fit the cropped content into the question zone (never scale above 1.6×)
    scale = min(max_w / max(content_w, 1), max_h / max(content_h, 1), 1.6)
    new_w = max(1, int(content_w * scale))
    new_h = max(1, int(content_h * scale))
    fitted = cropped.resize((new_w, new_h), Image.LANCZOS)

    # 4. Center horizontally, top-align vertically inside question zone
    px = qx1 + max(0, (max_w - new_w) // 2)
    py = qy1

    # 5. Compose
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (255, 255, 255))
    canvas.paste(fitted, (px, py))

    # 6. Separator line
    draw = ImageDraw.Draw(canvas)
    if mode == "two_column":
        sep_x = (qx2 + zones["solution_zone"][0]) // 2
        draw.line([(sep_x, 30), (sep_x, CANVAS_H - 30)],
                  fill=SEP_COLOR, width=2)
    elif mode in ("top_bottom",):
        sep_y = (qy2 + zones["solution_zone"][1]) // 2
        draw.line([(30, sep_y), (CANVAS_W - 30, sep_y)],
                  fill=SEP_COLOR, width=2)
    elif mode == "question_first":
        sep_x = (qx2 + zones["solution_zone"][0]) // 2
        draw.line([(sep_x, 30), (sep_x, CANVAS_H - 30)],
                  fill=SEP_COLOR, width=2)

    os.makedirs(os.path.dirname(canvas_out) or ".", exist_ok=True)
    canvas.save(canvas_out)

    layout = {
        "canvas": {"width": CANVAS_W, "height": CANVAS_H, "background": "white"},
        "layout_mode": mode,
        "question_zone": list(zones["question_zone"]),
        "solution_zone": list(zones["solution_zone"]),
        "question_image_box": [px, py, px + new_w, py + new_h],
        "content_scale": round(scale, 3),
        "readability_status": _readability_status(scale),
        "canvas_image": canvas_out,
        "source_image": image_path,
        "source_size": [img.width, img.height],
        "content_crop": [cx1, cy1, cx2, cy2],
        "scale": round(scale, 3),
    }
    with open(layout_out, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2, ensure_ascii=False)
    print(f"  Canvas {CANVAS_W}x{CANVAS_H} [{mode}] -> {canvas_out}")
    print(f"  Question pasted at {layout['question_image_box']} "
          f"(scale={scale:.2f}, readability={layout['readability_status']})")
    print(f"  Layout metadata -> {layout_out}")
    return layout


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "input/question.png"
    mode = sys.argv[2] if len(sys.argv) > 2 else None
    prepare_canvas(image, layout_mode=mode)
