#!/usr/bin/env python3
"""
Render the final annotated video with teacher-like actions.

NEW APPROACH:
  - Render DIRECTLY on the question image (no separate workspace below).
  - Support semantic teacher actions: underline_existing, write_equation, draw_arrow, tick_answer.
  - Use handwriting-style strokes for drawing with slight randomized jitter.
  - Write equations in the largest empty space of the image.
  - Animate underlines, arrows, and tick/diagonal line slashes progressively.
  - Sync equation-writing durations to the audio narration.
  - Reveal written equations token/word-by-word at a natural writing speed.
  - Draw a diagonal slash line crossing through the correct option indicator (e.g. (C)).
  - Use premium Windows handwriting font (Ink Free / Segoe Print).
  - Render square roots dynamically using hand-drawn lines to avoid missing font glyph boxes.
"""

import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoClip, AudioFileClip, vfx
import math
import random
import re
# ── Leaf modules extracted in Step 1 of the render_video refactor ────────────
# Behaviour-preserving: the SAME functions/constants, relocated into the render/
# package and imported back so the rest of this module is unchanged.
from render.constants import (
    PEN_COLOR, PEN_WIDTH, ANSWER_INK, MATCH_INK, WRITE_ACTIONS,
    _SUPERSCRIPT_MAP, _SUBSCRIPT_MAP,
)
from render.text_utils import (
    _contains_devanagari, _sanitize_text, _is_formula_like_text,
    split_grapheme_clusters, wrap_text_to_width, split_into_math_tokens,
)
from render.fonts import (
    _find_hindi_font, _find_font, _font_has_glyph, _glyph_fallback_fonts,
    _resolve_glyph_font, _sized_sub_font, _sized_variant, _note_font_size,
)
from render.strokes import (
    _draw_handwritten_line, _draw_progressive_polyline, _draw_progressive_underline,
    _draw_progressive_circle, _draw_progressive_arrow, _draw_progressive_diagonal_slash,
    _draw_progressive_ellipse, _draw_progressive_cross,
)
# ── Modules extracted in Step 2 of the render_video refactor ─────────────────
from render.text_render import (
    draw_custom_text, get_custom_text_width, draw_math_equation_with_radicals,
    _measure_block, _layout_text_lines, _render_text_layer, _build_text_layers,
    _paste_text_reveal,
)
from render.geometry import (
    _boxes_overlap, _segment_hits_rect, _arrow_crosses_text, _underline_for_box,
    _next_clear_y, _find_slot, _snap_box_to_ocr, _resolve_box, _coords_box,
)
# ── matching module (Step 3) ──
from render.matching import (
    _target_cue, _route_match_pairs, _is_matching_timeline,
    _matching_table_bounds, _option_box_or_infer,
)
# ── verdicts module (Step 3) ──
from render.verdicts import _resolve_verdict_box, _verdict_row_positions


def _is_workspace_write_action(ann):
    """Treat generated formula notes as worked-solution lines."""
    action = ann.get("action")
    return action in WRITE_ACTIONS or (
        action == "write_note" and _is_formula_like_text(ann.get("text", ""))
    )


# ── Proportional substring bounds estimator ───────────────────────────────
def get_substring_bounds(elem, target_substring):
    """
    Estimate the bounding box of a substring inside an OCRElement
    proproportionally to character indices.
    """
    text = elem.text
    # Find start index of target_substring in text (case-insensitive)
    start_idx = text.lower().find(target_substring.lower())
    if start_idx == -1:
        return elem.x1, elem.y1, elem.x2, elem.y2
        
    end_idx = start_idx + len(target_substring)
    L = len(text)
    
    # Proportional estimation of x coordinates
    sub_x1 = elem.x1 + int((elem.x2 - elem.x1) * (start_idx / L))
    sub_x2 = elem.x1 + int((elem.x2 - elem.x1) * (end_idx / L))
    
    # y coordinates remain the same
    return sub_x1, elem.y1, sub_x2, elem.y2


# Free-form working-note actions: written in scattered empty space, optionally
# anchored to a question word (with a connecting arrow).
NOTE_ACTIONS = ("annotate_word", "write_note")
# Actions that mark the correct option with a solid line/slash.
ANSWER_ACTIONS = ("tick_answer", "mark_answer")
# Per-statement verdict (green tick = true / red cross = false) beside a statement,
# for assertion-reason and "how many statements are correct" questions.
VERDICT_ACTIONS = ("verdict_mark",)
# Every action that writes text and is revealed with the crop wipe.
TEXT_ACTIONS = ("annotate_word", "write_note", "fill_placeholder")


def _label_index(lab):
    """Map a single placeholder letter to its sequence index (A->0, B->1, ...)."""
    lab = re.sub(r"[^A-Za-z]", "", str(lab)).upper()[:1]
    return (ord(lab) - ord("A")) if lab else None


def _fit_linear(xs, ys):
    """
    Least-squares fit y = m*x + c over the given points. Returns (m, c), or
    (0, mean) when the x's are degenerate (e.g. a perfectly vertical column,
    where the coordinate is constant across indices).
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom < 1e-6:
        return 0.0, my
    m = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / denom
    return m, my - m * mx


def _snap_to_column(px, known_cx, tol):
    """
    Snap a predicted x to the nearest detected column centre.

    Diagram blanks usually line up in a few vertical columns. A plain linear fit
    of x-vs-letter-index gets dragged sideways by a zig-zag layout (e.g. (B) in
    the left column while (A)/(C)/(D) are on the right), so we cluster the known
    x's into columns and snap the prediction onto the closest one when it is near.
    """
    if not known_cx:
        return px
    # Greedy 1-D clustering of column centres.
    cols = []
    for x in sorted(known_cx):
        if cols and x - cols[-1][-1] <= tol:
            cols[-1].append(x)
        else:
            cols.append([x])
    centres = [sum(c) / len(c) for c in cols]
    nearest = min(centres, key=lambda c: abs(c - px))
    return nearest if abs(nearest - px) <= tol else px


def infer_missing_placeholders(placeholders, annotations, W, H):
    """
    Geometrically infer the position of a diagram blank that neither OCR nor
    Gemini located reliably.

    Flowchart/diagram blanks (A), (B), (C), (D) are laid out on a regular grid —
    rows that advance with the letter index, in one or a few columns. So when a
    label is missing we fit cy(index) through the labels we DID find (rows are
    monotonic and reliable), predict the column with a fit that is then SNAPPED
    onto the nearest detected column, and place the blank there. Needs >= 2 known
    anchors. Returns {LABEL: (box, confident)} where `confident` is True when the
    row fit is clean (so the caller can prefer it over Gemini's box_2d, which we
    have observed to hallucinate placeholder coordinates).
    """
    if not placeholders or len(placeholders) < 2:
        return {}

    # Known anchors: index -> (cx, cy); also track a typical blank size.
    known = {}
    widths, heights = [], []
    for lab, box in placeholders.items():
        idx = _label_index(lab)
        if idx is None:
            continue
        x1, y1, x2, y2 = box
        known[idx] = ((x1 + x2) / 2, (y1 + y2) / 2)
        widths.append(x2 - x1)
        heights.append(y2 - y1)
    if len(known) < 2:
        return {}

    # Which labels does the timeline want to fill that we still can't place?
    wanted = set()
    for ann in annotations:
        if ann.get("action") != "fill_placeholder":
            continue
        lab = re.sub(r"[^A-Za-z]", "", str(ann.get("label") or ann.get("target") or "")).upper()[:1]
        idx = _label_index(lab)
        if idx is None or lab in placeholders:  # OCR already has it
            continue
        wanted.add(idx)
    if not wanted:
        return {}

    idxs = sorted(known)
    known_cx = [known[i][0] for i in idxs]
    my, cy = _fit_linear(idxs, [known[i][1] for i in idxs])
    mx, cx = _fit_linear(idxs, known_cx)
    bw = sorted(widths)[len(widths) // 2]
    bh = sorted(heights)[len(heights) // 2]

    # Row-fit residual: if the y's are well explained by a line of the index, the
    # layout is genuinely row-ordered and we trust the inference over box_2d.
    resid = max(abs(known[i][1] - (my * i + cy)) for i in idxs)
    confident = resid <= 0.06 * H
    col_tol = max(40, 0.06 * W)

    inferred = {}
    for idx in wanted:
        # Only interpolate / mild-extrapolate near the known range.
        if idx < idxs[0] - 1 or idx > idxs[-1] + 1:
            continue
        py = my * idx + cy
        px = _snap_to_column(mx * idx + cx, known_cx, col_tol)
        x1 = int(max(0, min(px - bw / 2, W - bw)))
        y1 = int(max(0, min(py - bh / 2, H - bh)))
        inferred[chr(ord("A") + idx)] = ((x1, y1, x1 + int(bw), y1 + int(bh)), confident)
    return inferred


def _placeholder_box(ann, placeholders, W, H, inferred=None):
    """
    Locate a diagram placeholder for a fill action.

    Priority: OCR-detected box (ground truth) > a CONFIDENT geometric inference
    from the other blanks' regular layout > Gemini `box_2d` > a low-confidence
    geometric inference. Gemini's box_2d is demoted below confident inference
    because it has been observed to place blanks far from the actual figure node.
    """
    label = str(ann.get("label") or ann.get("target") or "")
    lab = re.sub(r"[^A-Za-z]", "", label).upper()[:1]
    ocr_ph = placeholders.get(lab) if placeholders else None
    if ocr_ph:  # OCR found the "(X)" token — ground truth.
        return ocr_ph
    inf = inferred.get(lab) if inferred else None  # (box, confident) or None
    if inf and inf[1]:
        return inf[0]
    coords = _coords_box(ann, W, H)
    if coords:
        return coords
    if inf:
        return inf[0]
    return None


# ── Schematic diagram engine (flowchart / sequence / cycle / axis) ──────────
# A generic node-graph builder: Gemini returns a `diagram` spec (nodes + edges),
# we auto-lay it out in empty space and draw it progressively as hand-drawn
# boxes + arrows. Covers hormone axes (with feedback loops), ordered sequences,
# cause→effect flows and cycles — the dominant diagram types in the corpus.

def _rounded_rect_outline(box, r=9, step=11):
    """A closed polyline tracing a rounded rectangle (for progressive stroking)."""
    x0, y0, x1, y1 = box
    r = min(r, (x1 - x0) / 2 - 1, (y1 - y0) / 2 - 1)
    segs = [((x0 + r, y0), (x1 - r, y0)), ((x1 - r, y0), (x1, y0 + r)),
            ((x1, y0 + r), (x1, y1 - r)), ((x1, y1 - r), (x1 - r, y1)),
            ((x1 - r, y1), (x0 + r, y1)), ((x0 + r, y1), (x0, y1 - r)),
            ((x0, y1 - r), (x0, y0 + r)), ((x0, y0 + r), (x0 + r, y0))]
    pts = []
    for (a, b) in segs:
        (ax, ay), (bx, by) = a, b
        L = math.hypot(bx - ax, by - ay)
        n = max(2, int(L / step))
        for i in range(n + 1):
            t = i / n
            pts.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    return pts


def _measure_node(text, font, mdraw, pad=12, min_w=120, wrap_w=210):
    """Wrap a node label and return (lines, box_w, box_h, line_height)."""
    raw = [ln for ln in str(text).split("\n") if ln.strip()] or [str(text)]
    lines = []
    for ln in raw:
        lines.extend(wrap_text_to_width(mdraw, ln, font, wrap_w))
    w = int(max((mdraw.textlength(l, font=font) for l in lines), default=10))
    line_h = int(font.size * 1.32)
    return lines, max(min_w, w + 2 * pad), line_h * len(lines) + 2 * pad, line_h


def _edge_ends(b0, b1):
    """Pick the natural attach points (box edge → box edge) for an arrow."""
    c0 = ((b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2)
    c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
    if abs(c1[1] - c0[1]) >= abs(c1[0] - c0[0]):          # mostly vertical
        return ((c0[0], b0[3]), (c1[0], b1[1])) if c1[1] > c0[1] else ((c0[0], b0[1]), (c1[0], b1[3]))
    return ((b0[2], c0[1]), (b1[0], c1[1])) if c1[0] > c0[0] else ((b0[0], c0[1]), (b1[2], c1[1]))


def _layout_diagram(spec, region, fonts, mdraw):
    """Position a diagram spec's nodes/edges inside `region`; return draw steps."""
    nodes = spec.get("nodes") or []
    edges = spec.get("edges") or []
    title = (spec.get("title") or "").strip()
    n = len(nodes)
    if n == 0:
        return None
    hindi_font = fonts[1] if len(fonts) > 1 else fonts[0]
    nfont = _sized_variant(hindi_font, 20)
    tfont = _sized_variant(hindi_font, 22)
    rx0, ry0, rx1, ry1 = region
    rw, rh = rx1 - rx0, ry1 - ry0

    order, measured, maxw, maxh = [], {}, 0, 0
    for i, nd in enumerate(nodes):
        nid = str(nd.get("id") or nd.get("label") or i)
        lines, bw, bh, lh = _measure_node(nd.get("label", ""), nfont, mdraw)
        measured[nid] = dict(lines=lines, lh=lh, hl=bool(nd.get("highlight")))
        order.append(nid)
        maxw, maxh = max(maxw, bw), max(maxh, bh)
    maxw = int(min(maxw, rw - 8))

    hint = (spec.get("layout") or "").lower()
    if hint not in ("vertical", "axis", "horizontal", "row", "snake"):
        hint = "vertical" if (rh >= rw * 0.8 and n <= 6) else "snake"
    title_h = 36 if title else 6
    ay0 = ry0 + title_h
    pos = {}

    if hint in ("vertical", "axis"):
        # Safety: if the column of nodes would overflow the region height (too many
        # nodes), shrink the node boxes so the whole chain still fits on the board.
        avail = ry1 - ay0
        if n > 1 and n * maxh + (n - 1) * 16 > avail:
            maxh = max(30, int((avail - (n - 1) * 16) / n))
        gap = (ry1 - ay0 - n * maxh) / (n - 1) if n > 1 else 20
        gap = max(14, min(gap, 58))
        total = n * maxh + (n - 1) * gap
        sy = ay0 + max(0, ((ry1 - ay0) - total) / 2)
        cx = (rx0 + rx1) / 2
        for i, nid in enumerate(order):
            y = sy + i * (maxh + gap)
            pos[nid] = (cx - maxw / 2, y, cx + maxw / 2, y + maxh)
    elif hint in ("horizontal", "row"):
        gap = (rw - n * maxw) / (n - 1) if n > 1 else 20
        gap = max(18, min(gap, 80))
        total = n * maxw + (n - 1) * gap
        sx = rx0 + max(0, (rw - total) / 2)
        cy = (ay0 + ry1) / 2
        for i, nid in enumerate(order):
            x = sx + i * (maxw + gap)
            pos[nid] = (x, cy - maxh / 2, x + maxw, cy + maxh / 2)
    else:                                                  # snake
        per_row = max(1, min(n, 4, int((rw + 30) // (maxw + 30))))
        rows = math.ceil(n / per_row)
        gx = max(20, min((rw - per_row * maxw) / (per_row + 1), 90))
        row_h = min(maxh + 90, maxh + max(40, (ry1 - ay0 - rows * maxh) / max(1, rows)))
        for i, nid in enumerate(order):
            r, c = divmod(i, per_row)
            if r % 2 == 1:
                c = per_row - 1 - c                        # reverse alternate rows
            pos[nid] = (rx0 + gx + c * (maxw + gx), ay0 + r * row_h,
                        rx0 + gx + c * (maxw + gx) + maxw, ay0 + r * row_h + maxh)

    idx_of = {nid: i for i, nid in enumerate(order)}
    edge_steps = []
    for e in edges:
        f, t = str(e.get("from")), str(e.get("to"))
        if f not in pos or t not in pos:
            continue
        b0, b1 = pos[f], pos[t]
        kind = (e.get("kind") or "").lower()
        # Route an edge that SKIPS a node (|index gap| > 1, in either direction) out
        # along the side as an elbow, so its line never cuts through an intermediate
        # node and its label never lands on that node's text.
        if kind in ("feedback", "loop") or abs(idx_of.get(t, 0) - idx_of.get(f, 0)) > 1:
            sp = (b0[2], (b0[1] + b0[3]) / 2)
            ep = (b1[2], (b1[1] + b1[3]) / 2)
            rxr = min(rx1 - 6, max(b0[2], b1[2]) + 42)
            edge_steps.append(dict(kind="feedback",
                                   pts=[sp, (rxr, sp[1]), (rxr, ep[1]), ep],
                                   label=(e.get("label") or "")))
        else:
            p0, p1 = _edge_ends(b0, b1)
            edge_steps.append(dict(kind="edge", p0=p0, p1=p1, label=(e.get("label") or "")))

    steps = []
    # Center the diagram title horizontally over the actual center of the nodes
    cx_nodes = (rx0 + rx1) / 2
    if order:
        xs_nodes = [pos[nid][0] for nid in order] + [pos[nid][2] for nid in order]
        if xs_nodes:
            cx_nodes = (min(xs_nodes) + max(xs_nodes)) / 2

    min_node_y = min(pos[nid][1] for nid in order)
    title_top = ry0
    if title:
        title_w = int(mdraw.textlength(title, font=tfont))
        tx = max(rx0 + 6, int(cx_nodes - title_w / 2))
        # Sit the title just ABOVE the first node so it reads as that block's
        # heading, instead of floating at the region top (where it collided with
        # nearby notes when the nodes were vertically centred lower down).
        title_top = max(ry0, int(min_node_y - 30))
        steps.append(dict(kind="title", pos=(tx, title_top), text=title, font=tfont))
    for nid in order:
        steps.append(dict(kind="node", box=pos[nid], lines=measured[nid]["lines"],
                          lh=measured[nid]["lh"], font=nfont,
                          hl=measured[nid].get("hl", False)))
    steps.extend(edge_steps)

    xs = [pos[n][0] for n in order] + [pos[n][2] for n in order]
    ys = [pos[n][1] for n in order] + [pos[n][3] for n in order]
    bbox = (min(xs) - 12, min(title_top, min_node_y) - 4, max(xs) + 50, max(ys) + 12)
    return dict(steps=steps, bbox=bbox, label_font=_sized_variant(hindi_font, 16),
                node_boxes={nid: pos[nid] for nid in order})


def _reveal_lines_centered(frame, lines, font, color, box, line_h, frac):
    """Crop-reveal (left→right) a node's centred multi-line label onto `frame`."""
    cx = (box[0] + box[2]) / 2
    layers = [_render_text_layer(l, font, color) for l in lines]
    total_w = sum(w for _, w, _ in layers) or 1
    y0 = (box[1] + box[3]) / 2 - line_h * len(lines) / 2 + 4
    reveal = frac * total_w
    for i, (layer, w, h) in enumerate(layers):
        if reveal <= 0:
            break
        show = int(min(w, reveal))
        if show > 0:
            crop = layer.crop((0, 0, show, h))
            frame.paste(crop, (int(cx - w / 2), int(y0 + i * line_h)), crop)
        reveal -= w


def _draw_edge_label(frame, text, font, color, mid):
    """Draw a short edge label with a white backing so it stays legible."""
    if not text:
        return
    layer, w, h = _render_text_layer(_sanitize_text(text), font, color)
    x, y = int(mid[0] - w / 2), int(mid[1] - h / 2)
    frame.paste((255, 255, 255), (x - 1, y, x + w - 1, y + h))
    frame.paste(layer, (x, y), layer)


def _draw_polyline_progressive(draw, pts, frac, color, width=PEN_WIDTH):
    """Draw the first `frac` (by arc length) of a polyline; returns last point."""
    seglens = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
               for i in range(len(pts) - 1)]
    total = sum(seglens) or 1
    target = frac * total
    out, acc = [pts[0]], 0.0
    last = pts[0]
    for i, L in enumerate(seglens):
        if acc + L <= target:
            out.append(pts[i + 1]); acc += L; last = pts[i + 1]
        else:
            t = (target - acc) / L if L > 0 else 0
            last = (pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t,
                    pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t)
            out.append(last); break
    if len(out) >= 2:
        draw.line(out, fill=color, width=width, joint="round")
    return last


def _arrowhead(draw, pfrom, pto, color, width=PEN_WIDTH, size=11):
    """Draw a small arrowhead at `pto` pointing away from `pfrom`."""
    x0, y0 = pfrom
    x1, y1 = pto
    a = math.atan2(y1 - y0, x1 - x0)
    for da in (math.radians(150), math.radians(-150)):
        draw.line([(x1, y1), (x1 + size * math.cos(a + da), y1 + size * math.sin(a + da))],
                  fill=color, width=width)


def _render_diagram(draw, frame, layout, progress, pen):
    """Reveal the diagram's ordered steps (title → nodes → edges) by `progress`."""
    steps = layout.get("steps") or []
    nfont = layout.get("label_font")
    n = len(steps)
    if n == 0:
        return
    for i, st in enumerate(steps):
        s0 = i / n
        if progress <= s0:
            break
        s1 = (i + 1) / n
        p = 1.0 if progress >= s1 else (progress - s0) / max(1e-6, s1 - s0)
        k = st["kind"]
        if k == "title":
            layer, w, h = _render_text_layer(_sanitize_text(st["text"]), st["font"], pen)
            show = int(min(w, max(0.0, p) * w))
            if show > 0:
                crop = layer.crop((0, 0, show, h))
                frame.paste(crop, (int(st["pos"][0]), int(st["pos"][1])), crop)
            if p > 0.9:
                _draw_handwritten_line(draw, st["pos"][0], st["pos"][1] + h + 3,
                                       st["pos"][0] + w, st["pos"][1] + h + 3, 2, pen)
        elif k == "node":
            # The answer node is drawn in green so the diagram visibly POINTS at the
            # correct option (e.g. the acellular layer), not just lists the parts.
            col = ANSWER_INK if st.get("hl") else pen
            pts = _rounded_rect_outline(st["box"])
            pb = min(1.0, p / 0.55)
            nn = max(2, int(len(pts) * pb))
            draw.line(pts[:nn], fill=col, width=PEN_WIDTH + (1 if st.get("hl") else 0),
                      joint="round")
            pt = max(0.0, (p - 0.55) / 0.45)
            if pt > 0:
                _reveal_lines_centered(frame, st["lines"], st["font"], col,
                                       st["box"], st["lh"], pt)
        elif k == "edge":
            _draw_progressive_arrow(draw, st["p0"][0], st["p0"][1], st["p1"][0], st["p1"][1],
                                    p, PEN_WIDTH, pen)
            if p >= 1.0 and st.get("label"):
                mid = ((st["p0"][0] + st["p1"][0]) / 2, (st["p0"][1] + st["p1"][1]) / 2)
                _draw_edge_label(frame, st["label"], nfont, pen, mid)
        elif k == "feedback":
            last = _draw_polyline_progressive(draw, st["pts"], p, pen)
            if p >= 1.0:
                _arrowhead(draw, st["pts"][-2], st["pts"][-1], pen)
                if st.get("label"):
                    top = min(pt[1] for pt in st["pts"])
                    _draw_edge_label(frame, st["label"], nfont, pen,
                                     (st["pts"][1][0], top - 2))


_VERDICT_WORDS = ("गलत", "सही", "सत्य", "असत्य", "true", "false",
                  "correct", "incorrect", "right", "wrong")
_STMT_LABELS = ("कथन", "अभिकथन", "कारण", "statement", "reason", "assertion")


def _prune_notes(annotations):
    """Defensive cleanup that mirrors the prompt's note rules, so even a stale or
    over-eager Gemini response renders clean:

      1. Drop write_notes that merely restate a verdict mark (the ✓/✗ already says
         "कथन I: गलत") — keep the final conclusion note that selects the option.
      2. Cap free notes (write_note + annotate_word) at 4; keep the conclusion plus
         the earliest informative notes, in original order.
    """
    def norm(s):
        return re.sub(r"\s+", " ", str(s or "")).strip().lower()

    def note_words(s):
        return [w for w in re.findall(r"[^\s,/():;=+\-]+", norm(s)) if len(w) >= 3]

    def near_duplicate(a, b):
        aw, bw = note_words(a.get("text")), note_words(b.get("text"))
        if not aw or not bw:
            return False
        aset, bset = set(aw), set(bw)
        overlap = len(aset & bset) / max(1, min(len(aset), len(bset)))
        an, bn = " ".join(aw), " ".join(bw)
        return overlap >= 0.82 or an in bn or bn in an

    verdict_targets = [norm(a.get("target")) for a in annotations
                       if a.get("action") == "verdict_mark"]

    def is_conclusion(note):
        txt = note.get("text", "")
        nt = norm(txt)
        if "→" in txt or "->" in txt:
            return True
        if sum(nt.count(s) for s in _STMT_LABELS) >= 2:
            return True
        return len(nt.split()) > 5

    def is_redundant_verdict(note):
        if note.get("action") != "write_note":
            return False
        nt = norm(note.get("text"))
        if not any(v in nt for v in _VERDICT_WORDS):
            return False
        if is_conclusion(note):
            return False
        # short single-statement judgement → already shown as a ✓/✗ mark
        if any(vt and vt in nt for vt in verdict_targets):
            return True
        return sum(nt.count(s) for s in _STMT_LABELS) == 1 and len(nt.split()) <= 4

    # Drop a note that merely RESTATES the diagram. The prompt forbids duplicating the
    # figure, but Gemini sometimes writes the same fact as both a node and a note → the
    # board says it twice. Only a STRONG overlap is pruned (>=2/3 of the note's content
    # words already appear as diagram labels), so notes that add a NEW fact survive.
    # Hindi inflects heavily, so words match by a shared 4+ char prefix, not equality.
    diagram_words = set()
    for a in annotations:
        if a.get("action") == "draw_diagram":
            dia = a.get("diagram") or {}
            labels = [dia.get("title", "")]
            labels += [n.get("label", "") for n in (dia.get("nodes") or [])]
            labels += [e.get("label", "") for e in (dia.get("edges") or [])]
            for lab in labels:
                diagram_words.update(w for w in re.findall(r"[^\s,/()]+", norm(lab))
                                     if len(w) >= 4)

    def _answer_conclusion(note):           # the option-selecting note — never prune it
        txt = note.get("text", "")
        return ("→" in txt or "->" in txt
                or sum(norm(txt).count(s) for s in _STMT_LABELS) >= 2)

    def duplicates_diagram(note):
        if not diagram_words or _answer_conclusion(note):
            return False
        words = [w for w in re.findall(r"[^\s,/()]+", norm(note.get("text")))
                 if len(w) >= 4]
        # Only prune a full sentence-like FACT (>=4 content words) the diagram already
        # states. A short note (a 2-word label/translation like the Sertoli-cell name,
        # or a fragment) earns its place even if its words appear in the figure — generic
        # domain words (कोशिका/शुक्राणु) recur across nodes and would otherwise false-match.
        if len(words) < 4:
            return False
        hits = sum(1 for w in words
                   if any(w[:4] == d[:4] and (w.startswith(d) or d.startswith(w))
                          for d in diagram_words))
        return hits / len(words) >= 0.67

    notes = [a for a in annotations if a.get("action") in ("write_note", "annotate_word")]
    survivors = [n for n in notes
                 if not is_redundant_verdict(n) and not duplicates_diagram(n)]

    deduped = []
    for n in survivors:
        if any(near_duplicate(n, old) for old in deduped):
            continue
        deduped.append(n)
    survivors = deduped

    concl = [n for n in survivors if is_conclusion(n)][:2]
    reg = [n for n in survivors if n not in concl]
    keep_ids = {id(n) for n in concl + reg[:max(0, 4 - len(concl))]}

    pruned = [a for a in annotations
              if a.get("action") not in ("write_note", "annotate_word") or id(a) in keep_ids]

    # Matching tables are a special case: the printed table is the primary visual.
    # Extra explainers beside the table compete with the pair strokes and can make a
    # correct mapping feel uncertain. Keep the table clean: underlines, match_pair
    # connectors, and the final answer mark are enough.
    if _is_matching_timeline(pruned):
        cleaned = []
        matching_dropped = 0
        for a in pruned:
            act = a.get("action")
            if act in ("draw_diagram", "annotate_word", "write_note"):
                matching_dropped += 1
                continue
            cleaned.append(a)
        pruned = cleaned
        if matching_dropped:
            print(f"  Matching cleanup dropped {matching_dropped} table note/duplicate diagram action(s)")
    dropped = len(annotations) - len(pruned)
    if dropped:
        print(f"  Pruned {dropped} redundant/excess note(s) (verdict-duplicate or >4 cap)")
    return pruned


# ── Animation schedule ──────────────────────────────────────────────────────
def _build_schedule(annotations, total_duration, enriched_ocr, option_positions,
                    fonts=None, image_size=(1280, 720), pen=PEN_COLOR):
    """
    Pre-compute geometry, layout and timing for every annotation.

    New behaviours vs. the original single-column writer:
      - `annotate_word`/`write_note` are placed in scattered empty space (never
        overlapping printed text), with dynamic font size and optional arrows.
      - underlines / answer marks are solid lines anchored to OCR'd text.
    Legacy `write_equation`/`write_text` keep the simple column layout so the
    English math flow is unchanged.
    """
    schedule = []
    W, H = image_size

    annotations = _prune_notes(annotations)

    font_body = fonts[0] if fonts else _find_font("body", 28)
    hindi_font = fonts[1] if fonts and len(fonts) > 1 else _find_hindi_font(30)
    _measure_img = Image.new("RGB", (10, 10))
    _measure_draw = ImageDraw.Draw(_measure_img)

    ocr_index = enriched_ocr.get("index") if enriched_ocr else None
    free_spaces = enriched_ocr.get("free_spaces", []) if enriched_ocr else []
    placeholders = enriched_ocr.get("placeholders", {}) if enriched_ocr else {}
    # Fill in any blank the timeline references that neither OCR nor Gemini
    # located, by extrapolating from the regular layout of the detected blanks.
    inferred_ph = infer_missing_placeholders(placeholders, annotations, W, H)
    if inferred_ph:
        desc = ", ".join(f"{k}{'*' if v[1] else '?'}" for k, v in sorted(inferred_ph.items()))
        print(f"  Geometrically inferred placeholder(s): {desc}  (* = confident)")
    rng = random.Random(20240623)  # seeded → reproducible but scattered layout

    # Occupancy map: every printed OCR box is an obstacle, plus the PW logo band.
    occupied = []
    for el in (enriched_ocr.get("elements") or []):
        b = el.get("bounds")
        if b:
            occupied.append(tuple(b))
    occupied.append((W - 175, 0, W, 150))  # top-right watermark logo
    protected_table = _matching_table_bounds(
        annotations, enriched_ocr.get("elements") or [], option_positions, W, H)
    if protected_table:
        occupied.append(protected_table)

    # Soft block over the flowchart/diagram node cluster so scattered NOTES don't
    # land on the figure. (fill_placeholder still writes right next to its blank.)
    opt_top = min((min(p[1] for p in v) for v in option_positions.values()), default=H)

    # Statements/options divider for verdict marks. Deriving it from option markers
    # is fragile: if EasyOCR misreads one marker (e.g. "(A)" → garbage), opt_top drops
    # INTO the option block and a "कथन I" inside an option would be mistaken for the
    # statement heading. The instruction line ("...विकल्पों में से सही उत्तर चुनिए" /
    # "...कूट का प्रयोग कर...") reliably sits between the statements and the options —
    # use its top as the ceiling so a verdict only ever lands on a statement heading.
    _INSTR_KEYS = ("विकल्प", "चुनिए", "कूट", "निम्नलिखित में से सही", "सही उत्तर")
    instr_tops = [b[1] for el in (enriched_ocr.get("elements") or [])
                  for b in [el.get("bounds")]
                  if b and b[1] < opt_top and any(k in (el.get("text") or "") for k in _INSTR_KEYS)]
    stmt_ceiling = min(instr_tops) if instr_tops else opt_top

    dxs, dys = [], []
    for el in (enriched_ocr.get("elements") or []):
        b = el.get("bounds")
        txt = (el.get("text") or "").strip()
        if b and 120 < (b[1] + b[3]) / 2 < opt_top - 10 and len(txt) < 25 and not re.search(r"\d{4}", txt):
            dxs += [b[0], b[2]]
            dys += [b[1], b[3]]
    diagram_block = (min(dxs) - 6, min(dys) - 6, max(dxs) + 6, max(dys) + 6) if dxs else None
    note_block = [b for b in (diagram_block, protected_table) if b]

    # Workspace zone: the clear right-margin column to the RIGHT of the printed
    # question text, where free factual notes stack tidily (legible, teacher-like)
    # instead of scattering over the slide. Derived from where the text ends.
    text_rights = [b[2] for b in occupied if b[2] < W - 50]   # exclude watermark band
    text_right = max(text_rights) if text_rights else int(0.5 * W)
    ws_zone_x0 = int(max(0.5 * W, min(W - 280, text_right + 24)))

    # Legacy single-column cursor (used only by write_equation/write_text).
    if free_spaces:
        rx1, ry1, rx2, ry2 = free_spaces[0]["bounds"]
        print(f"  Writing layout region selected: {free_spaces[0]['position']} bounds: {free_spaces[0]['bounds']}")
    else:
        rx1, ry1, rx2, ry2 = 680, 100, W - 40, H - 40
    wx = rx1 + 25
    wy = max(ry1 + 30, 150)
    region_w = max(200, (rx2 - rx1) - 50)
    fallback_y = max(200, int(H * 0.55))  # bottom fallback when no slot fits
    ph_legend = {"x": None, "y": None}    # running cursor for the fill-in legend

    # Per-statement ✓/✗ column for statement-evaluation / "how many are correct"
    # questions. Reconstructed ONCE from OCR geometry + Gemini box_2d order so that
    # every statement reliably gets its mark (a garbled enumerator must never drop
    # one) and the marks line up in a clean right-margin column.
    verdict_rows = _verdict_row_positions(
        annotations, enriched_ocr.get("elements") or [], stmt_ceiling, W, ws_zone_x0)

    # ── Wrong-option elimination geometry ────────────────────────────────────
    # Each rejected option gets ONE clean strikethrough across its whole row,
    # located by its (A)/(B)/(C)/(D) marker. This replaces ambiguous per-word
    # cross-outs that pile onto the topmost option (a word like "शुक्रवाहक"
    # appears in several options; resolving it grabs the wrong row).
    _opt_elems = [el.get("bounds") for el in (enriched_ocr.get("elements") or [])
                  if el.get("bounds")]
    opt_rows = {}
    for _L, _bbox in option_positions.items():
        _xs = [p[0] for p in _bbox]
        _ys = [p[1] for p in _bbox]
        opt_rows[_L] = (min(_xs), (min(_ys) + max(_ys)) / 2, min(_ys), max(_ys))

    def _option_row_right(cy, x_left):
        rights = [b[2] for b in _opt_elems
                  if abs((b[1] + b[3]) / 2 - cy) < 16 and b[2] > x_left]
        return max(rights) if rights else x_left + 320

    # A long MCQ option wraps onto 2+ lines; a single line-1 strike leaves the rest
    # uncrossed and reads as a half-done elimination. Compute ONE strike segment per
    # text line of an option so the whole option gets struck — each segment hugging
    # that line's own text (so the strike never overshoots into the blank margin).
    _marker_tops = sorted(opt_rows[L][2] for L in opt_rows)

    def _option_bottom(L):
        top = opt_rows[L][2]
        below = [mt for mt in _marker_tops if mt > top + 4]
        if below:
            return below[0]                      # next option's marker top bounds this one
        gaps = [b - a for a, b in zip(_marker_tops, _marker_tops[1:])]
        return top + (max(gaps) if gaps else 60)  # last option: one typical option height

    def _option_strike_segments(L):
        # Geometry-driven (NOT OCR line-clustering): EasyOCR routinely emits a tall
        # garbled box that spans both wrapped lines, which chains them into one cluster
        # and collapses the strike back to a single line. Instead, derive the line
        # COUNT from the option's vertical extent ÷ its marker height, place a strike at
        # each line, and hug that line's own text width via _option_row_right.
        top, bottom = opt_rows[L][2], _option_bottom(L)
        x_left, marker_cy = opt_rows[L][0], opt_rows[L][1]
        max_strike_x = min(ws_zone_x0 - 18, x_left + 430, 0.56 * W)
        mb_h = opt_rows[L][3] - top                       # marker box height ≈ line height
        line_h = mb_h if 18 <= mb_h <= 40 else 28
        n_lines = max(1, min(4, round((bottom - top) / line_h)))
        segs = []
        for i in range(n_lines):
            cy = marker_cy + i * line_h
            rights = [b[2] for b in _opt_elems
                      if abs((b[1] + b[3]) / 2 - cy) < 14 and b[2] > x_left
                      and b[0] < 0.62 * W]
            if not rights:               # no text on this line → don't strike empty space
                continue
            segs.append((x_left - 4, cy, min(max(rights) + 6, max_strike_x)))
        return segs or [(x_left - 4, marker_cy,
                         min(_option_row_right(marker_cy, x_left) + 6, max_strike_x))]

    answer_opt = None
    for _a in annotations:
        if _a.get("action") in ANSWER_ACTIONS:
            _t = re.sub(r"[^A-Da-d]", "",
                        str(_a.get("target") or _a.get("option") or "")).upper()
            if len(_t) == 1:
                answer_opt = _t
    struck_options = set()

    # Worked-solution workspace: size the step font/spacing so ALL derivation
    # lines fit the available height (a long numerical solution shrinks to fit
    # rather than overflowing off-screen).
    n_steps = sum(1 for a in annotations if _is_workspace_write_action(a))
    step_avail_h = max(80, ry2 - wy - 20)
    if n_steps > 0:
        per_line = step_avail_h / n_steps
        step_fs = int(max(15, min(int(font_body.size), (per_line - 12) / 1.4)))
    else:
        step_fs = int(font_body.size)
    step_font = _sized_variant(font_body, step_fs)
    # Horizontal fit: shrink so the widest Latin/math step stays inside the column.
    widest = 0.0
    for a in annotations:
        if _is_workspace_write_action(a):
            st = _sanitize_text(a.get("text", ""))
            if st and not _contains_devanagari(st):
                try:
                    widest = max(widest, _measure_draw.textlength(st, font=step_font))
                except Exception:
                    pass
    if widest > region_w > 0:
        step_fs = int(max(13, step_fs * region_w / widest))
        step_font = _sized_variant(font_body, step_fs)
    step_hindi_font = _sized_variant(hindi_font, step_fs + 2)
    step_gap = max(6, int(step_fs * 0.45))

    def _compute_diagram(spec, occ):
        """Lay a diagram spec into the larger clear region; return its layout dict."""
        text_bottom = max((b[3] for b in occ if b[3] < H - 20 and b[0] < W - 50),
                          default=int(0.5 * H))
        bottom_x2 = ws_zone_x0 - 24 if (n_steps > 0 and ws_zone_x0 - 24 > 340) else W - 24
        bottom_region = (24, int(text_bottom) + 18, bottom_x2, H - 24)
        right_top = 150
        col_notes = [b[3] for b in occ if b[2] > ws_zone_x0 and b[1] < 0.5 * H]
        if col_notes:
            right_top = max(right_top, int(max(col_notes)) + 18)
        right_region = (ws_zone_x0, right_top, W - 18, H - 24)
        area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])
        hint = (spec.get("layout") or "").lower()
        if hint in ("vertical", "axis"):
            region = right_region if area(right_region) >= 0.5 * area(bottom_region) else bottom_region
        else:
            bottom_h = bottom_region[3] - bottom_region[1]
            region = bottom_region if bottom_h >= 150 else right_region
        return _layout_diagram(spec, region, (font_body, hindi_font), _measure_draw)

    # Pre-reserve each diagram's footprint BEFORE the main loop places any notes, so a
    # note timed earlier than the diagram is never parked where the figure will be
    # drawn (which produced notes sitting on top of the diagram). The big figure wins
    # its space first; notes then flow into whatever is left.
    diagram_layouts = {}
    diagram_node_boxes = {}    # node-id -> box, so a fill_placeholder that targets a
    for i, ann in enumerate(annotations):  # diagram node lands AT the node, not a legend
        if ann.get("action") == "draw_diagram" and ann.get("diagram"):
            dg = _compute_diagram(ann["diagram"], occupied)
            if dg:
                diagram_layouts[i] = dg
                occupied.append(dg["bbox"])
                for nid, box in (dg.get("node_boxes") or {}).items():
                    diagram_node_boxes[str(nid)] = box

    # Pre-route all matching connectors together: lane assignment needs to see
    # every pair at once so they fan across the gutter instead of overlapping.
    match_routes = _route_match_pairs(annotations, ocr_index, option_positions, W, H)

    temp_schedule = []
    for i, ann in enumerate(annotations):
        action = ann["action"]
        t = ann["time"]
        entry = {**ann, "write_start": t, "write_end": t + 0.8}
        if action == "write_note" and _is_formula_like_text(ann.get("text", "")):
            action = "write_step"
            entry["action"] = action

        if action in ("circle_word", "circle_existing"):
            entry["write_end"] = t + 1.0
            box = _resolve_box(ann, ocr_index, W, H, option_positions=option_positions)
            entry["ellipse_params"] = None
            if box:
                x1, y1, x2, y2 = box
                # Tight, vertically-centred ring that HUGS the exact word/phrase
                # (small even padding) instead of a loose oval that cuts through it.
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                rx = max(14, (x2 - x1) / 2 + 8)
                ry = max(11, (y2 - y1) / 2 + 7)
                entry["ellipse_params"] = (cx, cy, rx, ry)
                occupied.append((cx - rx, cy - ry, cx + rx, cy + ry))

        elif action == "underline_existing":
            entry["write_end"] = t + 1.0
            box = _resolve_box(ann, ocr_index, W, H, option_positions=option_positions)
            # A thin underline is intentionally NOT added as an obstacle, so a
            # word's meaning can still be written just below it.
            entry["underline_params"] = _underline_for_box(box, occupied, W, H) if box else None

        elif action == "cross_out_word":
            entry["write_end"] = t + 0.8
            tgt = str(ann.get("target", "") or "")
            letter = re.sub(r"[^A-Da-d]", "", tgt).upper()
            is_marker = len(letter) == 1 and len(tgt.strip()) <= 4 and len(option_positions) >= 2
            opt_L = None
            box = None
            if is_marker:
                opt_L = letter                       # explicit marker target "(A)"
            else:
                box = _resolve_box(ann, ocr_index, W, H, option_positions=option_positions)
                if box:                              # word target → which option row?
                    bcy = (box[1] + box[3]) / 2
                    for L, (xl, cy, y1, y2) in opt_rows.items():
                        if y1 - 6 <= bcy <= y2 + 6:
                            opt_L = L
                            break
            if opt_L is not None:
                # One strike per wrong option; never strike the correct answer
                # (an ambiguous word may resolve into the wrong/right row).
                if opt_L != answer_opt and opt_L not in struck_options:
                    if opt_L in opt_rows:
                        xl, cy = opt_rows[opt_L][0], opt_rows[opt_L][1]
                    else:                            # marker OCR-missed → extrapolate its row
                        ib = _option_box_or_infer(opt_L, option_positions)
                        xl, cy = (ib[0], (ib[1] + ib[3]) / 2) if ib else (None, None)
                    if xl is not None:
                        struck_options.add(opt_L)
                        if opt_L in opt_rows:        # strike EVERY line of the option
                            entry["strike_lines"] = _option_strike_segments(opt_L)
                        else:                        # marker OCR-missed → single inferred row
                            entry["strike_lines"] = [
                                (xl - 4, cy, _option_row_right(cy, xl) + 6)]
                # else: suppress (redundant, or would cross the answer)
            else:
                entry["cross_params"] = box          # genuine stem-word cross → an X

        elif action in NOTE_ACTIONS:  # annotate_word, write_note
            text = _sanitize_text(ann.get("text", ""))
            target_str = ann.get("target")
            custom_box = ann.get("box") or ann.get("box_2d") or ann.get("box_norm")
            anchor = None
            if not custom_box and target_str:
                anchor = _resolve_box(ann, ocr_index, W, H, option_positions=option_positions)
            # A lone annotate_word ("अंडोत्सर्ग") is meaningless once it gets parked
            # away from its word, so make it self-contained: prefix a compact cue drawn
            # from its target ("एलएच तीव्र → अंडोत्सर्ग"). Harmless if an arrow connects.
            if (action == "annotate_word" and anchor and target_str
                    and "->" not in text and "→" not in text and len(text) <= 44):
                cue = _target_cue(target_str)
                if cue and cue not in text:
                    # ASCII arrow — the Devanagari font has no "→" glyph (renders tofu).
                    text = f"{cue} -> {text}"
            is_hi = _contains_devanagari(text)
            base = 25 if action == "annotate_word" else 28
            fnt = _sized_variant(hindi_font if is_hi else font_body,
                                 _note_font_size(text, base))
            max_w = 260 if action == "annotate_word" else 340
            layout = _build_text_layers(text, fnt, pen, max_w, _measure_draw)
            bw, bh = layout["block_w"], layout["block_h"]

            slot = None
            need_arrow = False
            if custom_box:
                slot = _resolve_box(ann, ocr_index, W, H, option_positions=option_positions)
            if slot is None:
                # 1) annotate_word: hug the target word directly when there's room.
                if anchor:
                    for (bx, by) in ((anchor[0], anchor[3] + 6), (anchor[0] - 12, anchor[3] + 8),
                                     (anchor[2] - bw, anchor[3] + 6)):
                        bx = max(8, min(int(bx), W - bw - 8)); by = int(by)
                        cand = (bx, by, bx + bw, by + bh)
                        if by + bh < H and not _boxes_overlap(cand, occupied + note_block, pad=3):
                            slot = cand; break
                # 2) tidy right-margin workspace column.
                if slot is None:
                    slot, _ = _find_slot(W, H, max(bw, 1), bh, occupied + note_block, rng,
                                         top_band=158, x_lo=ws_zone_x0, tidy=True)
                    if slot is not None and anchor:
                        need_arrow = True
                # 3) tidy LEFT columns (clear lower-left) — neat top-down stacks in empty
                #    space instead of a random scatter over the watermark/centre.
                if slot is None:
                    slot, _ = _find_slot(W, H, max(bw, 1), bh, occupied + note_block, rng,
                                         top_band=158, x_lo=16, tidy=True)
                    if slot is not None and anchor:
                        need_arrow = True
            if slot is None:
                slot = (40, fallback_y, 40 + bw, fallback_y + bh)
                fallback_y = min(H - bh - 10, fallback_y + bh + 14)
                need_arrow = (anchor is not None)
            occupied.append(slot)
            entry["write_pos"] = (slot[0], slot[1])
            entry["text_layout"] = layout
            # Subtle white backing card so the note stays legible over the slide's
            # faint watermark/background (matches the clean reference look).
            entry["note_card"] = (slot[0] - 8, slot[1] - 4, slot[0] + bw + 8, slot[1] + bh + 6)

            ap = ann.get("arrow_params")
            if ap and len(ap) == 4:
                entry["arrow_params"] = tuple(ap)
            elif need_arrow and anchor:
                # Connect the note to its word when the connector won't slice across
                # other text (the note now self-labels, so an unlinked one still reads).
                acx, acy = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
                scx, scy = slot[0] + 8, slot[1] + 6
                a_from, s_to = (acx, anchor[3] + 2), (scx, scy)
                if (math.hypot(acx - scx, acy - scy) <= 300
                        and not _arrow_crosses_text(a_from, s_to, occupied)):
                    entry["arrow_params"] = (acx, anchor[3] + 2, scx, scy)

        elif action == "fill_placeholder":  # answer for a printed-figure blank
            # CASE 1 — the fill targets a NODE of a drawn diagram (Gemini sometimes
            # "fills" a sequence/flowchart it just drew, label = node id like "n3").
            # Place the text BESIDE that node, synced to when it is spoken, instead of
            # a scattered "N = ..." legend disconnected from the figure.
            raw_label = str(ann.get("label") or ann.get("target") or "").strip()
            nbox = diagram_node_boxes.get(raw_label)
            if nbox is not None:
                value = _sanitize_text(ann.get("text", ""))
                is_hi = _contains_devanagari(value)
                fnt = _sized_variant(hindi_font if is_hi else font_body,
                                     _note_font_size(value, 22))
                layout = _build_text_layers(value, fnt, pen, 230, _measure_draw)
                bw, bh = layout["block_w"], layout["block_h"]
                ncy = (nbox[1] + nbox[3]) / 2
                prefer_left = (nbox[0] + nbox[2]) / 2 > 0.5 * W
                cands = []
                left = (int(nbox[0] - bw - 14), int(ncy - bh / 2))
                right = (int(nbox[2] + 14), int(ncy - bh / 2))
                below = (int(nbox[0]), int(nbox[3] + 4))
                cands = ([left, right] if prefer_left else [right, left]) + [below]
                lx = ly = None
                for cx, cy in cands:
                    cslot = (cx, cy, cx + bw, cy + bh)
                    if cx >= 6 and cx + bw <= W - 6 and not _boxes_overlap(cslot, occupied, pad=2):
                        lx, ly = cx, cy; break
                if lx is None:
                    lx, ly = below
                slot = (lx, ly, lx + bw, ly + bh)
                occupied.append(slot)
                entry["write_pos"] = (lx, ly)
                entry["text_layout"] = layout
                entry["note_card"] = (lx - 6, ly - 3, lx + bw + 6, ly + bh + 5)
                # short connector from the node edge to the milestone label
                if lx + bw <= nbox[0]:                       # label sits to the LEFT
                    entry["arrow_params"] = (nbox[0], ncy, lx + bw + 2, ly + bh / 2)
                elif lx >= nbox[2]:                          # label sits to the RIGHT
                    entry["arrow_params"] = (nbox[2], ncy, lx - 2, ly + bh / 2)
                temp_schedule.append(entry)
                continue
            # CASE 2 — a real printed-figure blank (A)(B)(C)(D).
            # The blanks are usually TINY cells in a dense printed figure — long Hindi
            # terms can't fit inside them, and scattering the answers around the figure
            # with loose arrows reads as a mess. So write them as a tidy LABELLED LEGEND
            # ("A = FSH") stacked in clear space, exactly like the teacher does. Each
            # entry still appears at the moment it is spoken (its own `time`).
            value = _sanitize_text(ann.get("text", ""))
            label = re.sub(r"[^A-Za-z0-9]", "",
                           str(ann.get("label") or ann.get("target") or "")).upper()[:1]
            text = f"{label} = {value}" if label else value
            is_hi = _contains_devanagari(text)
            fnt = _sized_variant(hindi_font if is_hi else font_body,
                                 _note_font_size(text, 27))
            layout = _build_text_layers(text, fnt, pen, 320, _measure_draw)
            bw, bh = layout["block_w"], layout["block_h"]

            if ph_legend["x"] is None:        # anchor the legend column in clear space
                ph_legend["x"] = int(max(ws_zone_x0, 0.52 * W))
                ph_legend["y"] = 172
            lx, ly = ph_legend["x"], ph_legend["y"]
            slot = (lx, ly, lx + bw, ly + bh)
            if ly + bh > H - 20 or _boxes_overlap(slot, occupied, pad=3):
                fb, _ = _find_slot(W, H, max(bw, 1), bh, occupied, rng,
                                   top_band=158, x_lo=ph_legend["x"], tidy=True)
                if fb is None:
                    fb, _ = _find_slot(W, H, max(bw, 1), bh, occupied, rng, top_band=158)
                if fb is not None:
                    lx, ly, slot = fb[0], fb[1], fb
            ph_legend["y"] = ly + bh + 12
            occupied.append(slot)
            entry["write_pos"] = (lx, ly)
            entry["text_layout"] = layout
            entry["note_card"] = (lx - 8, ly - 4, lx + bw + 8, ly + bh + 6)
            # Thin connector to the blank ONLY when its position is OCR-verified
            # (not Gemini's shifted box_2d, which would point into empty space).
            if label and placeholders and label in placeholders:
                pb = placeholders[label]
                pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                if not _arrow_crosses_text((pcx, pcy), (lx + 6, ly + bh / 2), occupied):
                    entry["arrow_params"] = (pcx, pcy, lx + 6, ly + bh / 2)

        elif action in WRITE_ACTIONS:  # stacked workspace column (derivation / Hindi)
            text = _sanitize_text(ann.get("text", ""))
            entry["text"] = text
            entry["is_hindi"] = _contains_devanagari(text)
            if entry["is_hindi"]:
                layout = _build_text_layers(text, step_hindi_font, pen, region_w, _measure_draw)
                entry["text_layout"] = layout
                bw, bh = layout["block_w"], layout["block_h"]
                clear_y, step_box = _next_clear_y(wx, wy, bw, bh, occupied, H, step_gap)
                entry["write_pos"] = (wx, clear_y)
                occupied.append(step_box)
                wy = clear_y + bh + step_gap
            else:
                entry["render_font"] = step_font  # math radical token reveal, sized to fit
                entry["line_height"] = int(step_fs * 1.45)
                try:
                    bw = min(region_w, int(_measure_draw.textlength(text, font=step_font)))
                except Exception:
                    bw = region_w
                bh = entry["line_height"]
                clear_y, step_box = _next_clear_y(wx, wy, bw, bh, occupied, H, step_gap)
                entry["write_pos"] = (wx, clear_y)
                occupied.append(step_box)
                wy = clear_y + bh + step_gap

        elif action == "draw_arrow":
            entry["write_end"] = t + 0.6
            fb = _resolve_box(ann, ocr_index, W, H, key_prefix="from_", option_positions=option_positions)
            tb = _resolve_box(ann, ocr_index, W, H, key_prefix="to_", option_positions=option_positions)
            if fb and tb:
                entry["arrow_params"] = ((fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2,
                                         (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2)

        elif action == "match_pair":  # connect a List-I item to its List-II match
            entry["write_end"] = t + 0.8
            arrow = match_routes.get(i)   # diagonal stroke across the gutter (rank-routed)
            if arrow:
                entry["match_arrow"] = arrow

        elif action in ANSWER_ACTIONS:
            target = (ann.get("target", "") or ann.get("option", ""))
            entry["write_end"] = t + 1.0
            # Extract the option letter robustly: "(A)", "Option C", "b." → A/C/B.
            opt = re.sub(r"[^A-D]", "", str(target).upper())[:1]
            # Fall back to extrapolating the marker's spot when OCR missed it (the
            # answer option, often A, is the one most likely to have been dropped).
            obox = _option_box_or_infer(opt, option_positions) if opt else None
            if obox:
                ox1, oy1, ox2, oy2 = obox
                # Ring the correct option label "(X)" (green) + a tick beside it —
                # a clean affirmative mark instead of a slash that reads like a cross.
                entry["answer_ring"] = (ox1 - 7, oy1 - 5, min(ox1 + 48, ox2 + 6), oy2 + 5)

        elif action == "verdict_mark":  # ✓/✗ beside a statement (assertion / count Qs)
            entry["write_end"] = t + 0.7
            v = str(ann.get("verdict", "")).strip().lower()
            is_true = v in ("true", "correct", "right", "yes", "सही", "1", "t")
            pos = verdict_rows.get(i)
            if pos:
                # Robust reconstructed column: guaranteed one mark per statement.
                mx, cy = int(pos[0]), pos[1]
                cand = (mx - 4, cy - 14, mx + 28, cy + 14)
                for _ in range(6):           # nudge only off an earlier ✓/✗, not text
                    if not _boxes_overlap(cand, occupied, pad=2):
                        break
                    cy += 26
                    cand = (mx - 4, cy - 14, mx + 28, cy + 14)
                entry["verdict_params"] = (mx, cy, is_true)
                occupied.append(cand)
                temp_schedule.append(entry)
                continue
            box = _resolve_verdict_box(ann.get("target"),
                                       enriched_ocr.get("elements") or [], stmt_ceiling)
            if not box:                          # non-statement target → generic resolve
                box = _resolve_box(ann, ocr_index, W, H, option_positions=option_positions,
                                   y_max=stmt_ceiling)
            if box:
                x1, y1, x2, y2 = box
                cy = (y1 + y2) / 2
                # Place the ✓/✗ just PAST the end of the statement's text on this row,
                # in the clear gap before the right-margin workspace/diagram column.
                # (The target is only the "अभिकथन A"/"कारण R" label, so we widen to the
                # full statement line via the OCR boxes on this row.) This keeps the
                # mark beside its statement and clear of both the figure and the logo.
                row_right = max([b[2] for b in occupied
                                 if b[1] - 4 < cy < b[3] + 4 and b[0] < ws_zone_x0] + [x2])
                mx = int(min(row_right + 18, ws_zone_x0 - 36))
                if mx < row_right:           # statement runs into the column: use margin
                    mx = int(min(row_right + 18, W - 46))
                cand = (mx - 4, cy - 14, mx + 28, cy + 14)
                if _boxes_overlap(cand, occupied, pad=2):
                    cy += 30  # nudge down to dodge another mark/note at this level
                    cand = (mx - 4, cy - 14, mx + 28, cy + 14)
                entry["verdict_params"] = (mx, cy, is_true)
                occupied.append(cand)

        elif action == "draw_diagram":  # build a schematic (flowchart/sequence/axis)
            dg = diagram_layouts.get(i)   # laid out + reserved in the pre-pass above
            if dg:
                entry["diagram_layout"] = dg
            entry["write_end"] = t + 5.0

        temp_schedule.append(entry)

    # Pass 2: stretch text-writing durations to fill the spoken segment.
    stretch_actions = TEXT_ACTIONS + WRITE_ACTIONS
    for i, entry in enumerate(temp_schedule):
        act = entry["action"]
        if act in stretch_actions or act == "draw_diagram":
            t_curr = entry["time"]
            t_next = temp_schedule[i + 1]["time"] if i + 1 < len(temp_schedule) else total_duration - 1.0
            cap = 14.0 if act == "draw_diagram" else 8.0   # diagrams build over a longer span
            write_dur = max(1.0, min(cap, (t_next - t_curr) * 0.9))
            entry["write_duration"] = write_dur
            entry["write_end"] = t_curr + write_dur
        schedule.append(entry)

    # Phase gate: keep cross-outs and the answer mark OUT of the reading phase, so
    # the pen never "cuts" an option while the teacher is still reading the stem.
    # The reading phase ends at the last stem underline — but CAP that at ~30% of the
    # audio: some questions underline a term again LATE (e.g. re-reading कारण R while
    # explaining it), and those late underlines must not drag the gate to mid-lecture
    # and re-clamp every verdict/note back to the end.
    underline_times = [e["time"] for e in schedule if e["action"] == "underline_existing"]
    eval_acts = ("cross_out_word", "match_pair", "annotate_word") + VERDICT_ACTIONS + ANSWER_ACTIONS
    eval_times = [e["time"] for e in schedule if e["action"] in eval_acts]
    # Reading ends at the teacher's FIRST evaluative mark; underlines that happen
    # AFTER that are re-reading mid-lecture (e.g. pointing at a term while explaining
    # it) and must NOT inflate the gate. Anchoring to an ABSOLUTE reading time (not a
    # % of total) is essential: on a long statement-count question, % of total pushed
    # the gate to ~75s and clamped the early ✓/✗ verdicts (statements judged at ~29s)
    # to mid-video, badly desyncing them from the narration.
    first_eval = min(eval_times) if eval_times else total_duration
    reading_underlines = [tt for tt in underline_times if tt <= first_eval + 0.5]
    last_reading = max(reading_underlines) if reading_underlines else 0.0
    gate = max(last_reading, min(0.16 * total_duration, 12.0))
    for e in schedule:
        if e["action"] in eval_acts and e["write_start"] < gate:
            dur = max(0.7, e["write_end"] - e["write_start"])
            e["write_start"] = gate
            e["write_end"] = gate + dur

    # If a generated timeline keeps the board mostly blank and bunches teaching
    # marks late, spread those non-answer actions through a usable teaching window.
    # The final answer mark is left for the conclusion and ordered below.
    teach_actions = [e for e in schedule
                     if e["action"] not in ("underline_existing",) + ANSWER_ACTIONS]
    if len(teach_actions) >= 3 and min(e["write_start"] for e in teach_actions) > 0.55 * total_duration:
        ordered = sorted(teach_actions, key=lambda e: e["write_start"])
        start = max(gate + 2.0, 0.32 * total_duration)
        end = min(total_duration - 3.0, 0.82 * total_duration)
        if end > start:
            span = end - start
            for k, e in enumerate(ordered):
                dur = max(0.7, e["write_end"] - e["write_start"])
                nt = start + (span * k / max(1, len(ordered) - 1))
                e["write_start"] = nt
                e["write_end"] = nt + dur

    # ── Deterministic order for option-elimination marks ──────────────────────
    # Gemini frequently FRONT-LOADS the answer + cross-outs to the very start (it
    # knows the answer immediately) and the cue-based audio sync can't always pull
    # them back, so the ✓/✗ on options pop up BEFORE the teacher evaluates them. A
    # teacher eliminates options DURING the explanation and marks the answer LAST.
    # Enforce that invariant — but only fix marks that actually violate it, so a
    # correctly-timed late answer/cross-out keeps its spoken sync.
    if option_positions:
        def _retime(e, nt):
            dur = max(0.7, e["write_end"] - e["write_start"])
            e["write_start"] = nt
            e["write_end"] = nt + dur

        def _has_cue(e):
            # A real rejection/conclusion cue (the teacher's actual words) means the sync
            # could anchor this mark to the MOMENT it is spoken; a missing/tiny cue means
            # it could only be pinned to where the option is first read/mentioned.
            return len(str(e.get("spoken_cue") or "").split()) >= 3

        # Reading runs longer on long questions (the teacher reads all four options), so
        # the gate alone under-estimates it; treat the first ~12% as reading too.
        read_end = max(gate + 0.5, 0.12 * total_duration)
        crosses = [e for e in schedule if e["action"] == "cross_out_word"]
        answers = [e for e in schedule if e["action"] in ANSWER_ACTIONS]
        if crosses or answers:
            tail = total_duration - 1.5
            # HYBRID: a cross WITH a genuine rejection cue stays at its spoken moment
            # (natural, per-option timing). A cue-less cross can only be pinned to where
            # the option is MENTIONED — a beat before the teacher finishes explaining why
            # it is wrong — so instead of striking it early, CLUSTER it in the conclusion:
            # the teacher finishes the reasoning, THEN strikes the wrong options and
            # circles the answer. Nothing is ever struck before its reasoning.
            cued = [e for e in crosses if _has_cue(e) and e["write_start"] >= read_end]
            loose = sorted((e for e in crosses if e not in cued),
                           key=lambda e: e["write_start"])
            win_start = max(read_end + 1.0, 0.55 * total_duration,
                            max((e["write_start"] for e in cued), default=0.0) + 1.0)
            win_start = min(win_start, max(read_end + 1.0, tail - 4.0))
            span = max(0.0, (tail - win_start) - 1.5)   # leave the final slot for the answer
            m = len(loose)
            for k, e in enumerate(loose):
                _retime(e, win_start + (span * k / m if m > 1 else 0.0))
            # The answer comes LAST — after every cross-out. Only move it when it is early
            # or would otherwise precede an elimination.
            cross_ts = [e["write_start"] for e in schedule if e["action"] == "cross_out_word"]
            last_cross = max(cross_ts, default=(win_start if loose else gate + 1.0))
            for e in sorted(answers, key=lambda e: e["write_start"]):
                if e["write_start"] < gate + 1.0 or (cross_ts and e["write_start"] < last_cross):
                    _retime(e, min(last_cross + 1.2, tail))
                last_cross = max(last_cross, e["write_start"])

    return schedule


# ── Frame renderer ──────────────────────────────────────────────────────────
def _render_frame_at(t, background, schedule, fonts, pen=PEN_COLOR):
    """
    Render a single frame at time t.
    All actions with write_start <= t are drawn cumulatively, in `pen` ink.
    """
    font_body = fonts[0]

    frame = Image.new("RGB", background.size, (255, 255, 255))
    frame.paste(background, (0, 0))
    draw = ImageDraw.Draw(frame, "RGBA")

    for idx, action in enumerate(schedule):
        start = action["write_start"]
        if t < start:
            continue
        action_type = action["action"]
        end = action["write_end"]
        progress = 1.0 if t >= end else (t - start) / max(end - start, 0.01)
        progress = max(0.0, min(1.0, progress))

        # Deterministic per-action jitter: the stroke shape is identical every
        # frame, so it grows smoothly instead of shaking frame-to-frame.
        random.seed(idx * 7919 + 17)

        if action_type in ("circle_word", "circle_existing"):
            p = action.get("ellipse_params")
            if p:
                _draw_progressive_ellipse(draw, p[0], p[1], p[2], p[3], progress, PEN_WIDTH, pen)

        elif action_type == "underline_existing":
            p = action.get("underline_params")
            if p:
                _draw_progressive_underline(draw, p[0], p[1], p[2], p[3], progress, PEN_WIDTH, pen)

        elif action_type == "cross_out_word":
            segs = action.get("strike_lines")
            if segs:  # one clean strike per line — covers a multi-line option fully
                for (x0, cy, x1) in segs:
                    ex = x0 + (x1 - x0) * progress
                    _draw_handwritten_line(draw, x0, cy, ex, cy, PEN_WIDTH, pen)
            elif action.get("strike_params"):       # back-compat single-row strike
                sp = action["strike_params"]
                ex = sp[0] + (sp[2] - sp[0]) * progress
                _draw_handwritten_line(draw, sp[0], sp[1], ex, sp[1], PEN_WIDTH, pen)
            else:
                p = action.get("cross_params")
                if p:
                    _draw_progressive_cross(draw, p[0], p[1], p[2], p[3], progress, PEN_WIDTH, pen)

        elif action_type in TEXT_ACTIONS:
            # Legible backing card for workspace notes (drawn fully so the text
            # is "written" onto a clean card over the faint slide watermark).
            card = action.get("note_card")
            if card:
                draw.rounded_rectangle([int(c) for c in card], radius=10,
                                       fill=(255, 255, 255, 236),
                                       outline=(208, 210, 220, 255), width=1)
            ap = action.get("arrow_params")
            if ap:
                _draw_progressive_arrow(draw, ap[0], ap[1], ap[2], ap[3],
                                        min(1.0, progress / 0.3), PEN_WIDTH, pen)
                text_progress = max(0.0, (progress - 0.3) / 0.7)
            else:
                text_progress = progress
            _paste_text_reveal(frame, action, text_progress)

        elif action_type in WRITE_ACTIONS:
            if action.get("text_layout"):
                _paste_text_reveal(frame, action, progress)
            else:
                text = action.get("text", "")
                tokens = split_into_math_tokens(text)
                k = int(progress * len(tokens))
                partial_text = "".join(tokens[:k])
                wx, wy = action["write_pos"]
                step_fnt = action.get("render_font") or font_body
                draw_math_equation_with_radicals(draw, wx, wy, partial_text, step_fnt, pen)

        elif action_type == "draw_arrow":
            p = action.get("arrow_params")
            if p:
                _draw_progressive_arrow(draw, p[0], p[1], p[2], p[3], progress, PEN_WIDTH, pen)

        elif action_type == "match_pair":
            a = action.get("match_arrow")
            if a:  # teal diagonal stroke with an arrowhead, like a teacher's match
                _draw_progressive_arrow(draw, a[0], a[1], a[2], a[3], progress,
                                        PEN_WIDTH, MATCH_INK)

        elif action_type == "draw_diagram":
            dg = action.get("diagram_layout")
            if dg:
                _render_diagram(draw, frame, dg, progress, pen)

        elif action_type == "verdict_mark":
            p = action.get("verdict_params")
            if p:
                mx, my, is_true = p
                if is_true:                               # green ✓
                    tp = progress
                    _draw_handwritten_line(draw, mx, my, mx + 9 * min(1.0, tp * 2),
                                           my + 9 * min(1.0, tp * 2), PEN_WIDTH, ANSWER_INK)
                    if tp > 0.5:
                        _draw_handwritten_line(draw, mx + 9, my + 9, mx + 24, my - 13,
                                               PEN_WIDTH, ANSWER_INK)
                else:                                     # red ✗
                    _draw_progressive_cross(draw, mx, my - 11, mx + 22, my + 11,
                                            progress, PEN_WIDTH, pen)

        elif action_type in ANSWER_ACTIONS:
            ring = action.get("answer_ring")
            if ring:
                cx, cy = (ring[0] + ring[2]) / 2, (ring[1] + ring[3]) / 2
                rx, ry = (ring[2] - ring[0]) / 2, (ring[3] - ring[1]) / 2
                _draw_progressive_ellipse(draw, cx, cy, rx, ry, progress, PEN_WIDTH, ANSWER_INK)
                if progress > 0.55:                       # affirmative tick beside the ring
                    tp = (progress - 0.55) / 0.45
                    tx, ty = ring[2] + 12, cy
                    _draw_handwritten_line(draw, tx, ty, tx + 8 * min(1.0, tp * 2),
                                           ty + 8 * min(1.0, tp * 2), PEN_WIDTH, ANSWER_INK)
                    if tp > 0.5:
                        _draw_handwritten_line(draw, tx + 8, ty + 8, tx + 22, ty - 13,
                                               PEN_WIDTH, ANSWER_INK)
            elif action.get("tick_params"):               # legacy slash fallback
                p = action["tick_params"]
                _draw_progressive_diagonal_slash(draw, p[0], p[1], p[2], p[3], progress, PEN_WIDTH, ANSWER_INK)

    return np.array(frame)


# ── Ink colour resolution ─────────────────────────────────────────────────────
_INK_COLORS = {
    "red": (200, 30, 30),
    "black": (0, 0, 0),
    "blue": (20, 40, 170),
    "green": (20, 120, 40),
}


def _resolve_ink(ink):
    """Resolve an ink name or (r,g,b) tuple to an RGB colour."""
    if isinstance(ink, (tuple, list)) and len(ink) == 3:
        return tuple(int(c) for c in ink)
    return _INK_COLORS.get(str(ink).lower().strip(), (200, 30, 30))


# ── Video assembly ──────────────────────────────────────────────────────────
def render_video(image_path, annotations_path, audio_path, output_path,
                 option_positions=None, question_bbox=None, enriched_ocr=None,
                 ink="red"):
    """
    Build the final video with teacher actions drawn directly on the image background.

    `ink` sets the annotation colour ("red" by default, matching the reference
    teacher video; also "black"/"blue"/"green" or an (r,g,b) tuple).
    """
    if option_positions is None:
        option_positions = {}
    if enriched_ocr is None:
        enriched_ocr = {}

    pen = _resolve_ink(ink)
    print(f"  Annotation ink: {ink} {pen}")

    # Load background question image
    background = Image.open(image_path).convert("RGB")

    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    # Canonicalise legacy action names from older/reused annotation files so the
    # scheduler only ever sees the one clean schema (action_schema.ALIAS_GROUPS).
    # Every alias is render-identical to its canonical name, so this changes nothing
    # visually — it just removes the dual-name handling as a source of edge bugs.
    from action_schema import normalize_actions
    annotations = normalize_actions(annotations)

    # Fonts (Ink Free for Latin/math handwriting; Kalam for Devanagari Hindi).
    font_body = _find_font("body", 28)
    font_hindi = _find_hindi_font(30)
    fonts = (font_body, font_hindi)

    # Audio details
    audio = AudioFileClip(audio_path)
    total_duration = audio.duration

    # Precompute layout coordinates and schedule
    schedule = _build_schedule(annotations, total_duration, enriched_ocr,
                               option_positions, fonts, image_size=background.size,
                               pen=pen)

    print(f"  Rendering {total_duration:.1f}s video at 24 fps...")
    
    # Frame cache key
    def _cache_key(t):
        for ann in schedule:
            if ann["write_start"] <= t < ann["write_end"]:
                return None  # active drawing
        # static: return count of completed actions
        return sum(1 for a in schedule if t >= a["write_end"])
        
    frame_cache = {}
    
    def make_frame(t):
        key = _cache_key(t)
        if key is not None and key in frame_cache:
            return frame_cache[key]
            
        frame = _render_frame_at(t, background, schedule, fonts, pen)
        if key is not None:
            frame_cache[key] = frame
        return frame

    # Create video clip
    video = VideoClip(make_frame, duration=total_duration)
    video = video.with_fps(24)

    # Fade effect
    video = video.with_effects([
        vfx.FadeIn(0.6),
        vfx.FadeOut(0.6),
    ])

    # Combine with audio
    video = video.with_audio(audio)

    video.write_videofile(
        output_path,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        logger="bar",
    )
    print(f"  Video rendering complete! Saved to {output_path}")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "input/question.png"
    ann = sys.argv[2] if len(sys.argv) > 2 else "output/annotations.json"
    aud = sys.argv[3] if len(sys.argv) > 3 else "input/narration.mp3"
    out = sys.argv[4] if len(sys.argv) > 4 else "output/final.mp4"
    render_video(img, ann, aud, out)
