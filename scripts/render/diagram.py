"""Schematic diagram engine: lay out and progressively hand-draw a node/edge
graph (flowchart / sequence / cycle / axis). Extracted from render_video.py
(Step 3)."""

import math

from render.constants import ANSWER_INK, PEN_WIDTH
from render.fonts import _sized_variant
from render.strokes import _draw_handwritten_line, _draw_progressive_arrow
from render.text_render import _render_text_layer
from render.text_utils import _sanitize_text, wrap_text_to_width


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
