"""Matching-question support: cue extraction, List-I->List-II connector
routing, matching-table bounds, and option-box inference. Extracted from
render_video.py (Step 3)."""

import re

from render.geometry import _resolve_box


_HI_STOP = {"की", "के", "का", "में", "से", "और", "है", "हैं", "को", "पर", "कि",
            "यह", "वह", "एक", "होता", "होती", "द्वारा", "लिए", "तथा", "या"}


def _target_cue(target):
    """A compact key-term cue drawn from a note's target word, e.g. for a lone
    annotate_word ("अंडोत्सर्ग") that can't be linked with an arrow we prefix
    "एलएच तीव्र → अंडोत्सर्ग" so it reads on its own. Picks the first content word
    (two when the first is a short acronym like एलएच / FSH)."""
    toks = [w for w in re.split(r"\s+", str(target or "").strip()) if w]
    content = [w for w in toks if w not in _HI_STOP] or toks
    if not content:
        return ""
    pick = content[:2] if len(content[0]) <= 4 and len(content) > 1 else content[:1]
    return " ".join(pick)


def _route_match_pairs(annotations, ocr_index, option_positions, W, H):
    """Plan clean diagonal connectors (सूची-I cell → सूची-II cell) for matching.

    Like a teacher's matching strokes: a straight arrow from each left item,
    crossing the empty gutter between the columns, landing on its correct right
    item. The crossings happen in blank space, never over cell text.

    The hard part is the TARGET ROW. The सूची-II column is frequently unreadable
    by OCR, and Gemini's `to_box_2d` row is unreliable (mirrored or shifted by a
    row). BUT the RELATIVE ORDER of the to_box y-values is correct, so we rank the
    right-hand boxes to recover each item's true row index, then place the arrow
    on the matching TABLE row — rows align across a matching table, so the left
    items' own y-centres give the row positions.

    Returns {annotation_index: (x1, y1, x2, y2)} — a straight arrow per pair.
    """
    midx = [(i, ann) for i, ann in enumerate(annotations)
            if ann.get("action") == "match_pair"]
    if not midx:
        return {}
    N = len(midx)
    elements = list(getattr(ocr_index, "elements", []) or [])

    # Options ceiling — the table sits above the answer options.
    opt_top = H
    if option_positions:
        try:
            opt_top = min(min(p[1] for p in v) for v in option_positions.values())
        except Exception:
            opt_top = H

    # Reconstruct the table cells from OCR. Gemini's box_2d is in a shifted frame
    # that doesn't match the pixels, so it is used ONLY for relative row order;
    # all pixel geometry comes from OCR. Keep just the narrow cell items in the
    # table band (drop the wide instruction/option/year lines).
    _SKIP = ("नीचे", "विकल्प", "चुनिए", "उत्तर", "मिलान", "कूट", "सही", "निम्न", "सूची")
    items = []
    for el in elements:
        b = (el.x1, el.y1, el.x2, el.y2)
        cy, cx = (b[1] + b[3]) / 2, (b[0] + b[2]) / 2
        txt = (el.text or "").strip()
        if not (140 < cy < opt_top - 6) or (b[2] - b[0]) > 160:
            continue
        if any(k in txt for k in _SKIP) or re.search(r"[\[\]()0-9]", txt):
            continue
        items.append((cx, cy, b))

    def _ykey(ann, pre):
        b = ann.get(pre + "box_2d") or ann.get(
            ("end_" if pre == "to_" else "start_") + "box_2d")
        if b and len(b) == 4:
            return b[0]
        rb = _resolve_box(ann, ocr_index, W, H, key_prefix=pre,
                          option_positions=option_positions)
        return rb[1] if rb else 0

    from_rank = {k: r for r, k in
                 enumerate(sorted(range(N), key=lambda k: _ykey(midx[k][1], "from_")))}
    to_rank = {k: r for r, k in
               enumerate(sorted(range(N), key=lambda k: _ykey(midx[k][1], "to_")))}

    routes = {}
    route_table_b = _matching_table_bounds(annotations, elements, option_positions, W, H)
    if not route_table_b:
        table_boxes = []
        for el in elements:
            b = getattr(el, "bounds", None)
            if not b and all(hasattr(el, k) for k in ("x1", "y1", "x2", "y2")):
                b = (el.x1, el.y1, el.x2, el.y2)
            if not b:
                continue
            txt = getattr(el, "text", "") or ""
            x1, y1, x2, y2 = b
            cy = (y1 + y2) / 2
            if not (70 <= cy <= opt_top - 70 and x1 < W * 0.45 and x2 < W * 0.50):
                continue
            if (x2 - x1) > 260 or re.search(r"\d{4}|\([A-D]\)", txt):
                continue
            table_boxes.append((x1, y1, x2, y2))
        if len(table_boxes) >= 4:
            route_table_b = (
                max(0, min(b[0] for b in table_boxes) - 34),
                max(0, min(b[1] for b in table_boxes) - 28),
                min(W, max(b[2] for b in table_boxes) + 34),
                min(H, max(b[3] for b in table_boxes) + 34),
            )
    if route_table_b:
        tx1, ty1, tx2, ty2 = route_table_b
        tw = tx2 - tx1
        th = ty2 - ty1

        # Best case: OCR can resolve the actual List-I and List-II text boxes.
        # Draw from the source word's right edge to the target word's left edge,
        # so the student reads "this item maps to that item" instead of seeing
        # arrows that appear to start from the roman-numeral column.
        direct_routes = {}
        for i, ann in midx:
            fb = _resolve_box(ann, ocr_index, W, H, key_prefix="from_",
                              option_positions=option_positions)
            tb = _resolve_box(ann, ocr_index, W, H, key_prefix="to_",
                              option_positions=option_positions)
            if not (fb and tb):
                continue
            fcx, fcy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
            tcx, tcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
            if not (ty1 <= fcy <= ty2 and ty1 <= tcy <= ty2):
                continue
            if not (tx1 <= fcx <= tx1 + tw * 0.58 and tx1 + tw * 0.46 <= tcx <= tx2):
                continue
            sx = min(fb[2] + 8, tx1 + tw * 0.52)
            ex = max(tb[0] - 8, tx1 + tw * 0.62)
            if ex - sx >= 40:
                direct_routes[i] = (sx, fcy, ex, tcy)
        if len(direct_routes) == N:
            return direct_routes

        left_right_x = tx1 + tw * 0.32
        right_left_x = tx1 + tw * 0.72
        body_top = ty1 + th * 0.35
        body_bottom = ty2 - th * 0.08
        row_y = [body_top + (body_bottom - body_top) * k / (N - 1) for k in range(N)] if N > 1 else [body_top]
        for k, (i, ann) in enumerate(midx):
            ly = row_y[min(from_rank[k], N - 1)]
            ry = row_y[min(to_rank[k], N - 1)]
            routes[i] = (left_right_x, ly, right_left_x, ry)
        return routes
    if len(items) >= 2:
        # Column split = the widest horizontal gap between cell centres.
        xs = sorted(c[0] for c in items)
        split = max(((xs[j + 1] - xs[j], (xs[j + 1] + xs[j]) / 2)
                     for j in range(len(xs) - 1)), default=(0, W / 2))[1]
        left = [it for it in items if it[0] < split]
        right = [it for it in items if it[0] >= split]
        left_right_x = max((it[2][2] for it in left), default=split - 40)
        right_left_x = min((it[2][0] for it in right), default=split + 40)
        # Pixel rows: cluster the OCR'd cell y's, then space N rows over their span
        # (table rows are evenly pitched, so this fills rows OCR happened to miss).
        clustered = []
        for y in sorted(it[1] for it in items):
            if not clustered or y - clustered[-1] > 26:
                clustered.append(y)
        top_y, bot_y = clustered[0], clustered[-1]
        row_y = [top_y + (bot_y - top_y) * k / (N - 1) for k in range(N)] if N > 1 else [top_y]
        for k, (i, ann) in enumerate(midx):
            ly = row_y[min(from_rank[k], N - 1)]
            ry = row_y[min(to_rank[k], N - 1)]
            routes[i] = (left_right_x + 8, ly, right_left_x - 8, ry)
        return routes

    # Fallback (table not reconstructable): straight arrow from resolved boxes.
    for i, ann in midx:
        fb = _resolve_box(ann, ocr_index, W, H, key_prefix="from_", option_positions=option_positions)
        tb = _resolve_box(ann, ocr_index, W, H, key_prefix="to_", option_positions=option_positions)
        if fb and tb:
            sx = fb[2] if fb[0] <= tb[0] else fb[0]
            ex = tb[0] if fb[0] <= tb[0] else tb[2]
            routes[i] = (sx + 4, (fb[1] + fb[3]) / 2, ex - 4, (tb[1] + tb[3]) / 2)
    return routes


def _is_matching_timeline(annotations):
    """True when the action set is for a List-I/List-II matching question."""
    return any(a.get("action") == "match_pair" for a in annotations)


def _matching_table_bounds(annotations, elements, option_positions, W, H):
    """
    Reconstruct a conservative protected box around the printed matching table.

    OCR sees the text inside table cells, not the table borders. Notes can
    otherwise fit in the white gaps between OCR words and cover the question.
    For matching timelines, reserve the full table area as a no-write zone while
    still allowing circles and match strokes to target the table contents.
    """
    if not _is_matching_timeline(annotations):
        return None
    opt_top = H
    if option_positions:
        try:
            opt_top = min(min(p[1] for p in v) for v in option_positions.values())
        except Exception:
            opt_top = H
    boxes = []
    for el in elements or []:
        b = el.get("bounds") if isinstance(el, dict) else getattr(el, "bounds", None)
        txt = (el.get("text") if isinstance(el, dict) else getattr(el, "text", "")) or ""
        if not b and all(hasattr(el, k) for k in ("x1", "y1", "x2", "y2")):
            b = (el.x1, el.y1, el.x2, el.y2)
        if not b:
            continue
        x1, y1, x2, y2 = b
        cy = (y1 + y2) / 2
        # Matching tables sit above the answer choices and usually occupy the
        # left half of the slide. Include header/row text; exclude the PW logo and
        # answer-option rows.
        if (70 <= cy <= opt_top - 70 and x1 < W * 0.58 and x2 < W * 0.50 and
                (x2 - x1) <= 260 and not re.search(r"\d{4}", txt)):
            boxes.append((x1, y1, x2, y2))
    if len(boxes) < 4:
        return None
    x1 = max(0, min(b[0] for b in boxes) - 28)
    y1 = max(0, min(b[1] for b in boxes) - 28)
    x2 = min(W, max(b[2] for b in boxes) + 34)
    y2 = min(H, max(b[3] for b in boxes) + 34)
    # Avoid swallowing the answer choices if OCR placed a stray option-like token
    # high in the page.
    y2 = min(y2, opt_top - 12)
    if x2 - x1 < 220 or y2 - y1 < 120:
        return None
    return (x1, y1, x2, y2)


def _option_box_or_infer(opt, option_positions):
    """Return (x1,y1,x2,y2) for an option letter, extrapolating from the detected
    options when OCR missed THAT marker.

    EasyOCR frequently drops a single option marker (e.g. reads "(A)" as noise),
    and on matching/assertion questions the missing one is often the very option
    that is the answer — leaving the green ring with nothing to circle. Options
    are evenly spaced, so a linear fit (ordinal → centre) on the markers that WERE
    found recovers the missing position.
    """
    def _box(letter):
        pts = option_positions[letter]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    if opt in option_positions:
        return _box(opt)
    known = sorted(k for k in option_positions if len(k) == 1 and k.isalpha())
    if len(known) < 2:
        return None
    ords = [ord(k) - ord("A") for k in known]
    boxes = [_box(k) for k in known]
    ws = sorted(b[2] - b[0] for b in boxes)
    hs = sorted(b[3] - b[1] for b in boxes)
    mw, mh = ws[len(ws) // 2], hs[len(hs) // 2]
    o = ord(opt) - ord("A")

    def _lin(vals):                               # least-squares centre vs ordinal
        n = len(ords)
        sx, sy = sum(ords), sum(vals)
        sxx = sum(v * v for v in ords)
        sxy = sum(a * b for a, b in zip(ords, vals))
        denom = n * sxx - sx * sx
        if denom == 0:
            return vals[0]
        a = (n * sxy - sx * sy) / denom
        return a * o + (sy - a * sx) / n

    cx = _lin([(b[0] + b[2]) / 2 for b in boxes])
    cy = _lin([(b[1] + b[3]) / 2 for b in boxes])
    return (cx - mw / 2, cy - mh / 2, cx + mw / 2, cy + mh / 2)
