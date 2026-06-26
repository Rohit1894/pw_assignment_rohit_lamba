"""Geometry & layout helpers: box overlap, segment/rect intersection, slot
finding, and OCR/Gemini target-box resolution. Extracted from
render_video.py (Step 2 refactor). Self-contained pure arithmetic."""


def _boxes_overlap(box, occupied, pad=10):
    """True if `box` intersects any box in `occupied` (with padding)."""
    x1, y1, x2, y2 = box
    for (ox1, oy1, ox2, oy2) in occupied:
        if x1 < ox2 + pad and x2 > ox1 - pad and y1 < oy2 + pad and y2 > oy1 - pad:
            return True
    return False


def _segment_hits_rect(p1, p2, rect, pad=0):
    """True if segment p1→p2 passes through axis-aligned `rect` (Liang–Barsky)."""
    x0, y0 = p1
    x1, y1 = p2
    rx0, ry0, rx1, ry1 = rect[0] - pad, rect[1] - pad, rect[2] + pad, rect[3] + pad
    dx, dy = x1 - x0, y1 - y0
    u1, u2 = 0.0, 1.0
    for p, q in ((-dx, x0 - rx0), (dx, rx1 - x0), (-dy, y0 - ry0), (dy, ry1 - y0)):
        if p == 0:
            if q < 0:
                return False        # parallel and outside this slab
        else:
            t = q / p
            if p < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)
    return u1 <= u2


def _arrow_crosses_text(p1, p2, occupied):
    """True if a note→word arrow would cut across an unrelated text box. Boxes
    touching either endpoint (the word itself, the note slot) are ignored."""
    for b in occupied:
        # skip boxes that contain an endpoint (the anchored word / the note slot)
        if (b[0] - 2 <= p1[0] <= b[2] + 2 and b[1] - 2 <= p1[1] <= b[3] + 2) or \
           (b[0] - 2 <= p2[0] <= b[2] + 2 and b[1] - 2 <= p2[1] <= b[3] + 2):
            continue
        if _segment_hits_rect(p1, p2, b, pad=-3):
            return True
    return False


def _underline_for_box(box, occupied, W, H):
    """Return a short underline that stays in the gap before the next text line."""
    x1, y1, x2, y2 = box
    h = max(1, y2 - y1)
    target_y = y2 + 4
    limit_y = H - 4
    for ob in occupied:
        ox1, oy1, ox2, oy2 = ob
        if abs(ox1 - x1) < 2 and abs(oy1 - y1) < 2 and abs(ox2 - x2) < 2 and abs(oy2 - y2) < 2:
            continue
        horizontal_overlap = ox1 < x2 - 8 and ox2 > x1 + 8
        lower_line = oy1 > y1 + 2
        if horizontal_overlap and lower_line:
            # EasyOCR boxes often overlap between adjacent printed lines. When
            # that happens, use an estimated visible top of the next line rather
            # than the raw box top, or underlines get pulled through the word.
            next_top = oy1
            if oy1 < y2 + 6:
                next_top = oy1 + 0.28 * max(1, oy2 - oy1)
            limit_y = min(limit_y, next_top - 3)
    y = min(target_y, limit_y)
    min_y = y1 + 0.55 * h
    if y < min_y and limit_y >= min_y - 2:
        y = min_y
    y = max(y1 + 0.42 * h, min(y, H - 4))
    return (x1, int(y), x2, int(y))


def _next_clear_y(x, y, w, h, occupied, H, gap=8):
    """Move a new written line down until it no longer overlaps placed content."""
    y = int(y)
    w = max(1, int(w))
    h = max(1, int(h))
    while y + h < H - 8:
        cand = (x - 8, y - 4, x + w + 8, y + h + 6)
        if not _boxes_overlap(cand, occupied, pad=4):
            return y, cand
        blockers = [
            ob for ob in occupied
            if ob[0] < cand[2] + 4 and ob[2] > cand[0] - 4
            and ob[1] < cand[3] + 4 and ob[3] > cand[1] - 4
        ]
        if blockers:
            y = max(y + gap, max(int(ob[3] + gap) for ob in blockers))
        else:
            y += gap
    cand = (x - 8, y - 4, x + w + 8, min(H - 2, y + h + 6))
    return y, cand


def _find_slot(W, H, w, h, occupied, rng, top_band, anchor=None, x_lo=16, tidy=False):
    """
    Find a top-left position for a (w x h) block in empty space.

    If `anchor` (a word box) is given, first try to place the block directly
    below it (in-place). Otherwise — or if that area is occupied — scan a grid
    over the empty area so notes don't overlap any printed text or placed note.

    `x_lo` restricts the horizontal scan to start at that x (used to keep free
    workspace notes in the clear right-margin column). `tidy` stacks candidates
    column-major (top-to-bottom, left column first) so notes form a neat
    workspace instead of a random scatter.

    Returns (box, needs_arrow). `needs_arrow` is True when an anchored note had
    to be placed away from its word and should be connected with an arrow.
    """
    if anchor:
        ax1, ay1, ax2, ay2 = anchor
        # Small pad here so the meaning can hug the word (sit in the gap just
        # below it) without being rejected for being near its own line.
        for (bx, by) in ((ax1, ay2 + 6), (ax1 - 12, ay2 + 8), (ax2 - w, ay2 + 6)):
            bx = max(8, min(int(bx), W - w - 8))
            box = (bx, by, bx + w, by + h)
            if by + h < H and not _boxes_overlap(box, occupied, pad=3):
                return box, False  # placed in-place, no arrow needed

    step = 26
    xs = list(range(int(x_lo), max(int(x_lo) + 1, W - w - 16), step))
    ys = list(range(top_band, max(top_band + 1, H - h - 14), step))
    if tidy:
        grid = [(x, y) for x in xs for y in ys]   # column-major → tidy top-down stack
    else:
        grid = [(x, y) for y in ys for x in xs]
        rng.shuffle(grid)
    for (x, y) in grid:
        box = (x, y, x + w, y + h)
        if not _boxes_overlap(box, occupied):
            return box, (anchor is not None)
    return None, (anchor is not None)


# ── Target box resolution (OCR text → box, else Gemini coords) ──────────────
def _snap_box_to_ocr(box, ocr_index, W, H):
    """
    Snap an approximate Gemini `box_2d` onto the printed label it is pointing at.

    Gemini's vision coordinates are roughly right but often off by tens of pixels.
    When the box overlaps (or sits very close to) an OCR'd text element, we return
    that element's exact bounds so circles/underlines/arrows land precisely on the
    diagram label. If nothing is near (e.g. the box points at blank space), the
    original box is kept — so this never drags an annotation onto an unrelated word.
    """
    if not box or not ocr_index or not getattr(ocr_index, "elements", None):
        return box
    x1, y1, x2, y2 = box
    bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
    best_overlap, best_el = 0.0, None
    nearest_d, nearest_el = None, None
    for el in ocr_index.elements:
        ox = max(0, min(x2, el.x2) - max(x1, el.x1))
        oy = max(0, min(y2, el.y2) - max(y1, el.y1))
        ov = ox * oy
        if ov > best_overlap:
            best_overlap, best_el = ov, el
        d = ((bcx - el.center_x) ** 2 + (bcy - el.center_y) ** 2) ** 0.5
        if nearest_d is None or d < nearest_d:
            nearest_d, nearest_el = d, el
    if best_el is not None:                       # overlapping label → refine to it
        return (best_el.x1, best_el.y1, best_el.x2, best_el.y2)
    # Near-miss: snap only if within ~one label-size of the target centre.
    box_span = max(x2 - x1, y2 - y1, 0.02 * H)
    if nearest_el is not None and nearest_d <= 1.2 * box_span:
        return (nearest_el.x1, nearest_el.y1, nearest_el.x2, nearest_el.y2)
    return box


def _resolve_box(ann, ocr_index, W, H, key_prefix="", option_positions=None, y_max=None):
    """
    Resolve an action's target to a pixel box (x1, y1, x2, y2).

    Priority: exact OCR text match (precise) → Gemini box_2d (normalised
    ymin,xmin,ymax,xmax 0-1000) → box_norm (x1,y1,x2,y2 0-1000) → box (pixels).
    `key_prefix` lets draw_arrow read 'from_'/'to_' variants.
    `y_max` bounds a verdict's `prefer_lowest` search to above the options.
    """
    target = None
    b2d = None
    bn = None
    bp = None

    if key_prefix == "from_":
        target = ann.get("from_target") or ann.get("start_target") or ann.get("from") or ann.get("start")
        b2d = ann.get("from_box_2d") or ann.get("start_box_2d")
        bn = ann.get("from_box_norm") or ann.get("start_box_norm")
        bp = ann.get("from_box") or ann.get("start_box")
    elif key_prefix == "to_":
        target = ann.get("to_target") or ann.get("end_target") or ann.get("to") or ann.get("end")
        b2d = ann.get("to_box_2d") or ann.get("end_box_2d")
        bn = ann.get("to_box_norm") or ann.get("end_box_norm")
        bp = ann.get("to_box") or ann.get("end_box")

    if not target:
        target = ann.get(key_prefix + "target") or (ann.get("target") if not key_prefix else None)
    if not b2d:
        b2d = ann.get(key_prefix + "box_2d") or ann.get("box_2d")
    if not bn:
        bn = ann.get(key_prefix + "box_norm") or ann.get("box_norm")
    if not bp:
        bp = ann.get(key_prefix + "box") or ann.get("box")

    action_type = ann.get("action", "")
    if target and action_type == "cross_out_word" and option_positions:
        clean_opt = str(target).strip().upper().strip("()[]{}. ")
        if len(clean_opt) == 1 and clean_opt in option_positions:
            bbox = option_positions[clean_opt]
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            # Cross just the option marker "(X)" at the row's left, not the whole row.
            return (min(xs) - 2, min(ys), min(xs) + 50, max(ys))

    if target and ocr_index:
        # Robust: tolerates OCR misreads and phrases split across boxes. For a
        # verdict, prefer the lower occurrence so the ✓/✗ lands on the STATEMENT,
        # not the same label where it appears in the question stem.
        box = ocr_index.find_phrase_box(target, threshold=0.5,
                                        prefer_lowest=(action_type == "verdict_mark"),
                                        y_max=y_max if action_type == "verdict_mark" else None)
        if box:
            return tuple(box)

    # Approximate Gemini coordinates → snap onto the printed label they point at.
    box = None
    if b2d and len(b2d) == 4:
        ymin, xmin, ymax, xmax = b2d
        box = (xmin / 1000 * W, ymin / 1000 * H, xmax / 1000 * W, ymax / 1000 * H)
    elif bn and len(bn) == 4:
        box = (bn[0] / 1000 * W, bn[1] / 1000 * H, bn[2] / 1000 * W, bn[3] / 1000 * H)
    elif bp and len(bp) == 4:
        box = tuple(bp)
    if box is not None:
        return _snap_box_to_ocr(box, ocr_index, W, H)
    return None


def _coords_box(ann, W, H):
    """Pixel box from explicit Gemini coords only (box_2d / box_norm / box)."""
    b2d = ann.get("box_2d")
    if b2d and len(b2d) == 4:
        ymin, xmin, ymax, xmax = b2d
        return (xmin / 1000 * W, ymin / 1000 * H, xmax / 1000 * W, ymax / 1000 * H)
    bn = ann.get("box_norm")
    if bn and len(bn) == 4:
        return (bn[0] / 1000 * W, bn[1] / 1000 * H, bn[2] / 1000 * W, bn[3] / 1000 * H)
    bp = ann.get("box")
    if bp and len(bp) == 4:
        return tuple(bp)
    return None
