"""Diagram/flowchart placeholder inference: map labels to indices, fit the
blank grid, and resolve a fill box. Extracted from render_video.py (Step 3)."""

import re

from render.geometry import _coords_box


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
