"""Hand-drawn progressive pen primitives: jittery lines, underlines, circles,
arrows, diagonal slashes, ellipses and cross-outs. Extracted from
render_video.py (Step 1 refactor)."""

import math
import random

import numpy as np

from render.constants import PEN_COLOR, PEN_WIDTH


# ── Handwriting-style drawing ───────────────────────────────────────────────
def _draw_handwritten_line(draw, x1, y1, x2, y2, width=PEN_WIDTH, color=PEN_COLOR):
    """
    Draw a continuous, slightly jittery SOLID line to simulate handwriting.

    Earlier this stamped spaced dots which read as a dotted/beaded line; we now
    draw a connected polyline so underlines, slashes and arrows are solid.
    """
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx**2 + dy**2)

    # A point every ~6px is enough for a smooth solid stroke.
    steps = max(int(dist / 6), 1)

    points = []
    for i in range(steps + 1):
        t = i / steps
        px = x1 + dx * t + random.uniform(-0.5, 0.5)
        py = y1 + dy * t + random.uniform(-0.5, 0.5)
        points.append((px, py))

    if len(points) >= 2:
        draw.line(points, fill=color, width=max(1, int(width)), joint="round")
        # Round the end caps so the stroke looks like a marker tip.
        r = width / 2
        for (px, py) in (points[0], points[-1]):
            draw.ellipse([px - r, py - r, px + r, py + r], fill=color)


def _draw_progressive_polyline(draw, pts, progress, width=PEN_WIDTH, color=PEN_COLOR,
                               end_dots=True):
    """Draw a multi-segment (elbow) connector progressively along its arc length.

    Used for matching-table connectors: a straight diagonal would slice through
    the intervening cell text, so the line is an L/Z elbow routed through the
    blank gutter between columns. The pen grows from the first point to the last
    at constant speed regardless of how many bends the path has.
    """
    if not pts or len(pts) < 2:
        return
    seg_len = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
               for i in range(len(pts) - 1)]
    total = sum(seg_len) or 1.0
    target = total * max(0.0, min(1.0, progress))
    if end_dots:                                  # small anchor dot at the start cell
        r = width / 2 + 1
        draw.ellipse([pts[0][0] - r, pts[0][1] - r, pts[0][0] + r, pts[0][1] + r], fill=color)
    run = 0.0
    for i in range(len(pts) - 1):
        if run >= target:
            break
        (x1, y1), (x2, y2) = pts[i], pts[i + 1]
        if run + seg_len[i] <= target:
            _draw_handwritten_line(draw, x1, y1, x2, y2, width, color)
        else:                                     # partial segment = the growing tip
            f = (target - run) / (seg_len[i] or 1.0)
            _draw_handwritten_line(draw, x1, y1, x1 + (x2 - x1) * f, y1 + (y2 - y1) * f,
                                   width, color)
        run += seg_len[i]
    if progress >= 0.999 and end_dots:            # landing dot on the target cell
        r = width / 2 + 1
        draw.ellipse([pts[-1][0] - r, pts[-1][1] - r, pts[-1][0] + r, pts[-1][1] + r], fill=color)


def _draw_progressive_underline(draw, x1, y1, x2, y2, progress, width=PEN_WIDTH, color=PEN_COLOR):
    """Draw underline progressively."""
    end_x = x1 + (x2 - x1) * progress
    _draw_handwritten_line(draw, x1, y1, end_x, y2, width, color)


def _draw_progressive_circle(draw, cx, cy, radius, progress, width=PEN_WIDTH, color=PEN_COLOR):
    """Draw circle progressively (from 0 to 360 degrees)."""
    steps = max(int(progress * 60), 2)
    angles = np.linspace(0, 2 * np.pi * progress, steps)
    for i in range(len(angles) - 1):
        x1 = cx + radius * math.cos(angles[i])
        y1 = cy + radius * math.sin(angles[i])
        x2 = cx + radius * math.cos(angles[i+1])
        y2 = cy + radius * math.sin(angles[i+1])
        _draw_handwritten_line(draw, x1, y1, x2, y2, width, color)


def _draw_progressive_arrow(draw, x1, y1, x2, y2, progress, width=PEN_WIDTH, color=PEN_COLOR):
    """Draw arrow shaft and head progressively."""
    end_x = x1 + (x2 - x1) * progress
    end_y = y1 + (y2 - y1) * progress
    _draw_handwritten_line(draw, x1, y1, end_x, end_y, width, color)
    
    if progress > 0.8:
        # Draw arrowhead pointing towards (x2, y2)
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            dx /= length
            dy /= length
            
            # Size of arrowhead
            arrow_len = 12
            arrow_width = 6
            
            # Back along shaft
            bx = end_x - dx * arrow_len
            by = end_y - dy * arrow_len
            
            # Left and right points
            p1x = bx + dy * arrow_width
            p1y = by - dx * arrow_width
            p2x = bx - dy * arrow_width
            p2y = by + dx * arrow_width
            
            _draw_handwritten_line(draw, end_x, end_y, p1x, p1y, width, color)
            _draw_handwritten_line(draw, end_x, end_y, p2x, p2y, width, color)


def _draw_progressive_diagonal_slash(draw, x1, y1, x2, y2, progress, width=PEN_WIDTH, color=PEN_COLOR):
    """Draw a diagonal slash line progressively to cross out/mark the option."""
    end_x = x1 + (x2 - x1) * progress
    end_y = y1 + (y2 - y1) * progress
    _draw_handwritten_line(draw, x1, y1, end_x, end_y, width, color)


# ── Hand-drawn primitives: ellipse + cross-out ──────────────────────────────
def _draw_progressive_ellipse(draw, cx, cy, rx, ry, progress, width=PEN_WIDTH,
                              color=PEN_COLOR):
    """Draw a hand-drawn ellipse, swept progressively, with radius noise."""
    sweep = 2 * math.pi * 1.08  # slight over-closure, like a real circling stroke
    end = progress * sweep
    n = max(6, int(progress * 72))
    start_ang = -math.pi * 0.55
    pts = []
    for i in range(n + 1):
        a = start_ang + end * (i / n)
        jr = 1.0 + random.uniform(-0.045, 0.045)
        pts.append((cx + rx * jr * math.cos(a), cy + ry * jr * math.sin(a)))
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=max(1, int(width)), joint="round")


def _draw_progressive_cross(draw, x1, y1, x2, y2, progress, width=PEN_WIDTH,
                            color=PEN_COLOR):
    """Cross out a box: a slash first, then a second stroke forms an X."""
    # First diagonal: bottom-left -> top-right over [0, 0.6].
    p1 = min(1.0, progress / 0.6)
    _draw_handwritten_line(draw, x1, y2, x1 + (x2 - x1) * p1, y2 - (y2 - y1) * p1,
                           width, color)
    # Second diagonal: top-left -> bottom-right over [0.6, 1.0].
    if progress > 0.6:
        p2 = (progress - 0.6) / 0.4
        _draw_handwritten_line(draw, x1, y1, x1 + (x2 - x1) * p2, y1 + (y2 - y1) * p2,
                               width, color)
