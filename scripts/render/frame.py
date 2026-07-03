"""Frame rendering and final video assembly.

Draws one fully-annotated frame at time t (_render_frame_at), resolves the ink
colour, and assembles the MoviePy clip with audio (render_video). This is the
render entry point; render_video.py re-exports render_video() from here.
Extracted from render_video.py (Step 5 refactor)."""

import json
import math
import random

import numpy as np
from PIL import Image, ImageDraw
from moviepy import VideoClip, AudioFileClip, vfx

from render.constants import (
    PEN_COLOR, PEN_WIDTH, ANSWER_INK, MATCH_INK,
    WRITE_ACTIONS, ANSWER_ACTIONS, TEXT_ACTIONS,
)
from render.text_utils import split_into_math_tokens
from render.text_render import _paste_text_reveal, draw_math_equation_with_radicals
from render.strokes import (
    _draw_handwritten_line, _draw_progressive_underline, _draw_progressive_arrow,
    _draw_progressive_diagonal_slash, _draw_progressive_ellipse, _draw_progressive_cross,
)
from render.diagram import _render_diagram
from render.fonts import _find_font, _find_hindi_font
from render.schedule import _build_schedule


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


# ── Idle "alive" cue during long teaching pauses (Fix C) ─────────────────────
# When the timeline has a long static hold — the teacher explaining a result
# verbally before writing the next line — the board would otherwise sit on a
# frozen frame. We draw a subtle, pulsing cursor where the pen last rested, so the
# pause reads as "paused, thinking" rather than "the video froze". It adds NO
# content (nothing that could be wrong) and is GATED to long holds, so normally
# paced videos render byte-identical. The pulse is a pure function of t (sine, no
# RNG), preserving frame determinism.
IDLE_HOLD_MIN = 22.0      # only fill holds longer than this — matches schedule.py's
                          # "comfortable hold" floor, so the regular ~20 s step cadence
                          # stays cleanly static and only a genuinely long pause is marked
IDLE_EDGE_FADE = 2.5      # ease the cue in/out at the hold's two ends (seconds)
IDLE_PERIOD = 1.15        # blink period (seconds)


def _action_rest_point(action):
    """Where the pen comes to rest after finishing `action` — the end of its last
    stroke. Used to park the idle cursor on the most recently written content."""
    lay = action.get("text_layout")
    wp = action.get("write_pos")
    if lay and wp:
        return (wp[0] + lay.get("block_w", 0), wp[1] + lay.get("block_h", 0))
    if wp:
        return (wp[0] + 40, wp[1] + 18)
    for key in ("underline_params", "cross_params", "ellipse_params",
                "answer_ring", "arrow_params"):
        p = action.get(key)
        if p and len(p) >= 4:
            return (p[2], p[3])
    return None


def _draw_idle_cue(draw, schedule, t, pen):
    """Draw the subtle resting-cursor during a long mid-solution hold (see above)."""
    last_end, last_act, next_start = None, None, None
    for a in schedule:
        s, e = a["write_start"], a["write_end"]
        if s <= t < e:
            return                      # something is being drawn — that IS the motion
        if e <= t:
            if last_end is None or e > last_end:
                last_end, last_act = e, a
        elif s > t:
            if next_start is None or s < next_start:
                next_start = s
    if last_act is None or next_start is None:
        return                          # lead-in or final tail: nothing to rest on
    if next_start - last_end < IDLE_HOLD_MIN:
        return                          # a normal pause — leave it static (byte-identical)
    into, remain = t - last_end, next_start - t
    edge = 1.0
    if into < IDLE_EDGE_FADE or remain < IDLE_EDGE_FADE:
        edge = max(0.0, min(into, remain) / IDLE_EDGE_FADE)
    pulse = 0.5 + 0.5 * math.sin(2 * math.pi * into / IDLE_PERIOD)
    alpha = int((55 + 90 * pulse) * edge)       # subtle ~55..145, fades fully in/out (no pop)
    if alpha <= 6:
        return
    pt = _action_rest_point(last_act)
    if not pt:
        return
    x, y = int(pt[0]) + 6, int(pt[1])
    col = (pen[0], pen[1], pen[2], alpha)
    draw.line([(x, y - 19), (x, y - 2)], fill=col, width=max(2, PEN_WIDTH - 1))


def _draw_answer_box(draw, box, progress):
    """Trace a hand-drawn rectangle around the final answer line, side by side
    (top → right → bottom → left) as `progress` runs 0..1. Green answer ink."""
    x1, y1, x2, y2 = box
    sides = (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
             ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1)))
    total = max(0.0, min(1.0, progress)) * 4
    for i, (a, b) in enumerate(sides):
        p = max(0.0, min(1.0, total - i))
        if p <= 0:
            break
        ex = a[0] + (b[0] - a[0]) * p
        ey = a[1] + (b[1] - a[1]) * p
        _draw_handwritten_line(draw, a[0], a[1], ex, ey, PEN_WIDTH, ANSWER_INK)


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
        # Page turn: once the next storyboard page starts writing, the previous
        # page's worked-solution writing is wiped off the board (the question
        # background remains). Skipping the draw reveals the now-blank zone.
        wipe_after = action.get("_wipe_after")
        if wipe_after is not None and t >= wipe_after:
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
            # Green hand-drawn box around the final-answer line (auto-audio
            # whiteboard mode), traced after the text itself has been written.
            ab = action.get("answer_box")
            if ab and progress >= 0.82:
                _draw_answer_box(draw, ab, min(1.0, (progress - 0.82) / 0.18))

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

    _draw_idle_cue(draw, schedule, t, pen)
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
                 ink="red", layout_path=None, mode=None):
    """
    Build the final video with teacher actions drawn directly on the image background.

    `ink` sets the annotation colour ("red" by default, matching the reference
    teacher video; also "black"/"blue"/"green" or an (r,g,b) tuple).

    Whiteboard-storyboard mode (auto-audio pipeline): pass the composed canvas
    as `image_path`, `mode="whiteboard_storyboard"` and `layout_path` pointing
    at layout.json — the solution zone from the layout then constrains where
    all step writing goes, keeping the question (left) and solution (right)
    areas cleanly separated. Classic image+audio mode is unchanged.
    """
    if option_positions is None:
        option_positions = {}
    if enriched_ocr is None:
        enriched_ocr = {}

    if mode == "whiteboard_storyboard" and layout_path:
        try:
            with open(layout_path, encoding="utf-8") as f:
                _layout = json.load(f)
            zone = _layout.get("solution_zone")
            if zone:
                enriched_ocr = dict(enriched_ocr)
                enriched_ocr["workspace_zone"] = zone
        except Exception as e:
            print(f"  layout load failed ({e}); using OCR-derived workspace")

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
