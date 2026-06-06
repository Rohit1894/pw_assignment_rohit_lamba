#!/usr/bin/env python3
"""
Render the final annotated video with writing animation.

Approach:
  1. Load the question image and extend the canvas downward to create a
     dedicated "solution workspace" (dark area below the question).
  2. Pre-compute an animation schedule: each annotation gets a writing window
     where its text appears character-by-character.
  3. Use a single VideoClip(make_frame) that renders every frame dynamically,
     drawing partially-revealed text, a glowing pen cursor, and highlights.
  4. Cache static frames (when no writing is active) to avoid redundant rendering.
  5. Add fade-in/fade-out and compose with audio.

Annotations PERSIST after they appear — they do not fade out.
The correct option gets a green highlight rectangle drawn around it.
The question text area gets a semi-transparent yellow highlight.
"""

import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy import VideoClip, AudioFileClip, CompositeVideoClip, vfx


# ── Colour palette ──────────────────────────────────────────────────────────

WORKSPACE_BG = (22, 22, 34)                    # dark blue-grey
TITLE_COLOR = (255, 215, 0)                     # gold
STEP_LABEL_COLOR = (120, 180, 255)              # soft blue
TEXT_COLOR = (240, 240, 240)                     # off-white
ANSWER_COLOR = (80, 255, 130)                    # green
HIGHLIGHT_RECT_COLOR = (80, 255, 130)            # green border
QUESTION_HIGHLIGHT_COLOR = (255, 255, 80, 45)   # semi-transparent yellow
CORRECT_HIGHLIGHT_COLOR = (80, 255, 130, 50)     # semi-transparent green
PEN_COLOR = (255, 200, 60)                       # warm yellow glow
PEN_GLOW_COLOR = (255, 220, 100, 120)            # outer glow

WORKSPACE_HEIGHT = 420                           # pixels added below the question
CHARS_PER_SEC = 14                               # writing speed
FADE_DURATION = 0.6                              # seconds for video fade in/out


# ── Font helpers ────────────────────────────────────────────────────────────

def _find_font(family="body", size=30):
    """Locate a handwriting-style TrueType font on the system."""
    if family == "title":
        candidates = [
            # macOS handwriting / chalk fonts
            "/System/Library/Fonts/Supplemental/Chalkduster.ttf",
            "/Library/Fonts/Chalkduster.ttf",
            "/System/Library/Fonts/Supplemental/Noteworthy.ttc",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            # Windows
            "C:/Windows/Fonts/comicbd.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]
    else:  # body / handwriting
        candidates = [
            # macOS
            "/System/Library/Fonts/Supplemental/ChalkboardSE.ttc",
            "/Library/Fonts/ChalkboardSE.ttc",
            "/System/Library/Fonts/Supplemental/Noteworthy.ttc",
            "/System/Library/Fonts/Supplemental/Chalkduster.ttf",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            # Windows
            "C:/Windows/Fonts/comic.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]

    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default(size=size)


# ── Animation schedule ──────────────────────────────────────────────────────

def _build_schedule(annotations, total_duration):
    """
    Pre-compute writing windows for each annotation.

    Each annotation gets:
      - write_start: when characters begin appearing
      - write_end:   when the last character is revealed
      - total_chars: length of the text

    Simultaneous annotations are sequenced so they don't overlap.
    """
    schedule = []
    last_write_end = 0.0

    for ann in annotations:
        t = ann["time"]
        text = ann.get("text", "")
        n_chars = len(text) if text else 0
        write_duration = n_chars / CHARS_PER_SEC if n_chars > 0 else 0.3

        # If this annotation would start before the previous one finished
        # writing, push it to start right after
        write_start = max(t, last_write_end + 0.1)

        # Don't let writing extend past the video
        write_end = min(write_start + write_duration, total_duration - 0.1)

        entry = {
            **ann,
            "write_start": write_start,
            "write_end": write_end,
            "total_chars": n_chars,
        }
        schedule.append(entry)
        last_write_end = write_end

    return schedule


# ── Frame renderer ──────────────────────────────────────────────────────────

def _chars_visible(ann, t):
    """How many characters of this annotation's text are visible at time t."""
    if t >= ann["write_end"]:
        return ann["total_chars"]
    if t < ann["write_start"]:
        return 0
    progress = (t - ann["write_start"]) / max(ann["write_end"] - ann["write_start"], 0.01)
    return int(progress * ann["total_chars"])


def _draw_pen_cursor(draw, x, y, font_size):
    """Draw a glowing pen dot at the current writing position."""
    r_outer = max(int(font_size * 0.3), 5)
    r_inner = max(int(font_size * 0.15), 3)

    # Outer glow
    draw.ellipse(
        [x - r_outer, y - r_outer, x + r_outer, y + r_outer],
        fill=PEN_GLOW_COLOR,
    )
    # Inner dot
    draw.ellipse(
        [x - r_inner, y - r_inner, x + r_inner, y + r_inner],
        fill=PEN_COLOR + (255,),
    )


def _draw_question_highlight(draw, question_bbox, canvas_w):
    """Draw a semi-transparent yellow wash over the question region."""
    if not question_bbox:
        return
    x1, y1, x2, y2 = question_bbox
    # Add some padding
    x1 = max(0, x1 - 8)
    y1 = max(0, y1 - 8)
    x2 = min(canvas_w, x2 + 8)
    y2 = y2 + 8
    draw.rectangle([x1, y1, x2, y2], fill=QUESTION_HIGHLIGHT_COLOR)


def _draw_option_highlight(draw, option_positions, option_letter):
    """Draw a green highlight rectangle around the correct option."""
    if option_letter not in option_positions:
        return
    pts = option_positions[option_letter]
    x1 = int(min(p[0] for p in pts)) - 8
    y1 = int(min(p[1] for p in pts)) - 8
    x2 = int(max(p[0] for p in pts)) + 8
    y2 = int(max(p[1] for p in pts)) + 8

    # Semi-transparent green fill
    draw.rectangle([x1, y1, x2, y2], fill=CORRECT_HIGHLIGHT_COLOR)
    # Solid green border
    draw.rectangle([x1, y1, x2, y2], outline=HIGHLIGHT_RECT_COLOR, width=3)
    # Checkmark
    draw.text((x2 + 10, y1), "\u2714", fill=ANSWER_COLOR + (255,))


def _render_frame_at(t, background, canvas_size, schedule, option_positions,
                     question_bbox, fonts):
    """
    Render a single frame at time `t`.

    Returns an (H, W, 3) numpy array.
    """
    font_body, font_title, font_small = fonts
    frame = Image.new("RGBA", canvas_size, WORKSPACE_BG + (255,))

    # Paste the original question at the top
    frame.paste(background.convert("RGBA"), (0, 0))

    draw = ImageDraw.Draw(frame)
    orig_h = background.size[1]
    canvas_w = canvas_size[0]

    # ── Question highlight (appears with first annotation) ───────────────
    if schedule and t >= schedule[0]["write_start"] and question_bbox:
        _draw_question_highlight(draw, question_bbox, canvas_w)

    # ── Workspace divider line ───────────────────────────────────────────
    draw.line(
        [(20, orig_h + 5), (canvas_w - 20, orig_h + 5)],
        fill=(60, 60, 90, 200), width=2,
    )

    # ── Annotations ──────────────────────────────────────────────────────
    y_cursor = orig_h + 22
    active_pen = None  # will hold (x, y) of pen if writing is happening

    for ann in schedule:
        n_visible = _chars_visible(ann, t)
        if n_visible == 0:
            continue

        action = ann["action"]
        full_text = ann.get("text", "")
        partial_text = full_text[:n_visible]
        is_writing = n_visible < ann["total_chars"]

        if action == "highlight_question":
            draw.text((30, y_cursor), partial_text,
                      fill=TITLE_COLOR + (255,), font=font_title)
            bbox = draw.textbbox((30, y_cursor), partial_text, font=font_title)
            if not is_writing:
                # Underline when fully written
                draw.line(
                    [(30, bbox[3] + 4), (bbox[2], bbox[3] + 4)],
                    fill=TITLE_COLOR + (140,), width=2,
                )
            if is_writing:
                active_pen = (bbox[2] + 2, (bbox[1] + bbox[3]) // 2)
            y_cursor = bbox[3] + 18

        elif action == "write":
            draw.text((40, y_cursor), partial_text,
                      fill=TEXT_COLOR + (255,), font=font_body)
            bbox = draw.textbbox((40, y_cursor), partial_text, font=font_body)
            if is_writing:
                active_pen = (bbox[2] + 2, (bbox[1] + bbox[3]) // 2)
            y_cursor = bbox[3] + 14

        elif action == "highlight_option":
            label = f"Answer: {partial_text}"
            draw.text((40, y_cursor), label,
                      fill=ANSWER_COLOR + (255,), font=font_title)
            bbox = draw.textbbox((40, y_cursor), label, font=font_title)
            if is_writing:
                active_pen = (bbox[2] + 2, (bbox[1] + bbox[3]) // 2)
            y_cursor = bbox[3] + 14

            # Draw green highlight on the correct option in the image
            if not is_writing:
                option_letter = full_text.replace("Option ", "").strip().upper()
                _draw_option_highlight(draw, option_positions, option_letter)

        elif action == "highlight_region":
            # Highlight a region of the question image (no text to write)
            # The region info is embedded in the annotation
            pass

    # ── Pen cursor ───────────────────────────────────────────────────────
    if active_pen:
        _draw_pen_cursor(draw, active_pen[0], active_pen[1], 28)

    return np.array(frame.convert("RGB"))


# ── Video assembly ──────────────────────────────────────────────────────────

def render_video(image_path, annotations_path, audio_path, output_path,
                 option_positions=None, question_bbox=None):
    """
    Build the final annotated video with writing animation.

    1. Extends the question image canvas with a workspace below.
    2. Pre-computes an animation schedule for character-by-character writing.
    3. Renders each frame dynamically using make_frame.
    4. Adds audio, fade-in, and fade-out.
    """
    if option_positions is None:
        option_positions = {}

    # Load resources
    background = Image.open(image_path).convert("RGB")
    orig_w, orig_h = background.size
    canvas_size = (orig_w, orig_h + WORKSPACE_HEIGHT)

    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    annotations.sort(key=lambda x: x["time"])

    # Fonts
    font_body = _find_font("body", 28)
    font_title = _find_font("title", 32)
    font_small = _find_font("body", 22)
    fonts = (font_body, font_title, font_small)

    # Audio duration determines video length
    audio = AudioFileClip(audio_path)
    total_duration = audio.duration

    # Build animation schedule
    schedule = _build_schedule(annotations, total_duration)

    print(f"  Animation schedule: {len(schedule)} annotations over {total_duration:.1f}s")
    for s in schedule:
        print(f"    [{s['write_start']:.1f}s-{s['write_end']:.1f}s] "
              f"{s['action']}: {s['text'][:50]}...")

    # ── Frame cache for static periods ───────────────────────────────────
    # Between writing windows, the frame doesn't change — cache it.
    frame_cache = {}
    fps = 24

    def _cache_key(t):
        """
        Determine which 'state' the video is in at time t.
        If we're between writing windows, the display is static and
        depends only on how many annotations are fully visible.
        During writing, every frame is unique (return None = no cache).
        """
        for ann in schedule:
            if ann["write_start"] <= t < ann["write_end"]:
                return None  # actively writing — unique frame
        # Static: key is the count of fully-visible annotations
        n_visible = sum(1 for a in schedule if t >= a["write_end"])
        return n_visible

    def make_frame(t):
        key = _cache_key(t)
        if key is not None and key in frame_cache:
            return frame_cache[key]

        frame = _render_frame_at(
            t, background, canvas_size, schedule,
            option_positions, question_bbox, fonts,
        )

        if key is not None:
            frame_cache[key] = frame
        return frame

    # Build video clip
    video = VideoClip(make_frame, duration=total_duration)
    video = video.with_fps(fps)

    # Add fade in and fade out
    video = video.with_effects([
        vfx.FadeIn(FADE_DURATION),
        vfx.FadeOut(FADE_DURATION),
    ])

    # Attach audio
    video = video.with_audio(audio)

    print(f"  Rendering {total_duration:.1f}s video at {fps} fps...")
    video.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        logger="bar",
    )
    print(f"  Video saved -> {output_path}")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "input/question.png"
    ann = sys.argv[2] if len(sys.argv) > 2 else "output/annotations.json"
    aud = sys.argv[3] if len(sys.argv) > 3 else "input/narration.mp3"
    out = sys.argv[4] if len(sys.argv) > 4 else "output/final.mp4"
    render_video(img, ann, aud, out)
