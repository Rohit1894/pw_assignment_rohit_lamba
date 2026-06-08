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

# ── Colour palette ──────────────────────────────────────────────────────────
PEN_COLOR = (0, 0, 0)                             # black pen style
PEN_WIDTH = 3                                     # marker width


# ── Font helper ─────────────────────────────────────────────────────────────
def _find_font(family="body", size=26):
    """Locate a handwriting-style or standard TrueType font on the system."""
    if family == "title":
        candidates = [
            "C:/Windows/Fonts/Inkfree.ttf",
            "C:/Windows/Fonts/segoeprb.ttf",
            "C:/Windows/Fonts/segoescb.ttf",
            "C:/Windows/Fonts/comicbd.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/System/Library/Fonts/Supplemental/ChalkboardSE.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:  # body / handwriting
        candidates = [
            "C:/Windows/Fonts/Inkfree.ttf",
            "C:/Windows/Fonts/segoepr.ttf",
            "C:/Windows/Fonts/segoesc.ttf",
            "C:/Windows/Fonts/comic.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Supplemental/ChalkboardSE.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

    for path in candidates:
        if os.path.exists(path):
            try:
                # Inkfree requires slightly larger size to match same visual weight
                font_size = size + 4 if "Inkfree.ttf" in path else size
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    return ImageFont.load_default(size=size)


# ── Tokenizer for math equations (Word-wise reveal) ────────────────────────
def split_into_math_tokens(text):
    """
    Split math equation into logical tokens (words, symbols, operators).
    Groups letters/numbers and subscripts together, separating math operators.
    """
    token_pattern = r'[A-Za-z0-9₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹]+|\s+|[^\w\s]'
    return re.findall(token_pattern, text)


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


# ── Handwriting-style drawing ───────────────────────────────────────────────
def _draw_handwritten_line(draw, x1, y1, x2, y2, width=PEN_WIDTH, color=PEN_COLOR):
    """Draw a slightly jittery line to simulate handwriting."""
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx**2 + dy**2)
    
    # Determine number of steps
    steps = max(int(dist / 4), 1)
    
    for i in range(steps + 1):
        t = i / steps
        px = x1 + dx * t
        py = y1 + dy * t
        
        # Add slight jitter
        px += random.uniform(-0.6, 0.6)
        py += random.uniform(-0.6, 0.6)
        
        r = width / 2
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color)


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


def draw_custom_text(draw, x, y, text, font, color):
    """Draw text character-by-character, mapping subscripts/superscripts/minus signs."""
    curr_x = x
    try:
        font_path = getattr(font, "path", None)
        if font_path and os.path.exists(font_path):
            sub_font = ImageFont.truetype(font_path, max(10, int(font.size * 0.65)))
        else:
            sub_font = font
    except Exception:
        sub_font = font

    SUPERSCRIPTS_MAP = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
    }
    
    for char in text:
        char_to_draw = char
        curr_font = font
        curr_y = y
        
        if char == '−': # Unicode minus
            char_to_draw = '-'
        elif '₀' <= char <= '₉':
            char_to_draw = str(ord(char) - 0x2080)
            curr_font = sub_font
            curr_y = y + int(font.size * 0.25)
        elif char in SUPERSCRIPTS_MAP:
            char_to_draw = SUPERSCRIPTS_MAP[char]
            curr_font = sub_font
            curr_y = y - int(font.size * 0.15)
            
        draw.text((curr_x, curr_y), char_to_draw, fill=color, font=curr_font)
        curr_x += draw.textlength(char_to_draw, font=curr_font)
        
    return curr_x - x


def get_custom_text_width(draw, text, font):
    """Calculate the width of the text using custom sub-font/subscript mapping."""
    curr_x = 0
    try:
        font_path = getattr(font, "path", None)
        if font_path and os.path.exists(font_path):
            sub_font = ImageFont.truetype(font_path, max(10, int(font.size * 0.65)))
        else:
            sub_font = font
    except Exception:
        sub_font = font

    SUPERSCRIPTS_MAP = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
    }
    
    for char in text:
        char_to_draw = char
        curr_font = font
        
        if char == '−': # Unicode minus
            char_to_draw = '-'
        elif '₀' <= char <= '₉':
            char_to_draw = str(ord(char) - 0x2080)
            curr_font = sub_font
        elif char in SUPERSCRIPTS_MAP:
            char_to_draw = SUPERSCRIPTS_MAP[char]
            curr_font = sub_font
            
        curr_x += draw.textlength(char_to_draw, font=curr_font)
        
    return curr_x


def draw_math_equation_with_radicals(draw, x, y, text, font, color):
    """
    Draw a math equation, rendering square root '√' symbols as real
    handwritten radical lines instead of drawing a missing font glyph box.
    """
    if "√" not in text:
        draw_custom_text(draw, x, y, text, font, color)
        return
        
    parts = text.split("√")
    curr_x = x
    
    for idx, part in enumerate(parts):
        if idx == 0:
            # Plain text before the first radical
            if part:
                curr_x += draw_custom_text(draw, curr_x, y, part, font, color)
        else:
            # This part is inside a radical
            # Find the parenthesis block if present
            if part.startswith("("):
                depth = 0
                closing_idx = -1
                for char_idx, char in enumerate(part):
                    if char == "(":
                        depth += 1
                    elif char == ")":
                        depth -= 1
                        if depth == 0:
                            closing_idx = char_idx
                            break
                if closing_idx != -1:
                    inside = part[1:closing_idx]
                    rest = part[closing_idx+1:]
                else:
                    inside = part[1:]
                    rest = ""
            else:
                # If no parenthesis, take digits/letters as inside, rest as rest
                match = re.match(r'^[0-9]+', part)
                if match:
                    inside = match.group(0)
                    rest = part[len(inside):]
                else:
                    inside = part
                    rest = ""
                    
            # Draw handwritten radical sign around the inside text
            inside_w = get_custom_text_width(draw, inside, font) if inside else 0
            
            # Draw radical symbol:
            # Tail starts at y + 15
            r_width = 2
            rx0 = curr_x
            ry0 = y + 15
            
            rx1 = curr_x + 6
            ry1 = y + 19
            
            rx2 = curr_x + 14
            ry2 = y + 30
            
            rx3 = curr_x + 22
            ry3 = y - 4
            
            rx4 = curr_x + 22 + int(inside_w) + 2
            ry4 = y - 4
            
            # Draw the radical strokes as a proper solid line
            draw.line([(rx0, ry0), (rx1, ry1), (rx2, ry2), (rx3, ry3), (rx4, ry4)], fill=color, width=r_width, joint="round")
            
            # Draw inside text inside the radical (shifted right of the sign)
            if inside:
                draw_custom_text(draw, curr_x + 24, y, inside, font, color)
                curr_x += 24 + inside_w + 6
                
            # Draw rest text
            if rest:
                curr_x += draw_custom_text(draw, curr_x, y, rest, font, color)


# ── Animation schedule ──────────────────────────────────────────────────────
def _build_schedule(annotations, total_duration, enriched_ocr, option_positions):
    """
    Pre-compute geometry parameters and layouts for all annotations.
    Arranges written equations in the largest empty space region.
    """
    schedule = []
    
    ocr_index = enriched_ocr.get("index") if enriched_ocr else None
    free_spaces = enriched_ocr.get("free_spaces", []) if enriched_ocr else []
    
    # Determine best empty space region
    if free_spaces:
        best_space = free_spaces[0]
        rx1, ry1, rx2, ry2 = best_space["bounds"]
        print(f"  Writing layout region selected: {best_space['position']} bounds: {best_space['bounds']}")
    else:
        rx1, ry1, rx2, ry2 = 680, 100, 1240, 680
        print(f"  No free spaces detected. Using default right side fallback bounds: [{rx1}, {ry1}, {rx2}, {ry2}]")
        
    wx = rx1 + 25
    wy = ry1 + 30
    
    temp_schedule = []
    for i, ann in enumerate(annotations):
        action = ann["action"]
        t = ann["time"]
        
        entry = {
            **ann,
            "write_start": t,
        }
        
        if action == "circle_existing":
            target = ann.get("target", "")
            entry["circle_duration"] = 0.8
            entry["write_end"] = t + 0.8
            
            if ocr_index:
                matches = ocr_index.find_by_text(target, threshold=0.5)
                if matches:
                    elem = matches[0]
                    sub_x1, sub_y1, sub_x2, sub_y2 = get_substring_bounds(elem, target)
                    cx = (sub_x1 + sub_x2) // 2
                    cy = (sub_y1 + sub_y2) // 2
                    entry["circle_params"] = (cx, cy, (sub_x2 - sub_x1) // 2 + 8)
                else:
                    if "(1, 2)" in target or "1, 2" in target:
                        entry["circle_params"] = (145, 91, 30)
                    elif "(4, 6)" in target or "4, 6" in target:
                        entry["circle_params"] = (255, 91, 30)
            else:
                entry["circle_params"] = (145, 91, 30)
                
        elif action == "underline_existing":
            target = ann.get("target", "")
            entry["underline_duration"] = 0.8
            entry["write_end"] = t + 0.8
            
            if ocr_index:
                matches = ocr_index.find_by_text(target, threshold=0.4)
                if not matches:
                    matches = ocr_index.find_by_text("distance", threshold=0.4)
                    
                if matches:
                    elem = matches[0]
                    sub_x1, sub_y1, sub_x2, sub_y2 = get_substring_bounds(elem, target)
                    entry["underline_params"] = (sub_x1, sub_y2 + 4, sub_x2, sub_y2 + 4)
                else:
                    if "1, 2" in target or "(1, 2)" in target:
                        entry["underline_params"] = (572, 117, 690, 117)
                    elif "4, 6" in target or "(4, 6)" in target:
                        entry["underline_params"] = (763, 117, 851, 117)
                    else:
                        entry["underline_params"] = (60, 117, 240, 117)
            else:
                if "1, 2" in target or "(1, 2)" in target:
                    entry["underline_params"] = (572, 117, 690, 117)
                elif "4, 6" in target or "(4, 6)" in target:
                    entry["underline_params"] = (763, 117, 851, 117)
                else:
                    entry["underline_params"] = (60, 117, 240, 117)
                    
        elif action == "write_equation":
            entry["write_pos"] = (wx, wy)
            wy += 60
            
        elif action == "draw_arrow":
            entry["arrow_duration"] = 0.6
            entry["write_end"] = t + 0.6
            
        elif action == "tick_answer":
            target = ann.get("target", "") or ann.get("option", "")
            entry["tick_duration"] = 0.5
            entry["write_end"] = t + 0.5
            
            opt_letter = target.replace("Option", "").strip().upper()
            if opt_letter in option_positions:
                bbox = option_positions[opt_letter]
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                ox1, oy1 = min(xs), min(ys)
                ox2, oy2 = max(xs), max(ys)
                
                # Diagonal line crossing through the option indicator (e.g. (C))
                # It starts slightly bottom-left and ends slightly top-right
                x_start = ox1 - 6
                y_start = oy2 + 6
                x_end = ox1 + 45
                y_end = oy1 - 6
                entry["tick_params"] = (x_start, y_start, x_end, y_end)
            else:
                entry["tick_params"] = (18, 300, 69, 240)
                
        temp_schedule.append(entry)
        
    # Pass 2: Calculate duration stretch and arrow coordinates
    for i, entry in enumerate(temp_schedule):
        if entry["action"] == "write_equation":
            t_curr = entry["time"]
            t_next = total_duration - 1.0
            
            if i + 1 < len(temp_schedule):
                t_next = temp_schedule[i + 1]["time"]
                
            # Stretch equation writing duration so it matches verbal explanation (max 10s)
            segment_dur = t_next - t_curr
            write_dur = max(0.8, min(10.0, segment_dur * 0.85))
            entry["write_duration"] = write_dur
            entry["write_end"] = t_curr + write_dur
            
        elif entry["action"] == "draw_arrow":
            prev_eq = None
            for j in range(i - 1, -1, -1):
                if temp_schedule[j]["action"] == "write_equation":
                    prev_eq = temp_schedule[j]
                    break
            
            next_eq = None
            for j in range(i + 1, len(temp_schedule)):
                if temp_schedule[j]["action"] == "write_equation":
                    next_eq = temp_schedule[j]
                    break
                    
            if prev_eq and next_eq:
                px, py = prev_eq["write_pos"]
                nx, ny = next_eq["write_pos"]
                entry["arrow_params"] = (px + 60, py + 35, nx + 60, ny - 12)
            else:
                entry["arrow_params"] = (wx + 60, wy - 80, wx + 60, wy - 30)
                
        schedule.append(entry)
        
    return schedule


# ── Frame renderer ──────────────────────────────────────────────────────────
def _render_frame_at(t, background, schedule, fonts):
    """
    Render a single frame at time t.
    All actions with write_start <= t are drawn cumulatively.
    """
    font_body = fonts[0]
    
    # Create white canvas copy of question image
    frame = Image.new("RGB", background.size, (255, 255, 255))
    frame.paste(background, (0, 0))
    
    draw = ImageDraw.Draw(frame, "RGBA")
    
    # Render all actions up to time t
    for action in schedule:
        start = action["write_start"]
        if t < start:
            continue
            
        action_type = action["action"]
        end = action["write_end"]
        duration = action.get(f"{action_type.split('_')[0]}_duration", 0.8)
        
        # Calculate progress [0.0, 1.0]
        progress = 1.0 if t >= end else (t - start) / max(duration, 0.01)
        progress = max(0.0, min(1.0, progress))
        
        if action_type == "circle_existing":
            params = action.get("circle_params")
            if params:
                cx, cy, r = params
                _draw_progressive_circle(draw, cx, cy, r, progress, PEN_WIDTH, PEN_COLOR)
                
        elif action_type == "underline_existing":
            params = action.get("underline_params")
            if params:
                x1, y1, x2, y2 = params
                _draw_progressive_underline(draw, x1, y1, x2, y2, progress, PEN_WIDTH, PEN_COLOR)
                
        elif action_type == "write_equation":
            text = action.get("text", "")
            wx, wy = action["write_pos"]
            
            # Math word-wise tokenized reveal
            tokens = split_into_math_tokens(text)
            n_tokens = len(tokens)
            k = int(progress * n_tokens)
            partial_text = "".join(tokens[:k])
            
            # Draw math equation with custom radical rendering to avoid missing glyph boxes
            draw_math_equation_with_radicals(draw, wx, wy, partial_text, font_body, PEN_COLOR)
            
        elif action_type == "draw_arrow":
            params = action.get("arrow_params")
            if params:
                x1, y1, x2, y2 = params
                _draw_progressive_arrow(draw, x1, y1, x2, y2, progress, PEN_WIDTH, PEN_COLOR)
                
        elif action_type == "tick_answer":
            params = action.get("tick_params")
            if params:
                x1, y1, x2, y2 = params
                _draw_progressive_diagonal_slash(draw, x1, y1, x2, y2, progress, PEN_WIDTH, PEN_COLOR)
                
    return np.array(frame)


# ── Video assembly ──────────────────────────────────────────────────────────
def render_video(image_path, annotations_path, audio_path, output_path,
                 option_positions=None, question_bbox=None, enriched_ocr=None):
    """
    Build the final video with teacher actions drawn directly on the image background.
    """
    if option_positions is None:
        option_positions = {}
    if enriched_ocr is None:
        enriched_ocr = {}

    # Load background question image
    background = Image.open(image_path).convert("RGB")
    
    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
        
    # Fonts (Ink Free size 28 is perfect for natural handwriting)
    font_body = _find_font("body", 28)
    fonts = (font_body,)

    # Audio details
    audio = AudioFileClip(audio_path)
    total_duration = audio.duration

    # Precompute layout coordinates and schedule
    schedule = _build_schedule(annotations, total_duration, enriched_ocr, option_positions)

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
            
        frame = _render_frame_at(t, background, schedule, fonts)
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
