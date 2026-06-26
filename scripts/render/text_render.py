"""Text rendering: per-glyph draw with sub/superscript + symbol-font fallback,
hand-drawn radical math, and the crop-reveal text-layer pipeline (perfect
Devanagari shaping). Extracted from render_video.py (Step 2 refactor)."""

import re

from PIL import Image, ImageDraw

from render.constants import _SUPERSCRIPT_MAP, _SUBSCRIPT_MAP
from render.fonts import _resolve_glyph_font, _sized_sub_font
from render.text_utils import _contains_devanagari, wrap_text_to_width


def draw_custom_text(draw, x, y, text, font, color):
    """Draw text character-by-character: map sub/superscripts (digits, signs AND
    letters) to a smaller shifted base glyph, and fall back to a symbol font for
    any glyph the handwriting font lacks (Greek, ∫ ∑ ≤ ± × ÷ ∞ →, …)."""
    curr_x = x
    sub_font = _sized_sub_font(font)

    for char in text:
        char_to_draw = char
        curr_font = font
        curr_y = y

        if char == '−':  # Unicode minus
            char_to_draw = '-'
        elif char in _SUBSCRIPT_MAP:
            char_to_draw = _SUBSCRIPT_MAP[char]
            curr_font = sub_font
            curr_y = y + int(font.size * 0.25)
        elif char in _SUPERSCRIPT_MAP:
            char_to_draw = _SUPERSCRIPT_MAP[char]
            curr_font = sub_font
            curr_y = y - int(font.size * 0.15)

        draw_font = _resolve_glyph_font(char_to_draw, curr_font)
        draw.text((curr_x, curr_y), char_to_draw, fill=color, font=draw_font)
        curr_x += draw.textlength(char_to_draw, font=draw_font)

    return curr_x - x


def get_custom_text_width(draw, text, font):
    """Width of text using the same sub/superscript + glyph-fallback logic as
    draw_custom_text (kept in lock-step so layout matches what is drawn)."""
    curr_x = 0
    sub_font = _sized_sub_font(font)

    for char in text:
        char_to_draw = char
        curr_font = font

        if char == '−':  # Unicode minus
            char_to_draw = '-'
        elif char in _SUBSCRIPT_MAP:
            char_to_draw = _SUBSCRIPT_MAP[char]
            curr_font = sub_font
        elif char in _SUPERSCRIPT_MAP:
            char_to_draw = _SUPERSCRIPT_MAP[char]
            curr_font = sub_font

        draw_font = _resolve_glyph_font(char_to_draw, curr_font)
        curr_x += draw.textlength(char_to_draw, font=draw_font)

    return curr_x


# Start of a stacked-fraction command: \frac{ , \dfrac{ or \tfrac{
_FRAC_RE = re.compile(r"\\(?:d|t)?frac\s*\{")


def _read_braced(s, pos):
    """`s[pos]` is '{'. Return (inner_text, index just AFTER the matching '}')."""
    depth = 0
    for i in range(pos, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[pos + 1:i], i + 1
    return s[pos + 1:], len(s)            # unbalanced -> take the rest


def _draw_fraction(draw, x, y, num, den, font, color):
    """Draw `num` over `den` with a horizontal bar (a stacked fraction), centred
    vertically on the line so it sits amid neighbouring inline text. Numerator and
    denominator are drawn inline (sub/superscripts + glyph fallback; a √ inside a
    fraction renders as a symbol glyph, not the hand-drawn spanning radical).
    Returns the right-edge x."""
    pad = 8
    num_w = get_custom_text_width(draw, num, font)
    den_w = get_custom_text_width(draw, den, font)
    w = max(num_w, den_w)
    fs = font.size
    bar_y = y + int(fs * 0.55)                        # near the inline text's centre
    num_y = bar_y - int(fs * 0.85)
    den_y = bar_y + int(fs * 0.08)
    draw_custom_text(draw, x + pad / 2 + (w - num_w) / 2, num_y, num, font, color)
    draw_custom_text(draw, x + pad / 2 + (w - den_w) / 2, den_y, den, font, color)
    draw.line([(x, bar_y), (x + w + pad, bar_y)], fill=color, width=2)
    return x + w + pad + 4


def draw_math_equation_with_radicals(draw, x, y, text, font, color):
    """Draw a math equation: hand-drawn '√' radicals, smaller shifted sub/super-
    scripts, and STACKED fractions written as ``\\frac{num}{den}`` (also
    ``\\dfrac`` / ``\\tfrac``). Returns the total drawn width.

    Text with NO ``\\frac`` is rendered by the unchanged inline path
    (``_draw_radicals_inline``), so every existing equation renders exactly as
    before; only ``\\frac`` spans use the new stacking.
    """
    if "\\frac" not in text:
        return _draw_radicals_inline(draw, x, y, text, font, color)
    curr_x, i, seg_start, n = x, 0, 0, len(text)
    while i < n:
        m = _FRAC_RE.match(text, i)
        if m:
            num, j = _read_braced(text, m.end() - 1)
            if j < n and text[j] == "{":
                den, k = _read_braced(text, j)
                inline = text[seg_start:i]      # flush text before the fraction
                if inline:
                    curr_x += _draw_radicals_inline(draw, curr_x, y, inline, font, color)
                curr_x = _draw_fraction(draw, curr_x, y, num, den, font, color)
                i = seg_start = k
                continue
            # malformed \frac (no second group) -> leave it in the inline run
        i += 1
    inline = text[seg_start:]
    if inline:
        curr_x += _draw_radicals_inline(draw, curr_x, y, inline, font, color)
    return curr_x - x


def _draw_radicals_inline(draw, x, y, text, font, color):
    """Inline equation drawing — the original draw_math_equation_with_radicals
    body, now RETURNING its drawn width (the drawing itself is unchanged). Handles
    hand-drawn '√' radicals + sub/superscripts; does NOT handle \\frac (the caller
    splits those out)."""
    if "√" not in text:
        return draw_custom_text(draw, x, y, text, font, color)

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
    return curr_x - x


def _measure_block(draw, text, font, max_w):
    """Wrap `text` to `max_w` and return (lines, width, height, line_height)."""
    lines = wrap_text_to_width(draw, text, font, max_w)
    width = int(max((draw.textlength(ln, font=font) for ln in lines), default=0))
    line_h = int(font.size * 1.4)
    return lines, width, line_h * max(1, len(lines)), line_h


# ── Crop-reveal text rendering (perfect Devanagari shaping) ─────────────────
def _layout_text_lines(text, font, max_w, measure_draw):
    """Split on explicit newlines, then word-wrap each part to max_w."""
    lines = []
    for part in str(text).split("\n"):
        part = part.strip()
        if not part:
            continue
        lines.extend(wrap_text_to_width(measure_draw, part, font, max_w))
    return lines or [str(text)]


def _render_text_layer(text, font, color):
    """
    Render `text` once to its own transparent RGBA layer (correct shaping).

    Returns (layer, width, height). Revealing a left-crop of this layer keeps
    Hindi matras/conjuncts perfectly shaped — we never re-shape a partial
    string, we just uncover more of an already-correct image.

    Devanagari is drawn as one whole string (shaping is mandatory). Latin/math
    text is drawn glyph-by-glyph so any symbol the handwriting font lacks (Greek,
    operators, arrows) falls back to a symbol font instead of an empty box.
    """
    tmp = Image.new("RGBA", (4, 4))
    d = ImageDraw.Draw(tmp)
    try:
        l, t, r, b = d.textbbox((0, 0), text or " ", font=font)
    except Exception:
        l, t, r, b = 0, 0, int(d.textlength(text, font=font)), font.size
    pad = 4
    h = max(1, (b - t)) + pad * 2

    if _contains_devanagari(text):
        w = max(1, (r - l)) + pad * 2
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((pad - l, pad - t), text, font=font, fill=color)
        return layer, w, h

    # Latin/math: measure and draw per glyph with symbol-font fallback.
    total_w = 0
    for ch in (text or ""):
        total_w += d.textlength(ch, font=_resolve_glyph_font(ch, font))
    w = max(1, int(total_w)) + pad * 2
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    cx = pad - l
    for ch in (text or ""):
        gf = _resolve_glyph_font(ch, font)
        ld.text((cx, pad - t), ch, font=gf, fill=color)
        cx += d.textlength(ch, font=gf)
    return layer, w, h


def _build_text_layers(text, font, color, max_w, measure_draw):
    """Pre-render every wrapped line to a layer. Returns layout dict."""
    line_strs = _layout_text_lines(text, font, max_w, measure_draw)
    layers = [_render_text_layer(ln, font, color) for ln in line_strs]
    line_h = int(font.size * 1.5)
    block_w = max((w for _, w, _ in layers), default=1)
    block_h = line_h * len(layers)
    total_w = sum(w for _, w, _ in layers) or 1
    return {
        "layers": layers, "line_height": line_h,
        "block_w": block_w, "block_h": block_h, "total_w": total_w,
    }


def _paste_text_reveal(frame, action, progress):
    """Paste the pre-rendered text layers, revealed left-to-right, line by line."""
    layout = action.get("text_layout")
    if not layout:
        return
    wx, wy = action["write_pos"]
    line_h = layout["line_height"]
    reveal_px = progress * layout["total_w"]
    for i, (layer, w, h) in enumerate(layout["layers"]):
        if reveal_px <= 0:
            break
        show = int(min(w, reveal_px))
        if show > 0:
            crop = layer.crop((0, 0, show, h))
            frame.paste(crop, (int(wx), int(wy + i * line_h)), crop)
        reveal_px -= w
