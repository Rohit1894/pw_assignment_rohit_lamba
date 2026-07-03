#!/usr/bin/env python3
"""Check that every symbol in the storyboard's board_lines can be rendered.

Detects missing glyphs before the video is rendered so the user sees a clear
error instead of mystery □ boxes. Falls back gracefully: symbols the primary
font can't draw are assigned a fallback font; if NO font can draw them they
are substituted with an ASCII-safe alternative.

Output: output/glyph_report.json
"""

import json
import os
import sys


# Symbols that frequently appear in STEM board writing but are absent from
# many handwriting fonts — pre-declare their expected fallback tier.
_MATH_SYMBOLS = set("√∫ΣπθαβγδεζηλμνρστφχψωΩΔ≤≥≈±×÷→⇌∞∝∂∇°¹²³⁰⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉")

# ASCII-safe fallback text for symbols that cannot be rendered at all
_SYMBOL_FALLBACK = {
    "√": "sqrt",
    "∫": "integral",
    "Σ": "Sigma",
    "π": "pi",
    "θ": "theta",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "Δ": "Delta",
    "ε": "epsilon",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "φ": "phi",
    "ψ": "psi",
    "ω": "omega",
    "Ω": "Ohm",
    "≤": "<=",
    "≥": ">=",
    "≈": "~=",
    "±": "+/-",
    "×": "x",
    "÷": "/",
    "→": "->",
    "⇌": "<->",
    "∞": "inf",
    "∝": "proportional to",
    "∂": "partial",
    "∇": "del",
    "°": "deg",
    "²": "2",
    "³": "3",
    "¹": "1",
    "⁰": "0",
    "₂": "2",
    "₃": "3",
    "₄": "4",
}


def _get_primary_font():
    """Return the Kalam (primary handwriting) font path from the render package."""
    try:
        from render.fonts import _bundled
        path = _bundled("Kalam-Regular.ttf")
        if os.path.exists(path):
            return path
    except Exception:
        pass
    return None


def _font_covers(font_path, ch):
    """True if the PIL/truetype font at font_path has a glyph for ch."""
    if not font_path or not os.path.exists(font_path):
        return False
    try:
        from PIL import ImageFont
        font = ImageFont.truetype(font_path, size=30)
        # PIL's getmask is the most reliable check: if the glyph is notdef the
        # bounding box of the mask is (0,0) for most modern fonts.
        from PIL import Image, ImageDraw
        img = Image.new("L", (60, 60), 0)
        draw = ImageDraw.Draw(img)
        draw.text((5, 5), ch, font=font, fill=255)
        return img.getbbox() is not None
    except Exception:
        return False


def _get_system_math_font():
    """Return a system math/unicode font path, or None."""
    candidates = [
        # Windows
        r"C:\Windows\Fonts\seguisym.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\times.ttf",
        # Linux
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        # macOS
        "/Library/Fonts/Arial Unicode MS.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _collect_symbols(storyboard: dict) -> list:
    """Extract every unique character from all board_lines in the storyboard."""
    seen = set()
    for step in storyboard.get("steps", []):
        for line in (step.get("board_lines") or []):
            for ch in str(line):
                if ord(ch) > 127:  # non-ASCII — the ones that may be missing
                    seen.add(ch)
    return sorted(seen)


def check_storyboard(storyboard: dict,
                     output_path: str = "output/glyph_report.json") -> dict:
    """Check every non-ASCII symbol in the storyboard board_lines.

    Returns a dict with status / missing_symbols / fallback_used and writes
    output/glyph_report.json.
    """
    symbols = _collect_symbols(storyboard)
    primary = _get_primary_font()
    math_font = _get_system_math_font()

    missing = []
    fallback_used = {}

    for ch in symbols:
        if _font_covers(primary, ch):
            continue
        # Try math / unicode fallback font
        if math_font and _font_covers(math_font, ch):
            fallback_used[ch] = "math_font"
            continue
        # No font can render this — record a text substitution
        sub = _SYMBOL_FALLBACK.get(ch, f"[{hex(ord(ch))}]")
        missing.append({"symbol": ch, "substitution": sub,
                        "description": f"No font glyph; will render as '{sub}'"})

    status = "safe" if not missing else "substituted"
    report = {
        "status": status,
        "symbols_checked": len(symbols),
        "missing_symbols": missing,
        "fallback_used": fallback_used,
        "primary_font": primary or "unknown",
        "math_font": math_font or "none",
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    n_fb = len(fallback_used)
    n_miss = len(missing)
    print(f"  Glyph check: {len(symbols)} non-ASCII symbols; "
          f"{n_fb} using fallback font, {n_miss} substituted -> {output_path}")
    if missing:
        for m in missing:
            print(f"    WARN: '{m['symbol']}' → '{m['substitution']}'")
    return report


def apply_substitutions(text: str, report: dict) -> str:
    """Replace un-renderable symbols in `text` with their ASCII substitutions."""
    for entry in report.get("missing_symbols", []):
        text = text.replace(entry["symbol"], entry["substitution"])
    return text


if __name__ == "__main__":
    sb_path = sys.argv[1] if len(sys.argv) > 1 else "output/storyboard.json"
    with open(sb_path, encoding="utf-8") as f:
        sb = json.load(f)
    check_storyboard(sb)
