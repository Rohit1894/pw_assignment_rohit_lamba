"""Font location, glyph-presence detection, per-glyph symbol fallback, and
sized-font caching. Extracted from render_video.py (Step 1 refactor)."""

import os

from PIL import ImageFont


# ── Font helper ─────────────────────────────────────────────────────────────
def _find_hindi_font(size=30):
    """
    Locate a Devanagari-capable font for writing Hindi solution lines.

    Prefers the bundled 'Kalam' handwriting font (keeps the hand-written
    teacher feel), then falls back to Windows' Nirmala UI, then any system
    Devanagari font. Hindi must be drawn as whole words/clusters (never
    character-by-character) so that matras and conjuncts shape correctly.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(here))
    candidates = [
        os.path.join(project_root, "fonts", "Kalam-Regular.ttf"),
        os.path.join(here, "fonts", "Kalam-Regular.ttf"),
        "C:/Windows/Fonts/Nirmala.ttc",
        "C:/Windows/Fonts/Nirmala.ttf",
        "C:/Windows/Fonts/mangal.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                # Kalam is a touch small visually; bump it slightly.
                font_size = size + 4 if "Kalam" in path else size
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    return ImageFont.load_default(size=size)


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


# ── Per-glyph font fallback ──────────────────────────────────────────────────
# Handwriting fonts (Ink Free / Segoe Print) lack Greek letters, math operators
# (∫ ∑ ≤ ≥ ≠ ± × ÷ ∞ ∂ ∝ √) and arrows, rendering them as empty boxes. When the
# primary font is missing a glyph we draw THAT glyph from a broad-coverage symbol
# font (Segoe UI Symbol, then Arial, then DejaVu on Linux) at the same size, so a
# chemistry/physics/maths solution in English renders every symbol.
_GLYPH_FALLBACK_PATHS = [
    "C:/Windows/Fonts/seguisym.ttf",   # Segoe UI Symbol: Greek + math + arrows
    "C:/Windows/Fonts/arial.ttf",      # Greek + Latin-1 math basics
    "C:/Windows/Fonts/cambria.ttc",    # Cambria Math: full math coverage
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


_GLYPH_FALLBACK_CACHE = {}      # size -> [loaded fallback fonts]


_GLYPH_PRESENCE_CACHE = {}      # (font_path, size, char) -> bool


def _font_has_glyph(font, ch):
    """True if `font` has a real (non-notdef) glyph for `ch`.

    Dependency-free: compares the glyph's 1-bit mask against the font's notdef
    mask (the same technique as check_notdef.py). A blank mask for a visible
    character, or a mask identical to notdef, means the glyph is missing.
    """
    if not ch or ch.isspace():
        return True
    key = (getattr(font, "path", None), getattr(font, "size", None), ch)
    cached = _GLYPH_PRESENCE_CACHE.get(key)
    if cached is not None:
        return cached
    present = True
    try:
        mask = font.getmask(ch, mode="1")
        if mask.getbbox() is None:
            present = False
        else:
            notdef = font.getmask("", mode="1")  # PUA: almost always notdef
            if mask.size == notdef.size and bytes(mask) == bytes(notdef):
                present = False
    except Exception:
        present = True
    _GLYPH_PRESENCE_CACHE[key] = present
    return present


def _glyph_fallback_fonts(size):
    """Load (once, cached) the fallback symbol fonts at `size`."""
    fonts = _GLYPH_FALLBACK_CACHE.get(size)
    if fonts is None:
        fonts = []
        for path in _GLYPH_FALLBACK_PATHS:
            if os.path.exists(path):
                try:
                    fonts.append(ImageFont.truetype(path, size))
                except Exception:
                    continue
        _GLYPH_FALLBACK_CACHE[size] = fonts
    return fonts


def _resolve_glyph_font(ch, base_font):
    """Return the font to draw `ch` with: the handwriting font if it owns the
    glyph, otherwise the first fallback font that does (else the base font)."""
    if _font_has_glyph(base_font, ch):
        return base_font
    size = int(getattr(base_font, "size", 26) or 26)
    for fb in _glyph_fallback_fonts(size):
        if _font_has_glyph(fb, ch):
            return fb
    return base_font


def _sized_sub_font(font):
    """0.65x variant of `font` for sub/superscript bases (cached in _FONT_CACHE)."""
    path = getattr(font, "path", None)
    size = max(10, int(getattr(font, "size", 26) * 0.65))
    if not path or not os.path.exists(path):
        return font
    key = (path, size)
    f = _FONT_CACHE.get(key)
    if f is None:
        try:
            f = ImageFont.truetype(path, size)
        except Exception:
            f = font
        _FONT_CACHE[key] = f
    return f


# Sized-font cache so dynamic font sizing doesn't reload TTFs every call.
_FONT_CACHE = {}


def _sized_variant(font, size):
    """Return `font` re-instantiated at `size` (clamped, cached)."""
    size = int(max(15, min(46, size)))
    path = getattr(font, "path", None)
    key = (path, size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    try:
        variant = ImageFont.truetype(path, size) if path else font
    except Exception:
        variant = font
    _FONT_CACHE[key] = variant
    return variant


def _note_font_size(text, base):
    """Pick a font size for a note based on its length (dynamic sizing)."""
    n = len(text)
    if n <= 8:
        return base + 6
    if n <= 16:
        return base + 2
    if n <= 28:
        return base - 2
    return base - 6
