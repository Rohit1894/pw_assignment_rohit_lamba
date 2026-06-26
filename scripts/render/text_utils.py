"""Text/script utilities: Devanagari detection, glyph sanitisation, grapheme-
cluster splitting, word-wrap, and math tokenisation. Extracted from
render_video.py (Step 1 refactor)."""

import re

from render.constants import _SUPERSCRIPT_MAP, _SUBSCRIPT_MAP


def _contains_devanagari(text):
    """True if the text contains any Devanagari (Hindi) characters."""
    return any("ऀ" <= ch <= "ॿ" for ch in (text or ""))


# Cosmetic glyph normalisation applied to ALL scripts — these substitutions are
# always safe and font-independent (fancy dashes/bullets → plain ASCII).
_GLYPH_FIXUPS_COMMON = {
    "•": "-", "·": "-", "—": "-", "–": "-",
}


# Arrows the Devanagari handwriting font (Kalam) lacks → would render as empty
# boxes on the Hindi crop-reveal path (which draws one whole string in a single
# font, with no per-glyph fallback). Map them to "=" ONLY for Devanagari text.
# English/Latin text keeps the REAL arrow: its draw path now falls back to a
# symbol font that can render it (see _resolve_glyph_font), so a note like
# "force → acceleration" no longer turns into "force = acceleration".
_GLYPH_FIXUPS_DEVANAGARI = {
    "→": "=", "➝": "=", "⟶": "=", "⇒": "=", "↓": "=", "←": "=",
}


_MATH_CHARS_RE = re.compile(r"[=√^₀-₉⁰¹²³⁴⁵⁶⁷⁸⁹]|\bformula\b", re.IGNORECASE)


def _sanitize_text(text):
    """Replace glyphs missing from the handwriting font with safe equivalents.

    The arrow→"=" substitution is scoped to Devanagari text only: the Hindi path
    pre-renders a whole string in one font (Kalam) with no per-glyph fallback, so
    an arrow there would be an empty box. English/Latin text is left intact
    because its draw path falls back to a symbol font per missing glyph.
    """
    text = str(text or "")
    for bad, good in _GLYPH_FIXUPS_COMMON.items():
        text = text.replace(bad, good)
    if _contains_devanagari(text):
        for bad, good in _GLYPH_FIXUPS_DEVANAGARI.items():
            text = text.replace(bad, good)
    return text


def _is_formula_like_text(text):
    """True for equation/formula notes that should use the math writer."""
    text = str(text or "")
    if len(text.strip()) < 6:
        return False
    return bool(_MATH_CHARS_RE.search(text))


def split_grapheme_clusters(text):
    """
    Split text into grapheme clusters suitable for a progressive 'writing'
    reveal of Devanagari (and plain Latin) text.

    A cluster = a base character plus any combining marks that attach to it
    (matras, anusvara, nukta, etc.). A virama (्) keeps the following
    consonant in the same cluster so conjuncts are never split mid-stroke.
    """
    clusters = []
    VIRAMA = "्"
    for ch in text:
        if not clusters:
            clusters.append(ch)
            continue
        prev = clusters[-1][-1]
        is_combining = "ऀ" <= ch <= "ः" or "ऺ" <= ch <= "ॏ" \
            or "॑" <= ch <= "ॗ" or ch in ("़", "ॢ", "ॣ",
                                                    "‌", "‍")
        if is_combining or prev == VIRAMA:
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


def wrap_text_to_width(draw, text, font, max_width):
    """Word-wrap `text` so each rendered line fits within `max_width` pixels."""
    words = text.split(" ")
    lines = []
    current = ""
    for word in words:
        trial = word if not current else current + " " + word
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


# ── Tokenizer for math equations (Word-wise reveal) ────────────────────────
# Word class includes Greek (U+0370–03FF) and every sub/superscript char so they
# are NOT dropped (the old class silently discarded ρ, ᵃ, etc.); the trailing
# "|." catch-all guarantees no character is ever lost during tokenisation.
_MATH_TOKEN_CHARS = "".join(_SUPERSCRIPT_MAP) + "".join(_SUBSCRIPT_MAP)


_MATH_TOKEN_RE = re.compile(
    "[A-Za-z0-9Ͱ-Ͽ" + _MATH_TOKEN_CHARS + r"]+|\s+|[^\w\s]|.",
    re.UNICODE,
)


def split_into_math_tokens(text):
    """
    Split a math equation into logical tokens (words, symbols, operators).
    Groups letters/numbers/Greek/sub-superscripts together, separating operators.
    """
    return _MATH_TOKEN_RE.findall(text or "")
