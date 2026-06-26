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


# ── LaTeX-to-Unicode substitution ────────────────────────────────────────────
# Common LaTeX commands the model emits in English physics/chemistry/maths.
# Applied in _sanitize_text so every render path benefits automatically.
# \\frac is intentionally ABSENT — it is handled by draw_math_equation_with_radicals
# which renders it as a proper stacked fraction, not a Unicode substitute.
_LATEX_SUBS = [
    (r'\propto',     '∝'),
    (r'\rightarrow', '→'),
    (r'\leftarrow',  '←'),
    (r'\Rightarrow', '⇒'),
    (r'\Leftarrow',  '⇐'),
    (r'\approx',     '≈'),
    (r'\times',      '×'),
    (r'\cdot',       '·'),
    (r'\pm',         '±'),
    (r'\mp',         '∓'),
    (r'\leq',        '≤'),
    (r'\geq',        '≥'),
    (r'\neq',        '≠'),
    (r'\infty',      '∞'),
    (r'\partial',    '∂'),
    (r'\nabla',      '∇'),
    (r'\Delta',      'Δ'),
    (r'\Sigma',      'Σ'),
    (r'\Pi',         'Π'),
    (r'\Omega',      'Ω'),
    (r'\delta',      'δ'),
    (r'\alpha',      'α'),
    (r'\beta',       'β'),
    (r'\gamma',      'γ'),
    (r'\lambda',     'λ'),
    (r'\mu',         'μ'),
    (r'\nu',         'ν'),
    (r'\pi',         'π'),
    (r'\rho',        'ρ'),
    (r'\sigma',      'σ'),
    (r'\theta',      'θ'),
    (r'\omega',      'ω'),
    (r'\eta',        'η'),
    (r'\epsilon',    'ε'),
    (r'\phi',        'φ'),
    (r'\psi',        'ψ'),
    (r'\xi',         'ξ'),
    (r'\zeta',       'ζ'),
    (r'\chi',        'χ'),
    (r'\tau',        'τ'),
    (r'\kappa',      'κ'),
    (r'\iota',       'ι'),
    (r'\%',          '%'),
]


def _expand_latex(text):
    """Substitute common LaTeX commands with their Unicode equivalents.

    Called from _sanitize_text. \\frac is deliberately excluded — it is rendered
    as a stacked fraction by draw_math_equation_with_radicals.
    """
    for cmd, sub in _LATEX_SUBS:
        text = text.replace(cmd, sub)
    return text


# Inverse of _SUPERSCRIPT_MAP: normal base char → Unicode superscript.
_TO_SUPERSCRIPT = {v: k for k, v in _SUPERSCRIPT_MAP.items()}

# Matches caret-notation exponents: ^(expr), ^{expr}, or ^X (single char).
_CARET_RE = re.compile(r'\^(\(([^)]*)\)|\{([^}]*)\}|([A-Za-z0-9+\-]))')


def _expand_carets(text):
    """Convert caret-notation superscripts to Unicode superscript characters.

    Handles:
      ^a       →  ᵃ          (single letter / digit / sign)
      ^(b+c)   →  ⁽ᵇ⁺ᶜ⁾     (parenthesised expression)
      ^{a-3b}  →  ⁽ᵃ⁻³ᵇ⁾    (LaTeX-braced expression)

    Characters with no Unicode superscript equivalent are kept as-is.
    """
    def _to_sup(s, wrap_parens):
        inner = ''.join(_TO_SUPERSCRIPT.get(c, c) for c in s)
        if wrap_parens:
            return _TO_SUPERSCRIPT.get('(', '(') + inner + _TO_SUPERSCRIPT.get(')', ')')
        return inner

    def replace(m):
        if m.group(2) is not None:   # ^(expr)
            return _to_sup(m.group(2), wrap_parens=True)
        if m.group(3) is not None:   # ^{expr}
            return _to_sup(m.group(3), wrap_parens=True)
        return _to_sup(m.group(4), wrap_parens=False)  # single char

    return _CARET_RE.sub(replace, text)


def _sanitize_text(text):
    """Replace glyphs missing from the handwriting font with safe equivalents,
    and expand LaTeX commands and caret notation to Unicode.

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
    text = _expand_latex(text)
    text = _expand_carets(text)
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
