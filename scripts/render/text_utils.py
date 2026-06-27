"""Text/script utilities: Devanagari detection, glyph sanitisation, grapheme-
cluster splitting, word-wrap, and math tokenisation. Extracted from
render_video.py (Step 1 refactor)."""

import re

from render.constants import _SUPERSCRIPT_MAP, _SUBSCRIPT_MAP


def _contains_devanagari(text):
    """True if the text contains any Devanagari (Hindi) characters."""
    return any("ऀ" <= ch <= "ॿ" for ch in (text or ""))


# Cosmetic glyph normalisation applied to ALL scripts.
# NOTE: middle-dot (·, U+00B7) is intentionally NOT here — it is a valid
# multiplication operator in chemistry/physics units (N·m, kg·m²/s) and the
# per-glyph fallback supplies it from a symbol font when Kalam lacks it.
_GLYPH_FIXUPS_COMMON = {
    "•": "-", "—": "-", "–": "-",
}


# Arrows the Devanagari handwriting font (Kalam) lacks → empty boxes on the
# Hindi crop-reveal path (whole string drawn in one font, no per-glyph fallback).
# Map to "=" ONLY for Devanagari text. English/Latin text keeps the real glyph
# because its draw path falls back to a symbol font per missing character.
_GLYPH_FIXUPS_DEVANAGARI = {
    "→": "=", "➝": "=", "⟶": "=", "⇒": "=",
    "↓": "=", "←": "=", "↑": "=", "⇌": "=",
}


_MATH_CHARS_RE = re.compile(
    r"[=√^∫∑∏∂∇∝∀∃₀-₉⁰¹²³⁴⁵⁶⁷⁸⁹]|\bformula\b", re.IGNORECASE
)


# ── LaTeX-to-Unicode substitution ────────────────────────────────────────────
# Common LaTeX commands the model may emit in English physics/chemistry/maths.
# Applied in _sanitize_text so every render path benefits automatically.
# \\frac is intentionally ABSENT — it is rendered as a proper stacked fraction
# by draw_math_equation_with_radicals, not a Unicode substitute.
#
# ORDERING RULES — longer/more-specific patterns MUST precede any shorter
# pattern that is a substring of them to avoid partial clobbering:
#   \rightleftharpoons  before  \rightarrow
#   \cdots              before  \cdot
#   \infty, \int        before  \in
#   \subseteq           before  \subset
#   \supseteq           before  \supset
#   \uparrow            before  \upsilon
#   \to  comes LAST among \t-entries (it is short; others don't overlap it
#        but safer to put it after the full Greek set)
_LATEX_SUBS = [
    # ── Multi-char arrows ──────────────────────────────────────────────────
    (r'\rightleftharpoons', '⇌'),   # equilibrium arrow (chemistry)
    (r'\longrightarrow',    '→'),
    (r'\Rightarrow',        '⇒'),
    (r'\Leftarrow',         '⇐'),
    (r'\rightarrow',        '→'),
    (r'\leftarrow',         '←'),
    (r'\uparrow',           '↑'),   # before \upsilon
    (r'\downarrow',         '↓'),
    # ── Set / logic ────────────────────────────────────────────────────────
    (r'\subseteq',          '⊆'),   # before \subset
    (r'\supseteq',          '⊇'),   # before \supset
    (r'\subset',            '⊂'),
    (r'\supset',            '⊃'),
    (r'\notin',             '∉'),
    (r'\forall',            '∀'),
    (r'\exists',            '∃'),
    (r'\implies',           '⟹'),
    (r'\iff',               '⟺'),
    (r'\cup',               '∪'),
    (r'\cap',               '∩'),
    # ── Must precede \in ───────────────────────────────────────────────────
    (r'\infty',             '∞'),
    (r'\int',               '∫'),   # before \in
    (r'\in',                '∈'),
    # ── Calculus / analysis ────────────────────────────────────────────────
    (r'\sum',               '∑'),
    (r'\prod',              '∏'),
    (r'\partial',           '∂'),
    (r'\nabla',             '∇'),
    (r'\propto',            '∝'),
    (r'\approx',            '≈'),
    (r'\therefore',         '∴'),
    (r'\because',           '∵'),
    (r'\lim',               'lim'),
    (r'\sqrt',              '√'),   # bare \sqrt (braced form handled by regex above)
    # ── Arithmetic / comparison ─────────────────────────────────────────────
    (r'\times',             '×'),
    (r'\cdots',             '…'),   # before \cdot
    (r'\cdot',              '·'),
    (r'\pm',                '±'),
    (r'\mp',                '∓'),
    (r'\leq',               '≤'),
    (r'\geq',               '≥'),
    (r'\neq',               '≠'),
    # ── Geometry / physics misc ─────────────────────────────────────────────
    (r'\angle',             '∠'),
    (r'\perp',              '⊥'),
    (r'\parallel',          '∥'),
    (r'\hbar',              'ℏ'),   # reduced Planck constant
    (r'\degree',            '°'),
    (r'\ldots',             '…'),
    (r'\dots',              '…'),
    # ── Uppercase Greek (full set) ──────────────────────────────────────────
    (r'\Delta',             'Δ'),
    (r'\Sigma',             'Σ'),
    (r'\Gamma',             'Γ'),
    (r'\Theta',             'Θ'),
    (r'\Lambda',            'Λ'),
    (r'\Xi',                'Ξ'),
    (r'\Pi',                'Π'),
    (r'\Phi',               'Φ'),
    (r'\Psi',               'Ψ'),
    (r'\Omega',             'Ω'),
    (r'\Upsilon',           'Υ'),
    # ── Lowercase Greek (full set) ──────────────────────────────────────────
    (r'\alpha',             'α'),
    (r'\beta',              'β'),
    (r'\gamma',             'γ'),
    (r'\delta',             'δ'),
    (r'\epsilon',           'ε'),
    (r'\zeta',              'ζ'),
    (r'\eta',               'η'),
    (r'\theta',             'θ'),
    (r'\iota',              'ι'),
    (r'\kappa',             'κ'),
    (r'\lambda',            'λ'),
    (r'\mu',                'μ'),
    (r'\nu',                'ν'),
    (r'\xi',                'ξ'),
    (r'\pi',                'π'),
    (r'\rho',               'ρ'),
    (r'\sigma',             'σ'),
    (r'\tau',               'τ'),
    (r'\upsilon',           'υ'),
    (r'\phi',               'φ'),
    (r'\chi',               'χ'),
    (r'\psi',               'ψ'),
    (r'\omega',             'ω'),
    # ── \to LAST among \t entries (short; put after full Greek to be safe) ──
    (r'\to',                '→'),
    # ── Misc ────────────────────────────────────────────────────────────────
    (r'\%',                 '%'),
]

# Regex handlers applied BEFORE the simple-replace table — these capture and
# transform braced arguments that the table's simple str.replace can't handle.
_SQRT_RE  = re.compile(r'\\sqrt\s*\{([^}]*)\}')
_VEC_RE   = re.compile(r'\\vec\s*\{([^}]*)\}')
_HAT_RE   = re.compile(r'\\hat\s*\{([^}]*)\}')
_BAR_RE   = re.compile(r'\\bar\s*\{([^}]*)\}')
_DOT_RE   = re.compile(r'\\dot\s*\{([^}]*)\}')


def _expand_latex(text):
    """Substitute common LaTeX commands with their Unicode equivalents.

    Called from _sanitize_text. \\frac is deliberately excluded — it is rendered
    as a stacked fraction by draw_math_equation_with_radicals.
    """
    # Braced-argument commands first (regex), then simple string replacements.
    text = _SQRT_RE.sub(lambda m: f'√({m.group(1)})', text)     # √(expr)
    text = _VEC_RE.sub(lambda m: m.group(1) + '⃗', text)        # X + combining right-arrow above
    text = _HAT_RE.sub(lambda m: m.group(1) + '̂', text)        # x + combining hat
    text = _BAR_RE.sub(lambda m: m.group(1) + '̅', text)        # x + combining overline
    text = _DOT_RE.sub(lambda m: m.group(1) + '̇', text)        # x + combining dot above
    for cmd, sub in _LATEX_SUBS:
        text = text.replace(cmd, sub)
    return text


# Inverse of _SUPERSCRIPT_MAP: normal base char → Unicode superscript.
_TO_SUPERSCRIPT = {v: k for k, v in _SUPERSCRIPT_MAP.items()}

# Inverse of _SUBSCRIPT_MAP: normal base char → Unicode subscript.
_TO_SUBSCRIPT = {v: k for k, v in _SUBSCRIPT_MAP.items()}

# Matches caret-notation exponents: ^(expr), ^{expr}, or ^X (single char).
_CARET_RE = re.compile(r'\^(\(([^)]*)\)|\{([^}]*)\}|([A-Za-z0-9+\-]))')

# Matches underscore-notation subscripts: _(expr), _{expr}, or _X (single char).
_UNDERSCORE_RE = re.compile(r'_(\(([^)]*)\)|\{([^}]*)\}|([A-Za-z0-9+\-]))')


def _expand_carets(text):
    """Convert caret-notation superscripts to Unicode superscript characters.

    Handles:
      ^a       →  ᵃ          (single char — no parens)
      ^{-1}    →  ⁻¹         (2-char braced — no parens; common physics pattern)
      ^{n+1}   →  ⁽ⁿ⁺¹⁾     (3+ char braced — parens added for grouping)
      ^(b+c)   →  ⁽ᵇ⁺ᶜ⁾     (parenthesised expression)

    Parens are added only for expressions of 3+ characters so that common
    patterns like T^{-1} render as T⁻¹, not T⁽⁻¹⁾.
    Characters with no Unicode superscript equivalent are kept as-is.
    """
    def _to_sup(s, wrap_parens):
        inner = ''.join(_TO_SUPERSCRIPT.get(c, c) for c in s)
        if wrap_parens:
            lp = _TO_SUPERSCRIPT.get('(', '(')
            rp = _TO_SUPERSCRIPT.get(')', ')')
            return lp + inner + rp
        return inner

    def replace(m):
        if m.group(2) is not None:   # ^(expr)
            expr = m.group(2)
            return _to_sup(expr, wrap_parens=len(expr) > 2)
        if m.group(3) is not None:   # ^{expr}
            expr = m.group(3)
            return _to_sup(expr, wrap_parens=len(expr) > 2)
        return _to_sup(m.group(4), wrap_parens=False)  # single char

    return _CARET_RE.sub(replace, text)


def _expand_underscores(text):
    """Convert underscore-notation subscripts to Unicode subscript characters.

    Handles:
      _2       →  ₂          (single digit)
      _{n}     →  ₙ          (1-char braced)
      _{n+1}   →  ₍ₙ₊₁₎     (3+ char braced — parens added)
      _(expr)  →  ₍ₑₓₚᵣ₎    (parenthesised)

    Most letters lack Unicode subscript equivalents (_SUBSCRIPT_MAP covers only a
    subset); unmappable characters are kept as-is. Parens only for 3+ chars.
    """
    def _to_sub(s, wrap_parens):
        inner = ''.join(_TO_SUBSCRIPT.get(c, c) for c in s)
        if wrap_parens:
            lp = _TO_SUBSCRIPT.get('(', '₍')
            rp = _TO_SUBSCRIPT.get(')', '₎')
            return lp + inner + rp
        return inner

    def replace(m):
        if m.group(2) is not None:   # _(expr)
            expr = m.group(2)
            return _to_sub(expr, wrap_parens=len(expr) > 2)
        if m.group(3) is not None:   # _{expr}
            expr = m.group(3)
            return _to_sub(expr, wrap_parens=len(expr) > 2)
        return _to_sub(m.group(4), wrap_parens=False)  # single char

    return _UNDERSCORE_RE.sub(replace, text)


def _sanitize_text(text):
    """Replace glyphs missing from the handwriting font with safe equivalents,
    and expand LaTeX commands and caret/underscore notation to Unicode.

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
    text = _expand_underscores(text)
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
