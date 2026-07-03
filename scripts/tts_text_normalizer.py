#!/usr/bin/env python3
"""Convert board/formula text into safe spoken Hinglish for Sarvam TTS.

Raw math notation sounds terrible in TTS:
  "v² = u² + 2as"  →  "v square equals u square plus two a s"
  "0² = 20² − 2×10×H"  →  "zero square equals twenty square minus two into ten into H"

This module is called BEFORE sending text to Sarvam so the narration sounds
like a real teacher, not a text-dump.
"""

import re


# ── Symbol → spoken form ────────────────────────────────────────────────────

_SYMBOL_MAP = [
    # Operators (order matters: multi-char first)
    (r"≤",       "less than or equal to"),
    (r"≥",       "greater than or equal to"),
    (r"≠",       "not equal to"),
    (r"≈",       "approximately equal to"),
    (r"⇌",       "reversible reaction"),
    (r"→",       "gives"),
    (r"⟶",       "gives"),
    (r"\+-",     "plus or minus"),
    (r"±",       "plus or minus"),
    (r"\*",      "into"),
    (r"×",       "into"),
    (r"÷",       "divided by"),
    (r"·",       "into"),
    # Greek letters
    (r"θ",       "theta"),
    (r"α",       "alpha"),
    (r"β",       "beta"),
    (r"γ",       "gamma"),
    (r"δ",       "delta"),
    (r"Δ",       "delta"),
    (r"ε",       "epsilon"),
    (r"ζ",       "zeta"),
    (r"η",       "eta"),
    (r"λ",       "lambda"),
    (r"μ",       "mu"),
    (r"ν",       "nu"),
    (r"π",       "pi"),
    (r"ρ",       "rho"),
    (r"σ",       "sigma"),
    (r"Σ",       "sigma"),
    (r"τ",       "tau"),
    (r"φ",       "phi"),
    (r"χ",       "chi"),
    (r"ψ",       "psi"),
    (r"ω",       "omega"),
    (r"Ω",       "ohm"),
    # Math symbols
    (r"√",       "square root of"),
    (r"∫",       "integral of"),
    (r"∞",       "infinity"),
    (r"∝",       "proportional to"),
    (r"∂",       "partial"),
    (r"∇",       "del"),
    # Degree / units
    (r"°C",      "degree Celsius"),
    (r"°F",      "degree Fahrenheit"),
    (r"°",       "degree"),
    # Superscripts (Unicode)
    (r"⁰",  "0"), (r"¹", "1"), (r"²", "2"), (r"³", "3"),
    (r"⁴",  "4"), (r"⁵", "5"), (r"⁶", "6"), (r"⁷", "7"),
    (r"⁸",  "8"), (r"⁹", "9"),
    # Subscripts (Unicode)
    (r"₀", "0"), (r"₁", "1"), (r"₂", "2"), (r"₃", "3"),
    (r"₄", "4"), (r"₅", "5"), (r"₆", "6"), (r"₇", "7"),
    (r"₈", "8"), (r"₉", "9"),
]

# Compiled once
_COMPILED = [(re.compile(pat), repl) for pat, repl in _SYMBOL_MAP]

# Common unit patterns → spoken form
_UNIT_MAP = {
    r"m/s²":   "meter per second square",
    r"m/s2":   "meter per second square",
    r"km/h":   "kilometer per hour",
    r"m/s":    "meter per second",
    r"kg/m³":  "kilogram per meter cube",
    r"N/m":    "Newton per meter",
    r"J/kg":   "joule per kilogram",
    r"W/m²":   "watt per meter square",
    r"mol/L":  "mole per litre",
}
_UNIT_RE = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in _UNIT_MAP) + r')\b'
)

# Caret/LaTeX superscript patterns: x^2, x^{10}, \frac{a}{b}
_CARET_RE   = re.compile(r'(\w)\^(\{[^}]+\}|\d+|-?\d+)')
_FRAC_RE    = re.compile(r'\\frac\{([^}]+)\}\{([^}]+)\}')
_SQRT_RE    = re.compile(r'\\sqrt\{([^}]+)\}')
_TEXT_RE    = re.compile(r'\\text\{([^}]+)\}')
_SUBSCRIPT_RE = re.compile(r'(\w)_(\{[^}]+\}|\d+|\w)')


def _expand_caret(m):
    base = m.group(1)
    exp = m.group(2).strip("{}")
    try:
        n = int(exp)
        word = {2: "square", 3: "cube"}.get(n, f"to the power {n}")
    except ValueError:
        word = f"to the power {exp}"
    return f"{base} {word}"


def _expand_subscript(m):
    base = m.group(1)
    sub = m.group(2).strip("{}")
    return f"{base} {sub}"


def _numbers_to_words_simple(text):
    """Replace standalone digit sequences with spoken words for common small numbers."""
    _ONES = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
        "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
        "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
        "18": "eighteen", "19": "nineteen", "20": "twenty",
        "25": "twenty five", "30": "thirty", "40": "forty", "50": "fifty",
        "60": "sixty", "70": "seventy", "80": "eighty", "90": "ninety",
        "100": "hundred", "400": "four hundred", "1000": "thousand",
    }
    def replace_num(m):
        s = m.group(0)
        return _ONES.get(s, s)
    return re.sub(r'\b\d+\b', replace_num, text)


def normalize_for_tts(text: str, expand_numbers: bool = False) -> str:
    """Convert board/formula text to a form safe for Sarvam TTS.

    This is best-effort — the storyboard's `tts_narration_text` is
    hand-crafted by Gemini, so normalisation here catches edge cases where
    a raw board line leaks into narration without being rewritten.
    """
    if not text:
        return text

    # 1. LaTeX macros
    text = _FRAC_RE.sub(lambda m: f"{m.group(1)} over {m.group(2)}", text)
    text = _SQRT_RE.sub(lambda m: f"square root of {m.group(1)}", text)
    text = _TEXT_RE.sub(lambda m: m.group(1), text)

    # 2. Caret / subscript notation
    text = _CARET_RE.sub(_expand_caret, text)
    text = _SUBSCRIPT_RE.sub(_expand_subscript, text)

    # 3. Units before generic symbol replacement
    text = _UNIT_RE.sub(lambda m: _UNIT_MAP[m.group(1)], text)

    # 4. Symbols
    for pat, repl in _COMPILED:
        text = pat.sub(repl, text)

    # 5. Equals sign: keep as "equals" only if it looks like an equation
    text = re.sub(r'\s*=\s*', " equals ", text)

    # 6. Optionally expand numbers
    if expand_numbers:
        text = _numbers_to_words_simple(text)

    # 7. Clean up multiple spaces
    text = re.sub(r' {2,}', ' ', text).strip()

    return text


def normalize_board_lines_for_tts(board_lines: list, expand_numbers: bool = True) -> str:
    """Join a list of board_lines into a single TTS-safe sentence.

    Used when a storyboard step's tts_narration_text falls back to the board
    content (shouldn't happen normally — Gemini writes the narration — but
    this is the safety net).
    """
    parts = [normalize_for_tts(line, expand_numbers)
             for line in (board_lines or []) if line]
    return ". ".join(p.rstrip(".") for p in parts if p)


if __name__ == "__main__":
    tests = [
        "v² = u² + 2as",
        "0² = 20² − 2 × 10 × H",
        "H₂SO₄",
        "CO₂",
        "m/s²",
        "√x",
        "θ = 30°",
        "\\frac{400}{20}",
        "x^2 + y^2 = r^2",
        "μ₀ = 4π × 10^{-7}",
    ]
    for t in tests:
        print(f"  {t!r:40s} → {normalize_for_tts(t, expand_numbers=True)!r}")
