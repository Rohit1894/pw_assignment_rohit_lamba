#!/usr/bin/env python3
"""Pre-expand numbers and math symbols in Roman Hinglish TTS narration text.

Sarvam's bulbul:v3 hi-IN model reads Arabic numerals as Hindi words (0→sunya,
2→do, 20→bees, 100→ek sau) and math symbols as Hindi terms (=→barabar,
×→guna).  This module converts them to English words BEFORE the API call.

Examples (after normalization):
  "LC = 0.01 mm"           → "LC equals zero point zero one millimeters"
  "N = 100"                → "N equals one hundred"
  "v² = u² - 2gH"         → "v squared equals u squared minus two gH"
  "g = 10 m/s²"           → "g equals ten meters per second squared"
  "ZE = + (5 × 0.01 mm)"  → "ZE equals plus (five times zero point zero one millimeters)"
  "4.60 mm - 0.05 mm"     → "four point six zero millimeters minus zero point zero five millimeters"

Usage:
    from narration_normalizer import normalize_tts_text
    spoken = normalize_tts_text(display_narration_roman)
"""

import re

# ─────────────────────────────────────────────────────────────────────────────
# Number → English words
# ─────────────────────────────────────────────────────────────────────────────
_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def _int_to_words(n: int) -> str:
    """0–9999 integer → English words.  Negative → 'minus …'."""
    if n < 0:
        return "minus " + _int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return (_TENS[n // 10] + (" " + _ONES[n % 10] if n % 10 else "")).strip()
    if n < 1000:
        h = _ONES[n // 100] + " hundred"
        r = n % 100
        return (h + " " + _int_to_words(r)).strip() if r else h
    if n < 10000:
        th = _int_to_words(n // 1000) + " thousand"
        r = n % 1000
        return (th + " " + _int_to_words(r)).strip() if r else th
    return str(n)   # fall back for very large numbers (rare in MCQ narration)


def _expand_decimal(m: "re.Match") -> str:
    """Regex callback: '4.60' → 'four point six zero', '-0.05' → 'minus zero point zero five'."""
    s = m.group()
    neg = s.startswith("-")
    s = s.lstrip("-")
    int_part, _, dec_part = s.partition(".")
    int_w = _int_to_words(int(int_part)) if int_part else "zero"
    dig_w = " ".join(_ONES[int(d)] for d in dec_part)
    result = int_w + " point " + dig_w
    return ("minus " + result) if neg else result


def _expand_int(m: "re.Match") -> str:
    """Regex callback: '100' → 'one hundred'."""
    return _int_to_words(int(m.group()))


# ─────────────────────────────────────────────────────────────────────────────
# Unit expansion (applied before number expansion so combined units like
# m/s² are handled as a unit phrase, not split into m / s²)
# ─────────────────────────────────────────────────────────────────────────────
def _expand_units(text: str) -> str:
    # Compound velocity / acceleration units — longest first
    text = re.sub(r'\bm/s²\b',   'meters per second squared', text)
    text = re.sub(r'\bm/s2\b',   'meters per second squared', text)
    text = re.sub(r'\bm/s\b',    'meters per second', text)
    text = re.sub(r'\bkm/h\b',   'kilometers per hour', text)
    text = re.sub(r'\bcm/s²\b',  'centimeters per second squared', text)
    text = re.sub(r'\bcm/s2\b',  'centimeters per second squared', text)
    text = re.sub(r'\bcm/s\b',   'centimeters per second', text)
    # Simple units — only expand when immediately after a digit to avoid
    # expanding variable names (e.g. "N" in "N = 100" must NOT become "newtons")
    text = re.sub(r'(?<=\d)\s*mm\b',   ' millimeters', text)
    text = re.sub(r'(?<=\d)\s*km\b',   ' kilometers', text)
    text = re.sub(r'(?<=\d)\s*cm\b',   ' centimeters', text)
    text = re.sub(r'(?<=\d)\s*m\b(?!m)', ' meters', text)   # after km so "km" isn't re-matched
    text = re.sub(r'(?<=\d)\s*kg\b',   ' kilograms', text)
    text = re.sub(r'(?<=\d)\s*mg\b',   ' milligrams', text)
    text = re.sub(r'(?<=\d)\s*MHz\b',  ' megahertz', text)
    text = re.sub(r'(?<=\d)\s*kHz\b',  ' kilohertz', text)
    text = re.sub(r'(?<=\d)\s*Hz\b',   ' hertz', text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Main normalizer
# ─────────────────────────────────────────────────────────────────────────────
def normalize_tts_text(text: str) -> str:
    """Return `text` with every digit, math symbol, and unit replaced by the
    English words that Sarvam hi-IN should say.

    Safe to call repeatedly — idempotent on already-expanded strings.
    Does NOT touch Hinglish prose; only digits and recognised symbols change.
    """
    if not text:
        return text

    # ── 1. Compound and simple units (before number expansion) ───────────────
    text = _expand_units(text)

    # ── 2. Superscripts ──────────────────────────────────────────────────────
    text = text.replace('²', ' squared').replace('³', ' cubed')
    text = re.sub(r'\^2\b', ' squared', text)
    text = re.sub(r'\^3\b', ' cubed',   text)

    # ── 3. Coefficients glued to variable letters: 2gH → two gH ─────────────
    # Matches a digit sequence not preceded by letter/digit/dot AND followed
    # by a letter, so "2g" → "two g" but "100" (standalone) is left for step 5.
    text = re.sub(
        r'(?<![a-zA-Z\d\.])(\d+)(?=[a-zA-Z])',
        lambda m: _int_to_words(int(m.group(1))) + ' ',
        text,
    )

    # ── 4. Decimal numbers (handles optional leading minus) ──────────────────
    text = re.sub(r'-?\d+\.\d+', _expand_decimal, text)

    # ── 5. Remaining integers ─────────────────────────────────────────────────
    text = re.sub(r'\b\d+\b', _expand_int, text)

    # ── 6. Math operators ─────────────────────────────────────────────────────
    text = text.replace('×', ' times ')
    text = text.replace('÷', ' divided by ')
    text = text.replace('√', ' root of ')
    text = re.sub(r'\s*=\s*', ' equals ', text)
    text = re.sub(r'(?<!\w)/(?!\w)', ' divided by ', text)   # standalone /
    text = re.sub(r'\s*%', ' percent', text)

    # ── 7. Plus / minus operators (after numbers are already words) ───────────
    # "-(+" inside parens: Corrected Reading = 4.60 - (+ 0.05)
    text = re.sub(r'\(\s*\+\s*', '(plus ',  text)
    text = re.sub(r'\(\s*-\s*',  '(minus ', text)
    # "-(" operator before a paren group: "4.60 - (plus..."
    text = re.sub(r'(?<=\s)-\s*\(', 'minus (', text)
    text = re.sub(r'(?<=\s)\+\s*\(', 'plus (',  text)
    # space-delimited + / - in expressions: "4 mm + 0.60 mm", "MR - ZE"
    text = re.sub(r'(?<=\s)-(?=\s)', 'minus', text)
    text = re.sub(r'(?<=\s)\+(?=\s)', 'plus',  text)

    # ── 8. Clean up ──────────────────────────────────────────────────────────
    text = re.sub(r'[ \t]+', ' ', text)            # collapse extra spaces
    text = re.sub(r' ([,.:;!?])', r'\1', text)     # no space before punctuation
    text = text.strip()

    return text


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test (run as: python narration_normalizer.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        # Numbers
        ("0 appears as sunya", "zero appears as sunya"),
        ("N = 100", "N equals one hundred"),
        ("P = 1 mm", "P equals one millimeters"),
        ("5 divisions", "five divisions"),
        ("60 circular scale", "sixty circular scale"),
        # Decimals
        ("LC = 0.01 mm", "LC equals zero point zero one millimeters"),
        ("ZE = + 0.05 mm", "ZE equals plus zero point zero five millimeters"),
        ("4.60 mm - 0.05 mm", "four point six zero millimeters minus zero point zero five millimeters"),
        ("4.55 mm", "four point five five millimeters"),
        ("0.60 mm", "zero point six zero millimeters"),
        # Operators
        ("5 × LC", "five times LC"),
        ("1 mm / 100", "one millimeters divided by one hundred"),
        ("ZE = + (5 × 0.01 mm)", "ZE equals plus (five times zero point zero one millimeters)"),
        ("CR = 4.60 mm - (+ 0.05 mm)", "CR equals four point six zero millimeters minus (plus zero point zero five millimeters)"),
        # Compound units
        ("g = 10 m/s²", "g equals ten meters per second squared"),
        ("u = 20 m/s", "u equals twenty meters per second"),
        # Coefficients + superscripts
        ("v² = u² - 2gH", "v squared equals u squared minus two gH"),
        # No-op (already words)
        ("Aaj hum sikhenge", "Aaj hum sikhenge"),
    ]

    passed = failed = 0
    for inp, expected in cases:
        got = normalize_tts_text(inp)
        ok = got == expected
        status = "PASS" if ok else "FAIL"
        if not ok:
            print(f"[{status}] {inp!r}")
            print(f"       expected: {expected!r}")
            print(f"       got:      {got!r}")
            failed += 1
        else:
            print(f"[{status}] {inp!r}")
            passed += 1

    print(f"\n{passed}/{passed+failed} tests passed.")
