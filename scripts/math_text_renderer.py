#!/usr/bin/env python3
"""Math text rendering utilities for the board.

Classifies each board_line as plain text, math, or chemistry so the renderer
can apply the right drawing path (existing render/text_utils.py handles the
actual rendering — this module is the pre-processing / classification layer).

Board-line types:
  "text"      — plain English, rendered with primary handwriting font
  "math"      — formula line (superscripts, fractions, operators) — uses
                the existing draw_math_equation_with_radicals pipeline
  "chemistry" — chemical formula (subscript numbers, element symbols)
  "heading"   — section heading like "Given:" or "Formula:" — bold/underlined

Usage:
    from math_text_renderer import classify_board_lines
    typed_lines = classify_board_lines(["Given:", "v² = u² + 2as", "H₂SO₄"])
    # [{"type": "heading", "value": "Given:"},
    #  {"type": "math",    "value": "v² = u² + 2as"},
    #  {"type": "chemistry","value": "H₂SO₄"}]
"""

import re

# Heading keywords that end with a colon
_HEADING_RE = re.compile(
    r"^(Given|Formula|Substitut\w*|Calculat\w*|Answer|Concept|Key idea|"
    r"Pairs|Explanation|Labels|Fill|Blanks|Assertion|Reason|Link|"
    r"Option check\w*|Eliminat\w*|Correct pair\w*)\s*:?\s*$",
    re.IGNORECASE,
)

# Lines that are likely formulas: contain math operators / superscripts
_MATH_CHARS = set("²³⁰¹⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉√∫ΣπθαβγδλμνρστφψωΩΔ≤≥≈±×÷→⇌∂∇")
_MATH_RE = re.compile(
    r"(\^|\\frac|\\sqrt|=\s*\d|[0-9]+\s*[+\-×÷/]\s*[0-9A-Za-z]"
    r"|[A-Za-z]\s*²|[A-Za-z]\^)"
)

# Chemical formula: element symbol + subscript digit, or formula with H₂, CO₂ etc.
_CHEM_RE = re.compile(
    r"\b([A-Z][a-z]?)\d|[₀₁₂₃₄₅₆₇₈₉]|"
    r"\b(H₂O|CO₂|H₂SO₄|NaCl|HCl|NaOH|CH₄|C₆H₁₂O₆)\b"
)


def classify_line(line: str) -> str:
    """Return 'heading', 'math', 'chemistry', or 'text'."""
    s = line.strip()
    if not s:
        return "text"
    if _HEADING_RE.match(s):
        return "heading"
    # Check for math characters
    if any(ch in _MATH_CHARS for ch in s) or _MATH_RE.search(s):
        return "math"
    # Check for chemistry — only if it looks like a formula, not a sentence
    if _CHEM_RE.search(s) and len(s) < 60 and not s.endswith("."):
        return "chemistry"
    return "text"


def classify_board_lines(board_lines: list) -> list:
    """Classify a list of board_line strings into typed dicts."""
    return [{"type": classify_line(line), "value": line}
            for line in (board_lines or [])]


def board_lines_to_text(board_lines: list) -> str:
    """Join board_lines into a single string for the renderer's text field."""
    return "\n".join(str(line) for line in (board_lines or []) if line)


def is_formula_line(line: str) -> bool:
    return classify_line(line) in ("math", "chemistry")


if __name__ == "__main__":
    samples = [
        "Given:",
        "u = 20 m/s",
        "v² = u² + 2as",
        "0² = 20² - 2 × 10 × H",
        "H₂SO₄",
        "CO₂",
        "H = 20 m",
        "Correct option: B",
        "Formula:",
        "E = mc²",
    ]
    for s in samples:
        print(f"  {classify_line(s):12s} | {s}")
