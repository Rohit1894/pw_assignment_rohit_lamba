"""Shared constants for the render package: ink palette, action sets, and
sub/superscript glyph maps. Extracted verbatim from render_video.py
(Step 1 refactor) as the single source of truth. Dependency-free leaf."""

# ── Colour palette ──────────────────────────────────────────────────────────
PEN_COLOR = (0, 0, 0)                             # black pen style


PEN_WIDTH = 3                                     # marker width


# Meaningful accent for the CORRECT answer (green), so the student instantly
# reads green = correct vs the base ink (red) used for emphasis/wrong marks.
ANSWER_INK = (22, 132, 64)


# Distinct teal for matching-table connectors, so the linking strokes read as
# "this pairs with that" rather than emphasis (red) or the answer (green).
MATCH_INK = (13, 115, 122)


# Actions that write new lines of the worked solution onto the board.
WRITE_ACTIONS = ("write_equation", "write_text", "write_step")


# ── Sub/superscript glyph maps ──────────────────────────────────────────────
# Unicode sub/superscripts are mapped to a normal base character drawn smaller
# and shifted, so they keep the handwriting look instead of relying on the font
# actually owning the precomposed glyph. DIGITS, SIGNS and LETTERS are covered:
# letters matter for science (e.g. the physics answer v = rᵃρᵇsᶜ, or eˣ, xᵢ).
_SUPERSCRIPT_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "⁼": "=", "⁽": "(", "⁾": ")",
    "ᵃ": "a", "ᵇ": "b", "ᶜ": "c", "ᵈ": "d", "ᵉ": "e",
    "ᶠ": "f", "ᵍ": "g", "ʰ": "h", "ⁱ": "i", "ʲ": "j",
    "ᵏ": "k", "ˡ": "l", "ᵐ": "m", "ⁿ": "n", "ᵒ": "o",
    "ᵖ": "p", "ʳ": "r", "ˢ": "s", "ᵗ": "t", "ᵘ": "u",
    "ᵛ": "v", "ʷ": "w", "ˣ": "x", "ʸ": "y", "ᶻ": "z",
}


_SUBSCRIPT_MAP = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "₊": "+", "₋": "-", "₌": "=", "₍": "(", "₎": ")",
    "ₐ": "a", "ₑ": "e", "ₕ": "h", "ᵢ": "i", "ⱼ": "j",
    "ₖ": "k", "ₗ": "l", "ₘ": "m", "ₙ": "n", "ₒ": "o",
    "ₚ": "p", "ᵣ": "r", "ₛ": "s", "ₜ": "t", "ᵤ": "u",
    "ᵥ": "v", "ₓ": "x",
}


# Free-form working-note actions: written in scattered empty space, optionally
# anchored to a question word (with a connecting arrow).
NOTE_ACTIONS = ("annotate_word", "write_note")

# Actions that mark the correct option with a solid line/slash.
ANSWER_ACTIONS = ("tick_answer", "mark_answer")

# Per-statement verdict (green tick = true / red cross = false) beside a statement,
# for assertion-reason and "how many statements are correct" questions.
VERDICT_ACTIONS = ("verdict_mark",)

# Every action that writes text and is revealed with the crop wipe.
TEXT_ACTIONS = ("annotate_word", "write_note", "fill_placeholder")
