#!/usr/bin/env python3
"""Generate the canonical solution — the source of truth for the whole video.

Solves the question cleanly (no animation/timing concerns here). The output
canonical_solution.json drives verification, the storyboard, and ultimately
what appears on the board.

Output: output/canonical_solution.json
"""

import json
import os
import sys

from gemini_utils import call_gemini_json, image_part

_TYPE_GUIDES = {
    "numerical_mcq": """This is a NUMERICAL question. solution_steps MUST follow this exact sequence:
  1. "Given"        — the known quantities, symbol = value unit, comma-separated
  2. "Formula"      — the governing formula/equation
  3. "Substitution" — the formula with numbers substituted
  4. "Calculation"  — the arithmetic simplification (one or two short lines)
  5. "Answer"       — the final result with unit""",
    "simple_mcq": """This is a THEORY MCQ. solution_steps MUST follow:
  1. "Key idea"        — the one concept that decides the answer
  2. "Option checking" — evaluate the plausible options briefly
  3. "Elimination"     — why the wrong options fail (short)
  4. "Answer"          — the correct option and why""",
    "matching": """This is a MATCHING question. solution_steps MUST follow:
  1. "Correct pairs" — each left item with its right match (e.g. "1-c, 2-a, 3-d, 4-b")
  2. "Explanation"   — one short reason per non-obvious pair
  3. "Answer"        — the option listing the correct combination""",
    "diagram": """This is a DIAGRAM/figure question. solution_steps MUST follow:
  1. "Labels"  — the important labels/blanks and what they are
  2. "Fill"    — the value/name for each blank or asked part
  3. "Answer"  — the correct option/final identification""",
    "flowchart": """This is a FLOWCHART question. solution_steps MUST follow:
  1. "Blanks"  — the lettered blanks and what belongs in each
  2. "Fill"    — the value for each blank
  3. "Answer"  — the correct option matching those fills""",
    "assertion_reason": """This is an ASSERTION-REASON question. solution_steps MUST follow:
  1. "Assertion" — is the assertion true/false, and why (short)
  2. "Reason"    — is the reason true/false, and why (short)
  3. "Link"      — does the reason correctly explain the assertion?
  4. "Answer"    — the correct option""",
}


def _build_prompt(understanding, language):
    qtype = understanding.get("question_type", "simple_mcq")
    guide = _TYPE_GUIDES.get(qtype, _TYPE_GUIDES["simple_mcq"])
    opts = "\n".join(f'  ({o["label"]}) {o["text"]}'
                     for o in understanding.get("options", []))
    return f"""You are an expert teacher solving an exam question correctly and cleanly.
The question image is attached. Structured understanding of it:

SUBJECT: {understanding.get('subject')}
QUESTION TYPE: {qtype}
QUESTION: {understanding.get('question_text')}
OPTIONS:
{opts if opts else '  (none)'}
GIVEN VALUES: {', '.join(understanding.get('given_values', [])) or '(none)'}
UNKNOWN: {understanding.get('unknown') or '(n/a)'}

Solve it. Double-check your arithmetic and reasoning before answering.

{guide}

Return ONLY a JSON object with EXACTLY these keys:
{{
  "language": "{language}",
  "question_type": "{qtype}",
  "final_answer": {{"option": "<letter like B, or empty string if no options>",
                    "text": "<the final answer value/statement, e.g. '20 m'>"}},
  "given": ["<given fact 1>", ...],
  "concept": "<the single key concept in one short sentence>",
  "formula": "<main formula used, or empty string>",
  "solution_steps": [
    {{
      "id": "step_1",
      "title": "<step title>",
      "board_lines": ["<line 1>", "<line 2>", ...]
    }},
    ...
  ]
}}

Board-text rules for solution_steps[].board_lines:
- An ARRAY of short strings — each string is ONE board line (under ~55 chars).
- Never a paragraph inside a single string. Split long content across lines.
- Never use "\\n", "/n", markdown, or JSON text inside the string values.
- Real math notation: "v² = u² + 2as", "0² = 20² − 2 × 10 × H", superscripts,
  ×, −, √, θ directly. For a stacked fraction use LaTeX \\frac{{num}}{{den}}.
- The final step's board_lines must state the final answer explicitly.
- Board writing must be ENGLISH only (no Devanagari).
Return the raw JSON object only."""


def solve_question(image_path, understanding, language="hinglish",
                   output_path="output/canonical_solution.json"):
    """Generate and save the canonical solution. Returns the solution dict."""
    data = call_gemini_json(
        [image_part(image_path), _build_prompt(understanding, language)],
        temperature=0.1, label="Canonical solve")

    if not isinstance(data, dict):
        raise RuntimeError("Solver: Gemini returned a non-object")

    # ── Validate the essentials; fail loudly rather than render a wrong board ──
    steps = data.get("solution_steps")
    if not isinstance(steps, list) or not steps:
        raise RuntimeError("Solver: no solution_steps in Gemini output")
    norm_steps = []
    for i, s in enumerate(steps, 1):
        if not isinstance(s, dict):
            continue
        # Support both new board_lines array and legacy flat text field
        raw_lines = s.get("board_lines")
        if isinstance(raw_lines, list):
            board_lines = [str(l).strip() for l in raw_lines if str(l).strip()]
        elif isinstance(raw_lines, str) and raw_lines.strip():
            board_lines = [raw_lines.strip()]
        else:
            # Fall back to legacy "text" field
            text = str(s.get("text") or "").strip()
            board_lines = [text] if text else []
        if not board_lines:
            continue
        # Sanitize: remove \n escapes, markdown artifacts
        clean = []
        for line in board_lines:
            line = line.replace("\\n", " ").replace("/n", " ").strip()
            line = line.lstrip("#*`").strip()
            if line:
                clean.append(line)
        if not clean:
            continue
        norm_steps.append({"id": s.get("id") or f"step_{i}",
                           "title": str(s.get("title") or f"Step {i}").strip(),
                           "board_lines": clean})
    if not norm_steps:
        raise RuntimeError("Solver: solution_steps were all empty")

    fa = data.get("final_answer") or {}
    if not isinstance(fa, dict):
        fa = {"option": "", "text": str(fa)}
    option = str(fa.get("option") or "").strip().upper()[:1]
    answer_text = str(fa.get("text") or "").strip()
    has_options = bool(understanding.get("options"))
    if has_options and not option:
        raise RuntimeError("Solver: question has options but no final option "
                           "was chosen — refusing to continue")
    if not answer_text and not option:
        raise RuntimeError("Solver: empty final answer")
    valid_letters = {o["label"] for o in understanding.get("options", [])}
    if option and valid_letters and option not in valid_letters:
        raise RuntimeError(f"Solver: answer option '{option}' is not one of the "
                           f"question's options {sorted(valid_letters)}")

    solution = {
        "language": language,
        "board_language": "english",
        "voice_language": "hinglish",
        "question_type": understanding.get("question_type", "simple_mcq"),
        "subject": understanding.get("subject", "general"),
        "final_answer": {"option": option, "text": answer_text},
        "given": [str(g) for g in (data.get("given") or [])],
        "concept": str(data.get("concept") or "").strip(),
        "formula": str(data.get("formula") or "").strip(),
        "solution_steps": norm_steps,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2, ensure_ascii=False)
    print(f"  Solution: answer = {option or '-'} '{answer_text}', "
          f"{len(norm_steps)} steps -> {output_path}")
    return solution


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "input/question.png"
    with open(sys.argv[2] if len(sys.argv) > 2
              else "output/question_understanding.json", encoding="utf-8") as f:
        und = json.load(f)
    solve_question(image, und)
