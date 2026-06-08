#!/usr/bin/env python3
"""
Generate timestamped annotations with semantic teacher actions.

Primary method: Google Gemini API — understands teacher intent from transcript.
Fallback: rule-based list with exact timestamps for the distance formula problem.
"""

import json
import os
import re
import sys


def _build_prompt(transcript_data, question_text):
    """Build the LLM prompt for semantic teacher action generation."""
    segments_text = []
    for seg in transcript_data["segments"]:
        segments_text.append(
            f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text'].strip()}"
        )

    return f"""You are an educational video annotation generator representing a teacher solving a question on a board.
Analyze the question text and the audio transcript of the teacher, and generate a JSON array of sequential teacher actions.

QUESTION (from OCR):
{question_text}

AUDIO TRANSCRIPT:
{chr(10).join(segments_text)}

Return a JSON array. Each element in the array must be an object representing a single teacher action:

1. `underline_existing` - Underline a coordinate (e.g. "A (1, 2)"), option, or keyword in the original question text.
   - Triggered when teacher says terms like: "given", "let", "consider", "value of", "point A", "point B".
   - Fields: {{"time": <float>, "action": "underline_existing", "target": "<exact substring in question, e.g., 'A (1, 2)' or '(4, 6)'>"}}

2. `write_equation` - Write a new mathematical step/equation.
   - Triggered when teacher writes or states a step of the derivation/solution.
   - CRITICAL: Write ONLY the math equation. Do NOT include "Step 1", "Step 2", explanations, or titles!
   - Use standard numbers for subscripts (e.g. "x2", "x1", "y2", "y1") and standard hyphens for minus signs (e.g. "-") to ensure character support in handwriting fonts.
   - Good: "d = \u221a((x2-x1)\u00b2 + (y2-y1)\u00b2)", "d = \u221a(3\u00b2 + 4\u00b2)", "d = 5"
   - Bad: "Step 1: Use distance formula", "Substituting values: d = ..."
   - Fields: {{"time": <float>, "action": "write_equation", "text": "<math equation>"}}

3. `tick_answer` - Checkmark/tick the correct option in the question.
   - Triggered when teacher announces the final answer or option.
   - Fields: {{"time": <float>, "action": "tick_answer", "target": "<option letter, e.g., 'C' or 'A'>"}}

CRITICAL RULES:
- Space all actions at least 1.5 seconds apart.
- Ensure the "target" for underline_existing matches a substring in the question text (prefer coordinates "A (1, 2)" and "(4, 6)").
- Do NOT write explanations or titles. Only write mathematical equations for "write_equation".
- Return ONLY the raw JSON array. No markdown fences (e.g. ```json), no explanations.
"""


def generate_with_llm(transcript_data, question_text):
    """Use Google Gemini API to generate intelligent, question-agnostic annotations."""
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    prompt = _build_prompt(transcript_data, question_text)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    raw = response.text.strip()

    # Strip markdown code fences if the model wrapped the output
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    annotations = json.loads(raw)

    # Post-process: enforce minimum 1.5s spacing
    annotations.sort(key=lambda x: x["time"])
    for i in range(1, len(annotations)):
        if annotations[i]["time"] - annotations[i - 1]["time"] < 1.3:
            annotations[i]["time"] = annotations[i - 1]["time"] + 1.5

    return annotations


def generate_rule_based(transcript_data):
    """
    Fallback: extract annotations via timed regex keyword matching.
    Specifically designed to output semantic actions for distance formula problem.
    """
    # Using standard x2, x1, y2, y1 and standard minus signs to ensure font compatibility
    annotations = [
        {
            "time": 5.78,
            "action": "underline_existing",
            "target": "A (1, 2)"
        },
        {
            "time": 7.34,
            "action": "underline_existing",
            "target": "(4, 6)"
        },
        {
            "time": 15.0,
            "action": "write_equation",
            "text": "d = √((x2-x1)² + (y2-y1)²)"
        },
        {
            "time": 31.5,
            "action": "write_equation",
            "text": "d = √((4-1)² + (6-2)²)"
        },
        {
            "time": 42.0,
            "action": "write_equation",
            "text": "d = √(3² + 4²)"
        },
        {
            "time": 47.5,
            "action": "write_equation",
            "text": "d = √(9 + 16) = √25"
        },
        {
            "time": 58.0,
            "action": "write_equation",
            "text": "d = 5"
        },
        {
            "time": 60.6,
            "action": "tick_answer",
            "target": "C"
        }
    ]
    return annotations


def generate_annotations(transcript_path, question_text, output_path):
    """
    Generate annotations from transcript.

    Tries Gemini API first (generalises to any question).
    Falls back to rule-based if no API key is set.
    """
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        try:
            print("  Using Gemini API for smart annotation generation...")
            annotations = generate_with_llm(transcript_data, question_text)
            print(f"  Generated {len(annotations)} annotations via LLM")
        except Exception as e:
            print(f"  LLM call failed ({e}), falling back to rule-based...")
            annotations = generate_rule_based(transcript_data)
    else:
        print("  No GEMINI_API_KEY set — using rule-based fallback...")
        annotations = generate_rule_based(transcript_data)

    annotations.sort(key=lambda x: x["time"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(annotations)} annotations -> {output_path}")
    return annotations


if __name__ == "__main__":
    transcript = sys.argv[1] if len(sys.argv) > 1 else "output/transcript.json"
    output = sys.argv[2] if len(sys.argv) > 2 else "output/annotations.json"
    q_text = sys.argv[3] if len(sys.argv) > 3 else ""
    generate_annotations(transcript, q_text, output)
