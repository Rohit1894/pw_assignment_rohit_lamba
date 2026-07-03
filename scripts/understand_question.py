#!/usr/bin/env python3
"""Understand the question image: Gemini Vision (semantics) + EasyOCR (geometry).

Gemini reads the ORIGINAL image (full resolution → best comprehension) and
returns structured semantics: subject, question type, text, options, given
values. EasyOCR runs on the COMPOSED CANVAS so every text box / option marker
is already in canvas coordinates for the renderer to target.

Output: output/question_understanding.json
"""

import json
import os
import sys

from gemini_utils import call_gemini_json, image_part

VALID_TYPES = {"simple_mcq", "numerical_mcq", "diagram", "flowchart",
               "matching", "assertion_reason"}

_PROMPT = """You are analysing a school/competitive-exam question image
(physics / chemistry / maths / biology / general, in English, Hindi or mixed).
Return ONLY a JSON object with EXACTLY these keys:

{
  "language": "english" | "hindi" | "hinglish",
  "detected_script": "english" | "hindi" | "mixed",
  "subject": "physics" | "chemistry" | "maths" | "biology" | "general",
  "question_type": "simple_mcq" | "numerical_mcq" | "diagram" | "flowchart" | "matching" | "assertion_reason",
  "question_text": "<the full question stem, transcribed exactly>",
  "options": [{"label": "A", "text": "..."}, ...],
  "given_values": ["<each given quantity with symbol, value and unit, e.g. 'u = 20 m/s'>", ...],
  "unknown": "<what the question asks for, e.g. 'maximum height H'>"
}

Rules:
- question_type: use "numerical_mcq" when solving needs calculation;
  "simple_mcq" for theory/recall MCQs; "matching" for List-I/List-II;
  "assertion_reason" for assertion-reason; "flowchart" when the image shows a
  flowchart with blanks; "diagram" when a printed figure must be read/labelled.
- Transcribe mathematical notation faithfully (superscripts, units, fractions).
- If there are no options, return "options": [].
- given_values / unknown: empty list / empty string for pure theory questions.
Return the raw JSON object only."""


def understand_question(image_path, layout=None, language="hinglish",
                        output_path="output/question_understanding.json",
                        canvas_ocr=None):
    """Run Gemini Vision + EasyOCR and save question_understanding.json.

    `canvas_ocr` is the (full_text, option_positions, question_bbox,
    enriched_ocr) tuple from extract_question_info() run on the composed
    canvas; pass it in so OCR runs once for the whole pipeline. If omitted,
    OCR runs here on the canvas image from `layout` (or the original image).
    """
    if canvas_ocr is None:
        from ocr_question import extract_question_info
        ocr_target = (layout or {}).get("canvas_image") or image_path
        print(f"  Running EasyOCR on {ocr_target}...")
        canvas_ocr = extract_question_info(ocr_target)
    full_text, option_positions, question_bbox, enriched_ocr = canvas_ocr

    data = call_gemini_json(
        [image_part(image_path), _PROMPT],
        temperature=0.1, label="Question understanding")

    if not isinstance(data, dict):
        raise RuntimeError("Question understanding: Gemini returned a non-object")

    # ── Validate / normalise the model output ───────────────────────────
    qtype = str(data.get("question_type", "")).strip().lower()
    if qtype not in VALID_TYPES:
        # Cheap fallback classification: options + digits → numerical.
        qtype = "numerical_mcq" if any(ch.isdigit() for ch in data.get("question_text", "")) \
            else "simple_mcq"
        print(f"  question_type invalid; falling back to '{qtype}'")
    subject = str(data.get("subject", "general")).strip().lower()
    if subject not in {"physics", "chemistry", "maths", "biology", "general"}:
        subject = "general"
    question_text = str(data.get("question_text") or "").strip()
    if not question_text:
        question_text = full_text.strip()
        if not question_text:
            raise RuntimeError("Question understanding: no question text from "
                               "Gemini or OCR — cannot proceed")
        print("  Gemini question_text empty; using OCR text")

    options = []
    for o in (data.get("options") or []):
        if isinstance(o, dict) and o.get("label"):
            options.append({"label": str(o["label"]).strip().upper()[:1],
                            "text": str(o.get("text") or "").strip()})

    ocr_boxes = []
    for el in (enriched_ocr.get("elements") or []):
        b = el.get("bounds")
        if b:
            ocr_boxes.append({"text": el.get("text", ""), "bounds": list(b)})

    understanding = {
        "language": language,
        "detected_script": str(data.get("detected_script", "english")).lower(),
        "subject": subject,
        "question_type": qtype,
        "question_text": question_text,
        "options": options,
        "given_values": [str(g) for g in (data.get("given_values") or [])],
        "unknown": str(data.get("unknown") or ""),
        "ocr_boxes": ocr_boxes,
        "option_positions": {k: v for k, v in (option_positions or {}).items()},
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(understanding, f, indent=2, ensure_ascii=False)
    print(f"  Understanding: {subject}/{qtype}, {len(options)} options, "
          f"{len(understanding['given_values'])} given values -> {output_path}")
    return understanding


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "input/question.png"
    understand_question(image)
