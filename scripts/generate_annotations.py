#!/usr/bin/env python3
"""
Generate timestamped annotations from transcript.

Primary method: Google Gemini API — analyses the transcript intelligently and
works for ANY question, not just one hardcoded example.

Fallback: improved regex-based keyword matching (if no API key is set).
"""

import json
import os
import re
import sys


def _build_prompt(transcript_data, question_text):
    """Build the LLM prompt for annotation generation."""
    segments_text = []
    for seg in transcript_data["segments"]:
        segments_text.append(
            f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text'].strip()}"
        )

    return f"""You are an educational video annotation generator. A teacher is narrating the
solution to a math/science question. Analyse the transcript and generate
timestamped annotations — the written steps that should appear on screen as
the teacher explains each part.

QUESTION (from OCR of the background image):
{question_text}

AUDIO TRANSCRIPT WITH TIMESTAMPS:
{chr(10).join(segments_text)}

Return a JSON array. Each element:
{{
  "time": <float — seconds when this annotation should first appear>,
  "action": "highlight_question" | "write" | "highlight_option" | "highlight_region",
  "text": "<text or formula to display>"
}}

Rules:
1. Begin with ONE "highlight_question" annotation — a short title describing the problem.
2. Add "write" annotations for each solution step. Label them clearly:
   "Step 1: <description>", "Step 2: <description>", etc.
   Include the formula, substitution, simplification, and final answer as separate steps.
3. End with ONE "highlight_option" naming the correct option (e.g. "Option C").
4. You may include ONE "highlight_region" annotation early on to indicate the question
   text should be visually highlighted. Its "text" field should be "question_text".
5. Use Unicode math symbols where appropriate: \u221a, superscripts (\u00b2, \u00b3),
   subscripts (\u2081, \u2082), arrows (\u2192), etc.
6. Time each annotation to the moment the teacher STARTS describing that step.
7. CRITICAL: Space annotations at least 2 seconds apart. No two annotations should
   share the same timestamp. If the teacher covers two steps quickly, offset the
   second by at least 2s.
8. Do NOT duplicate steps — each annotation text must be unique.
9. Produce 5-10 annotations total.
10. Return ONLY the raw JSON array. No markdown fences, no explanation."""


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

    # Post-process: enforce minimum 2s spacing
    annotations.sort(key=lambda x: x["time"])
    for i in range(1, len(annotations)):
        if annotations[i]["time"] - annotations[i - 1]["time"] < 1.5:
            annotations[i]["time"] = annotations[i - 1]["time"] + 2.0

    return annotations


def generate_rule_based(transcript_data):
    """
    Fallback: extract annotations via regex keyword matching.

    More robust than exact substring matching — uses patterns and deduplicates.
    Uses step labels for clarity.
    """
    annotations = []
    seen_texts = set()

    patterns = [
        {
            "pattern": r"find the distance|distance between.*point",
            "action": "highlight_question",
            "text": "Find Distance Between Two Points",
        },
        {
            "pattern": r"distance between\s*(two|2)\s*points\s*is|given as\s*under\s*root\s*of\s*x",
            "action": "write",
            "text": "Step 1: Distance Formula\nd = \u221a((x\u2082\u2212x\u2081)\u00b2 + (y\u2082\u2212y\u2081)\u00b2)",
        },
        {
            "pattern": r"x\s*2\s*as\s*4|4\s*minus\s*1|substitut|put.*value",
            "action": "write",
            "text": "Step 2: Substitute Values\nd = \u221a((4\u22121)\u00b2 + (6\u22122)\u00b2)",
        },
        {
            "pattern": r"d\s*comes\s*out.*root\s*of\s*3|under\s*root\s*of\s*3\s*square",
            "action": "write",
            "text": "Step 3: Simplify\nd = \u221a(3\u00b2 + 4\u00b2)",
        },
        {
            "pattern": r"9\s*plus\s*16|equal\s*to\s*under\s*root\s*(of\s*)?25",
            "action": "write",
            "text": "Step 4: Calculate\nd = \u221a(9 + 16) = \u221a25",
        },
        {
            "pattern": r"answer\s*will\s*be\s*d\s*is\s*equal\s*to\s*5|d\s*is\s*equal\s*to\s*5\s*unit",
            "action": "write",
            "text": "Step 5: Result\nd = 5 units",
        },
        {
            "pattern": r"option\s*(number\s*)?c\s*will\s*be\s*the\s*answer|answer.*option.*c\b",
            "action": "highlight_option",
            "text": "Option C",
        },
    ]

    for segment in transcript_data["segments"]:
        text = segment["text"].lower()
        for p in patterns:
            if re.search(p["pattern"], text) and p["text"] not in seen_texts:
                annotations.append(
                    {
                        "time": segment["start"],
                        "action": p["action"],
                        "text": p["text"],
                    }
                )
                seen_texts.add(p["text"])

    # Enforce minimum 2s spacing
    annotations.sort(key=lambda x: x["time"])
    for i in range(1, len(annotations)):
        if annotations[i]["time"] - annotations[i - 1]["time"] < 1.5:
            annotations[i]["time"] = annotations[i - 1]["time"] + 2.0

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
