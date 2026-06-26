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


def _contains_devanagari(text):
    """True if the text contains any Devanagari (Hindi) characters."""
    return any("\u0900" <= ch <= "\u097f" for ch in (text or ""))


def detect_language(question_text, transcript_data=None):
    """
    Decide the working language from the question (and transcript) text.

    Returns a short code: "hi" for Hindi/Devanagari content, else "en".
    """
    if _contains_devanagari(question_text):
        return "hi"
    if transcript_data and _contains_devanagari(transcript_data.get("text", "")):
        return "hi"
    return "en"


def _build_prompt(transcript_data, question_text, language="en"):
    """Build the LLM prompt for semantic teacher action generation.

    The prompt is subject- and language-agnostic: it works for a math problem
    written in English just as well as for a Hindi biology MCQ. The teacher's
    written solution lines are produced in the SAME language and script as the
    question, so a Hindi question yields a Hindi (Devanagari) solution.
    """
    segments_text = []
    for seg in transcript_data["segments"]:
        segments_text.append(
            f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text'].strip()}"
        )

    if language == "hi":
        lang_rules = (
            "- The QUESTION and TRANSCRIPT are in HINDI. Write all `text` fields in"
            " HINDI (Devanagari script), exactly as a teacher would write on the"
            " board. Do NOT translate to English and do NOT romanise.\n"
            "- Keep tokens that are genuinely Latin in the question (hCG, HPL, DNA,"
            " ATP, FSH, LH, GnRH) in Latin.\n"
            "- `target` fields MUST be an exact Hindi substring copied from the"
            " QUESTION/options above (so it can be located on the image)."
        )
        examples = (
            "EXAMPLES of good notes/meanings (Hindi). Use '=' or ':' to relate"
            " terms, NOT arrow characters:\n"
            "  {\"action\":\"annotate_word\",\"target\":\"सगर्भता\",\"text\":\"गर्भावस्था\"}\n"
            "  {\"action\":\"write_note\",\"text\":\"hCG = कॉर्पस ल्यूटियम\"}\n"
            "  {\"action\":\"write_note\",\"text\":\"अपरा = अंतःस्रावी ग्रंथि\"}\n"
        )
    else:
        lang_rules = (
            "- Write `text` fields in the SAME language as the question.\n"
            "- For a MATH problem a `write_note` may be an equation; use plain"
            " subscripts (x2, x1) and a normal hyphen for minus, e.g."
            " \"d = √((x2-x1)² + (y2-y1)²)\".\n"
            "- `target` fields MUST be an exact substring of the QUESTION/options."
        )
        examples = ""

    return f"""You are annotating an educational video where a teacher solves a question on a board, writing and explaining as they go. Watch the AUDIO TRANSCRIPT (with timestamps) and the QUESTION, and output a JSON array of timed teacher actions that mirror what the teacher actually SAYS — nothing more.

QUESTION (from OCR):
{question_text}

AUDIO TRANSCRIPT (timestamps in seconds):
{chr(10).join(segments_text)}

Output a JSON array. Each element is ONE action with a `time` (float seconds) set to WHEN the teacher says that thing in the transcript:

1. `underline_existing` — underline an important word/phrase already printed in the question or options.
   {{"time": <float>, "action": "underline_existing", "target": "<exact substring from the question>"}}

2. `circle_word` — draw a circle around a key term, diagram blank, or option letter.
   {{"time": <float>, "action": "circle_word", "target": "<exact word/letter from question, e.g. '(A)' or 'GnRH'>"}}

3. `cross_out_word` — strike out an incorrect word inside an option, or an incorrect option letter.
   {{"time": <float>, "action": "cross_out_word", "target": "<exact substring/letter to cross out, e.g. 'सर्टोली कोशिकाएँ', 'A'>"}}

4. `draw_arrow` — draw an arrow connecting two terms/notes.
   {{"time": <float>, "action": "draw_arrow", "start_target": "<source word>", "end_target": "<destination word>"}}

5. `fill_placeholder` — write the correct term inside a diagram/flowchart blank.
   {{"time": <float>, "action": "fill_placeholder", "label": "<placeholder letter, e.g. 'A'>", "text": "<text to write>"}}

6. `annotate_word` — write a short explanation next to a word on the slide.
   {{"time": <float>, "action": "annotate_word", "target": "<exact word from question>", "text": "<short meaning, 1-3 words>"}}

7. `write_note` — write a short free working note or summary in the empty space.
   {{"time": <float>, "action": "write_note", "text": "<short note>"}}

8. `mark_answer` — mark the correct option letter.
   {{"time": <float>, "action": "mark_answer", "target": "<option letter, e.g. 'C'>"}}

{examples}
CRITICAL RULES:
- ONLY create an action for something the teacher ACTUALLY says/explains in the transcript. Do NOT invent extra steps.
- Set each `time` to the transcript timestamp where it is spoken, so writing stays in sync with the audio. Keep actions in increasing time order, spaced >= 1.2s apart.
- The text you write should be CORRECT for the subject even if the transcript is garbled — use your own knowledge.
- Option evaluation: The teacher evaluates each option in order. For incorrect options, cross out the incorrect term and the option letter. For the correct option, use `mark_answer` on the option letter.
{lang_rules}
- Return ONLY the raw JSON array. No markdown fences, no explanations.
"""


# Models tried in order. The free tier for some models (e.g. gemini-2.0-flash)
# is periodically disabled (HTTP 429, "limit: 0"); when that happens we fall
# through to the next model rather than dropping to the rule-based fallback.
GEMINI_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
)


def generate_with_llm(transcript_data, question_text, language="en"):
    """Use Google Gemini API to generate intelligent, question-agnostic annotations."""
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    prompt = _build_prompt(transcript_data, question_text, language)

    response = None
    last_err = None
    for model in GEMINI_MODELS:
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            print(f"    (model: {model})")
            break
        except Exception as e:
            last_err = e
            print(f"    model {model} unavailable, trying next...")
    if response is None:
        raise RuntimeError(f"All Gemini models failed; last error: {last_err}")

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
    """DISABLED — this used to return a hardcoded distance-formula (physics)
    annotation list REGARDLESS of the actual question, so a biology question whose
    LLM path failed could silently render a wrong-subject board. Refusing is strictly
    safer than rendering a confidently-wrong video. The real generalising path is
    ``generate_with_llm`` (multimodal Gemini upstream); this stub now raises so the
    pipeline aborts loudly instead of producing unrelated annotations.
    """
    raise RuntimeError(
        "Rule-based annotation fallback is disabled: it emitted hardcoded "
        "distance-formula annotations unrelated to the question (wrong-subject "
        "board). Provide GEMINI_API_KEY / fix the multimodal path so annotations "
        "are generated from THIS question."
    )


def generate_annotations(transcript_path, question_text, output_path, *args, **kwargs):
    """
    Generate annotations from transcript.

    Tries Gemini API first (generalises to any question).
    Falls back to rule-based if no API key is set.
    """
    # Question 7 Zoology Flowchart Slide Override
    is_q7 = (
        "शुक्रजनन" in question_text or 
        "GnRH" in question_text or 
        "ICSH" in question_text or 
        "सर्टोली" in question_text
    )
    # Hardcoded "Question 7" override DISABLED: the trigger above fires on ANY question
    # merely containing शुक्रजनन/GnRH/ICSH/सर्टोली (e.g. q40 contains शुक्रजनन), which would
    # paste q7's GnRH-axis flowchart onto an unrelated question — a wrong-subject board.
    # The whisper fallback now always uses the generalising LLM path below.
    if False:
        print("  [Override] Question 7 detected. Using hand-tuned perfect annotations.")
        q7_annotations = [
          { "time": 3.5, "action": "underline_existing", "target": "शुक्रजनन" },
          { "time": 18.0, "action": "circle_word", "target": "GnRH" },
          { 
            "time": 21.0, 
            "action": "write_note", 
            "text": "हाइपोथैलेमस", 
            "box": [640, 100, 800, 140],
            "arrow_params": [160, 140, 630, 120] 
          },
          { "time": 38.0, "action": "circle_word", "target": "LH" },
          { "time": 42.0, "action": "fill_placeholder", "label": "B", "text": "लीडिंग कोशिका" },
          { "time": 46.0, "action": "circle_word", "target": "एंड्रोजन" },
          { "time": 49.5, "action": "circle_word", "target": "शुक्राणुप्रसू का निर्माण" },
          { 
            "time": 56.0, 
            "action": "write_note", 
            "text": "पीयूष ग्रंथि", 
            "box": [290, 175, 410, 215],
            "arrow_params": [445, 220, 360, 195] 
          },
          { "time": 60.0, "action": "fill_placeholder", "label": "A", "text": "FSH" },
          { "time": 64.0, "action": "fill_placeholder", "label": "C", "text": "सर्टोली कोशिका" },
          { "time": 72.0, "action": "fill_placeholder", "label": "D", "text": "शुक्राणुजनन" },
          { 
            "time": 76.0, 
            "action": "write_note", 
            "text": "शुक्राणुजनन / शुक्राणुअंतरण", 
            "box": [500, 365, 800, 410],
            "arrow_params": [450, 430, 495, 385] 
          },
          { "time": 85.0, "action": "circle_word", "target": "सर्टोली कोशिकाएँ" },
          { "time": 89.0, "action": "cross_out_word", "target": "सर्टोली कोशिकाएँ" },
          { "time": 92.0, "action": "cross_out_word", "target": "(A)" },
          { "time": 114.0, "action": "circle_word", "target": "ICSH" },
          { "time": 118.0, "action": "cross_out_word", "target": "ICSH" },
          { "time": 121.0, "action": "cross_out_word", "target": "(B)" },
          { "time": 135.0, "action": "mark_answer", "target": "C" },
          { "time": 146.0, "action": "cross_out_word", "target": "ICSH" },
          { "time": 150.0, "action": "cross_out_word", "target": "(D)" },
          { 
            "time": 153.0, 
            "action": "write_note", 
            "text": "A = FSH\nB = लीडिंग कोशिका\nC = सर्टोली कोशिका\nD = शुक्राणुजनन", 
            "box": [680, 480, 920, 620] 
          }
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(q7_annotations, f, indent=2, ensure_ascii=False)
        return q7_annotations
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    language = detect_language(question_text, transcript_data)
    print(f"  Detected working language: {language}")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No GEMINI_API_KEY/GOOGLE_API_KEY set and the rule-based fallback is "
            "disabled (it emitted wrong-subject distance-formula annotations). Set the "
            "key so annotations come from THIS question, or fix the multimodal path."
        )
    try:
        print("  Using Gemini API for smart annotation generation...")
        annotations = generate_with_llm(transcript_data, question_text, language)
        print(f"  Generated {len(annotations)} annotations via LLM")
    except Exception as e:
        raise RuntimeError(
            f"LLM annotation generation failed ({e}); refusing to fall back to "
            f"hardcoded wrong-subject annotations. Fix the API/LLM path and retry."
        ) from e

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
