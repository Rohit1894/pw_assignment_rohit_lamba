#!/usr/bin/env python3
"""
Multimodal annotation generator.

Instead of transcribing audio locally with Whisper (which drops/garbles Hindi
segments around intro music or silence) and then reasoning over text only, this
module sends the *audio* and the *slide image* DIRECTLY to Gemini in a single
multimodal call. Gemini listens to the teacher, looks at the slide, and returns
a timestamped JSON timeline of board actions that stays in sync with the audio.

It is fully question-agnostic: the model decides — per question — what to
underline, circle, cross out, explain, and where any diagram placeholders are.
"""

import json
import os
import re
import sys


# Models tried in order. The free tier for some models (e.g. gemini-2.0-flash)
# is periodically disabled (HTTP 429, "limit: 0"); when that happens we fall
# through to the next model. All of these accept audio + image input.
GEMINI_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
)

_AUDIO_MIME = {
    ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".wav": "audio/wav",
    ".ogg": "audio/ogg", ".flac": "audio/flac", ".aac": "audio/aac",
}
_IMAGE_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

# Bump when the prompt/schema changes so stale cached responses are not reused.
PROMPT_VERSION = "v21-multisubject"
_CACHE_DIR = os.path.join("output", ".mm_cache")


def _cache_key(audio_path, image_path, prompt):
    """Content hash of (audio + image + prompt) — identical inputs reuse the
    cached Gemini response instead of spending another API call/quota."""
    import hashlib
    h = hashlib.sha256()
    h.update(PROMPT_VERSION.encode())
    for p in (audio_path, image_path):
        with open(p, "rb") as f:
            h.update(f.read())
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()[:16]


def _build_prompt(question_text, image_width, image_height, duration_hint):
    """Prompt describing the action schema and the slide geometry."""
    dur = f"~{duration_hint:.0f}s" if duration_hint else "unknown"
    return f"""You are generating whiteboard annotations for an educational video in which a teacher
solves the attached question SLIDE while explaining it in the attached AUDIO
(Hindi or English). Your job: watch the slide, listen to the audio, and output a
JSON object with keys "question_type", "transcript" and "actions", that recreate
how the teacher annotates the slide, perfectly synced to WHEN they say each thing.

"transcript": a timestamped transcription of the WHOLE audio — an array of
{{"t": <seconds from start>, "text": "<the words spoken starting at t, in the
audio's own language/script>"}}. Emit a new entry roughly every 4-8 seconds so
the timeline densely covers the entire audio from 0 to the end. Timestamps MUST
be the real moment those words are spoken (listen carefully; do not bunch them
at the start). This transcript is used to time the actions precisely.

"actions": the array of timed board actions (schema below). Each action's time
MUST equal the transcript timestamp at which the teacher speaks about it.

SLIDE SIZE: {image_width} x {image_height} pixels. AUDIO LENGTH: {dur}.
OCR TEXT FROM THE SLIDE (for copying exact substrings):
{question_text}

Whiteboard Teaching Annotation Style Rules:
1. Underline key terms (precise, NOT generous): underline the important stem terms
   as the teacher reads them — 2 to 4 underlines per question, NEVER MORE THAN 4.
   Each underline MUST be SHORT (1-3 words), targeting the exact key phrase. NEVER
   underline a whole sentence or a whole statement line (that highlights nothing);
   for an assertion/कथन, underline only the 1-2 pivotal words inside it, not the
   entire statement. This is the reading-phase emphasis.
2. Circle only the ONE pivotal term: use `circle_word` to ring a SINGLE most
   important word/phrase (e.g. the crux term the question hinges on, or a specific
   wrong keyword inside an option). Ring that exact word — NEVER a whole option
   line or sentence. Use at most 1-2 circles in the whole question; prefer
   underlines over circles for ordinary emphasis.
3. Option Evaluation Flow (do this ONLY during evaluation, AFTER reading the
   question — never cross anything while still reading the stem):
   - For Multiple Choice Questions: the teacher evaluates the options one by one.
   - For an incorrect option/term: cross out the wrong keyword (`cross_out_word`)
     when explaining why it is wrong, then cross out that option letter. Keep these
     in the evaluation/conclusion part of the audio, not at the start.
   - For the correct option: use `mark_answer` with the option letter when the
     teacher declares it correct (it is ringed in green for you).
4. Diagrams, Figures & Blanks (works for ANY figure, not just lettered flowcharts):
   - Blanks to fill: a blank may be lettered ("(A)", "(B)", "(C)", "(D)"), an empty line ("__", "____"), an empty box, or a "?". For EVERY blank the teacher fills, emit `fill_placeholder` with the answer `text` AND a `box_2d` at the blank's exact location in the figure, at the moment the teacher reveals it. Include `label` only if the blank has a letter.
   - Labeled figures (e.g. a biology diagram of a cell/organ with parts): when the teacher points at or names a PART of the figure, target that part by its printed label text if it has one, otherwise give a `box_2d` over that part. Use `circle_word` to ring the part, `annotate_word` to write its name/meaning beside it, and `draw_arrow` to connect the part to your written note.
   - Always prefer giving `box_2d` for anything inside a figure (the printed text OCR can miss small in-figure labels); coordinates are normalised 0-1000 as [ymin,xmin,ymax,xmax].
5. Explanatory Notes & Arrow Connectors:
   - If the teacher explains the meaning of a word on the slide, write a short explanation note next to it. Use `annotate_word` with the `target` word (or a `box_2d`) and the explanation `text`.
   - If the teacher writes a free working note or equation on the empty board space, use `write_note`.
   - If the teacher links two things (a figure part to its label, a word to its definition, two terms), use `draw_arrow` with `start_target`/`end_target` (or `from_box_2d`/`to_box_2d`).
6. Spoken Language & Script:
   - Match the script of the slide and narration. If the teacher is speaking/writing in Hindi, all written `text` fields MUST be in Hindi (Devanagari script), e.g. "लीडिंग कोशिका" rather than "Leydig cell". Do not translate or romanize Hindi.
   - Keep standard Latin abbreviations/terms in Latin if they are printed that way (e.g. FSH, LH, GnRH, hCG, HPL).
7. Sync to Audio (CRITICAL):
   - You must listen to the audio track and map each action (underlining, circling, crossing out, writing notes, filling placeholders) to the EXACT timestamp (in seconds) in the audio timeline when the teacher actually speaks about that specific option, term, or concept.
   - DO NOT group all annotations at the beginning of the video. The annotations must be progressively generated across the entire audio duration (e.g., if the audio is 160 seconds long, options should be evaluated throughout the 160 seconds, and the final answer marked when the teacher announces it).

ACTION TYPES:
- underline_existing: underline a short key printed word/phrase on the slide.
  {{"time": <float>, "action": "underline_existing", "target": "<exact substring on the slide>"}}
  The target must be only the specific term/value being mentioned, never the
  whole question sentence or a long clause. GOOD: "distance", "(1, 2)", "(4, 6)".
  BAD: "Find the distance between the points A (1, 2) and (4, 6)."
- circle_word: draw a circle around a printed word/phrase, diagram blank, or option letter.
  {{"time": <float>, "action": "circle_word", "target": "<exact substring on the slide, e.g. '(A)', 'GnRH'>", "box_2d": [ymin, xmin, ymax, xmax]}}
  (box_2d optional, normalised 0-1000; include it for diagram/figure items OCR may miss.)
- cross_out_word: cross out an incorrect word inside an option or an incorrect option letter.
  {{"time": <float>, "action": "cross_out_word", "target": "<exact substring to cross out, e.g. 'सर्टोली कोशिकाएँ', 'A'>", "spoken_cue": "<the teacher's EXACT words at the MOMENT this option is REJECTED — the '...इसलिए यह गलत हो जाता है' conclusion, NOT when the option is first read aloud>"}}
  ALWAYS include spoken_cue, and set `time` to when the teacher CONCLUDES the option is
  wrong (after the reasoning), so the strike never appears before its explanation.
- draw_arrow: draw an arrow connecting two terms.
  {{"time": <float>, "action": "draw_arrow", "start_target": "<source word/phrase on the slide>", "end_target": "<destination word/phrase on the slide>"}}
- fill_placeholder: write the correct text for a diagram/flowchart blank.
  {{"time": <float>, "action": "fill_placeholder", "label": "<placeholder name, e.g. 'A'>", "text": "<text to write>", "box_2d": [ymin, xmin, ymax, xmax]}}
  ALWAYS include box_2d (normalised 0-1000, [ymin,xmin,ymax,xmax]) for the exact
  location of that blank INSIDE the figure, so the answer can be placed correctly.
- annotate_word: write a brief explanation/meaning next to a word on the slide.
  {{"time": <float>, "action": "annotate_word", "target": "<exact word on slide to explain>", "text": "<short meaning text>"}}
- write_note: write a short free-form explanation note or final summary note in empty space.
  {{"time": <float>, "action": "write_note", "text": "<note text>"}}
  For numerical/math questions, do NOT use write_note for formulas or equations;
  use write_step so the renderer writes it as part of the worked solution.
- write_step: write ONE line of a worked numerical solution. Emit several in order
  (given values, the formula, substitution, the result); they stack as a tidy
  column in the empty workspace, each revealed when the teacher says it.
  {{"time": <float>, "action": "write_step", "text": "<one line, e.g. 'v = u + at = 0 + 10*2 = 20 m/s'>"}}
  Keep each line short; put one equation/step per action.
- mark_answer: mark the correct option letter (it is ringed in green for you).
  {{"time": <float>, "action": "mark_answer", "target": "<correct option letter, e.g. 'C'>", "spoken_cue": "<the teacher's EXACT words when ANNOUNCING the answer, e.g. 'तो सही विकल्प नंबर बी है'>"}}
  ALWAYS include spoken_cue and set `time` to the CONCLUSION moment, so the answer is
  ringed at the end when it is announced — never early.
- verdict_mark: put a ✓ (true) or ✗ (false) BESIDE a statement, as the teacher
  judges each one true/false (for assertion-reason and "how many are correct"
  questions). The mark is drawn in the margin at the end of that statement's line.
  {{"time": <float>, "action": "verdict_mark", "target": "<the statement's leading
    text, e.g. 'अभिकथन A' or 'कथन I'>", "verdict": "true" | "false",
    "box_2d": [ymin,xmin,ymax,xmax]}}
  Recommended: give box_2d covering the WHOLE statement line so the ✓/✗ sits just
  after it. Use a green-tick (true) only for correct statements; ✗ for wrong ones.
- match_pair: draw a connecting line between an item in the LEFT list/column and
  its correct match in the RIGHT list/column (for "match the following" questions).
  {{"time": <float>, "action": "match_pair",
    "from_target": "<left item text, e.g. 'शीर्ष'>", "to_target": "<right item, e.g. 'एंजाइम'>",
    "from_box_2d": [ymin,xmin,ymax,xmax], "to_box_2d": [ymin,xmin,ymax,xmax]}}
  Give from_box_2d/to_box_2d at the two cells (recommended — table cells are easy
  to mislocate by text alone).
- draw_diagram: draw a clean SCHEMATIC (flowchart / process sequence / hormone
  axis / cycle) from labelled boxes + arrows, in empty board space, to TEACH the
  concept visually. The boxes and arrows are auto-laid-out and hand-drawn for you.
  {{"time": <float>, "action": "draw_diagram", "spoken_cue": "<phrase being said>",
    "diagram": {{
       "type": "flowchart" | "sequence" | "cycle",
       "title": "<short title in the SLIDE's own language (English or Hindi)>",
       "layout": "vertical" | "horizontal" | "snake",
       "nodes": [{{"id": "n1", "label": "हाइपोथैलेमस"}},
                 {{"id": "n2", "label": "पीयूष ग्रंथि", "highlight": true}}],
       "edges": [{{"from": "n1", "to": "n2", "label": "GnRH"}},
                 {{"from": "n4", "to": "n1", "label": "फीडबैक", "kind": "feedback"}}]
    }}}}
  Node labels MUST be SHORT (1-3 words). Use 3 to 6 nodes — NEVER MORE THAN 6 (a
  taller chain overflows the board); merge related steps into one node and drop
  outcome/leaf nodes rather than exceeding 6. Use "kind":"feedback"
  for a regulatory feedback edge (drawn as a return loop). For a "correct order /
  sequence" question, list the nodes in the correct order with "type":"sequence"
  and "layout":"snake". Edge "label" is optional (e.g. the hormone on an arrow).
  EDGES MUST CHAIN CONSECUTIVE NODES ONLY: n1->n2, n2->n3, n3->n4 (never skip a
  node like n1->n3 — a skipping edge has to detour around the node it jumps).
  Set "highlight": true on the ONE node that IS the answer (e.g. the acellular
  layer, the correct structure) — it is drawn in green so the diagram points at
  the answer. Highlight exactly one node, and only when a node is the answer.

DIAGRAMS (a schematic is useful for many questions, but must not clutter tables):
- EXCEPTION — questions that ALREADY SHOW A FIGURE on the slide (a printed
  flowchart with blanks (A)(B)(C)(D), or a labelled figure to identify):
  "flowchart_fill" and "diagram_label". For these you MUST NOT emit any
  `draw_diagram` — the printed figure IS the diagram; redrawing it just covers the
  board with a duplicate. Only FILL or LABEL the existing figure.
- Include one `draw_diagram` for non-table questions whose topic has a structure,
  order, axis, pathway, or set of related parts. For matching questions, do NOT
  emit `draw_diagram`; the printed table plus `match_pair` lines are the visual.
- For matching questions, the PRINTED TABLE is the primary visual. Do NOT write
  `annotate_word` notes inside or beside individual table cells. Use `match_pair`
  lines for the pairings and mark the answer. Do NOT add side notes unless the
  source question itself explicitly asks for a written rule.
- Always build the diagram from labelled nodes + arrows (a schematic) — never describe or
  request an image. Prefer ONE clear schematic over many scattered marks.
- Choose the diagram that teaches THIS question's answer — this works for EVERY
  subject (physics / chemistry / maths / biology), built from labelled boxes +
  arrows (NOT pictures of apparatus, structures or circuits):
    * Hormone/endocrine axis (biology) -> vertical flowchart; add a "feedback" edge
      if the teacher mentions feedback/regulation.
    * "Correct order / sequence / arrange" -> a sequence of nodes IN THE CORRECT
      ORDER (this is the visual answer).
    * Process / pathway / cause-effect -> a flowchart. E.g. a physics
      cause→effect or derivation chain, a chemistry reaction sequence
      (reactant → intermediate → product), or an ordered list of steps.
- Place the draw_diagram action at the moment the teacher BEGINS explaining that
  flow (it builds node-by-node across that spoken segment), and give it a
  spoken_cue. Use at most one or two diagrams per question.

QUESTION TYPE (classify first, then annotate accordingly):
Set a top-level "question_type" field to ONE of: "mcq", "assertion_reason",
"statement_count", "matching", "flowchart_fill", "diagram_label", "sequence",
"numerical", "other". Then follow the matching behaviour for that type:
- "mcq": underline the ONE key discriminating word in the stem (the exact property
  being asked, e.g. "अकोशिकीय"). Briefly evaluate each option as the teacher does;
  cross out a wrong option only AT THE MOMENT the teacher rejects it (never before,
  and never an option the teacher does not actually rule out). Then mark_answer the
  correct option, and add exactly ONE short write_note stating the RULE/REASON the
  answer is correct (a few words, in the slide's language — e.g. Hindi
  "जोना पेलुसिडा = ग्लाइकोप्रोटीन, कोशिका-रहित", or English "only (C) conserves momentum").
- "assertion_reason": use `verdict_mark` to judge the Assertion (A) and the Reason
  (R) each true/false as the teacher evaluates them (✓/✗ in the margin). Do NOT
  cross out the options — the answer follows from the two verdicts. Add exactly ONE
  short write_note stating the conclusion that selects the option, e.g.
  "A सत्य, R सत्य, पर R, A की व्याख्या नहीं → D" or "A सत्य, R असत्य → A". Then
  mark_answer that option. DO NOT write a separate note that merely restates a
  verdict (e.g. "कथन I: गलत" or "A असत्य") — the ✓/✗ mark already shows that; the
  conclusion note is the ONLY judgement note allowed. A `draw_diagram` of the
  underlying concept (e.g. the hormone axis) often makes the reason clear.
- "statement_count" ("how many of the following are correct"): `verdict_mark`
  EACH statement true/false as the teacher judges it, then `write_note` ONLY the
  final tally (e.g. "सही कथन = 2"), then mark_answer the matching option. Do NOT
  add a per-statement "कथन I: सही/गलत" note — the ✓/✗ marks already convey that.
- "matching" (List-I/List-II, Column-A/Column-B): as the teacher states each
  correct pairing, emit a `match_pair` connecting that left item to its right
  item. Keep the table clean: do NOT emit `annotate_word` for table terms and do
  NOT write notes over the table. Optionally include EXACTLY ONE small
  `draw_diagram` in the empty board space only when it genuinely teaches the
  shared structure/process behind the left column; otherwise omit it. Add at most
  ONE short standalone `write_note` only if the diagram and match lines do not
  already show the fact. Then mark_answer the option letter listing all correct
  pairs.
- "sequence" ("arrange in correct order"): use ONE `draw_diagram` with
  "type":"sequence" listing the nodes IN THE CORRECT ORDER (this is the visual
  answer), then mark_answer the option giving that order.
- "flowchart_fill": the slide shows a printed flowchart with blanks labelled
  (A),(B),(C),(D). Emit ONE `fill_placeholder` per blank, each with `label` set to
  that blank's letter and `text` set to its answer, synced to when the teacher says
  it. Do NOT emit a `draw_diagram` (the figure already shows the flowchart). Then
  mark_answer the option whose A/B/C/D values match your fills.
- "diagram_label": circle/annotate the named parts of the EXISTING printed figure;
  arrow to short explanations. Do NOT emit a `draw_diagram` — the figure is already
  there; only annotate it.
- "numerical": underline only the key givens/asked term, then work the solution
  as a sequence of `write_step` lines (given → formula → substitution → result),
  each synced to when the teacher says it; finally mark_answer the matching option.
  Every formula/equation must be a `write_step`, not `write_note`.

SUBJECT COVERAGE (the slide may be PHYSICS, CHEMISTRY, MATHS or BIOLOGY, in
English or Hindi — handle every subject the same way; NEVER assume biology):
- Use real mathematical/scientific notation in `text` — the renderer draws Greek
  letters, operators and sub/superscripts correctly, so do NOT spell them out as
  words. Write "x²", "v = u + at", "T⁻¹", "λ = h/p", "½mv²", "F ∝ 1/r²", "θ",
  "√", "∫", "Σ", "≤", "≠", "±", "→" directly.
- Physics / maths numerical example (write_step lines, English):
    {{"action":"write_step","text":"[v] = T⁻¹, [ρ] = ML⁻³, [s] = MT⁻²"}}
    {{"action":"write_step","text":"compare powers of M, L, T"}}
    {{"action":"write_step","text":"a = −3/2, b = −1/2, c = 1/2"}}
- Chemistry example: write formulae with subscripts and reaction arrows, e.g.
    {{"action":"write_step","text":"2H₂ + O₂ → 2H₂O"}}, "CH₄", "[H⁺]", "Δ".
- For an English MCQ, underline the key stem term, briefly evaluate each option,
  cross out a wrong option ONLY when the teacher rejects it, then mark_answer.
- All written `text` (notes, steps) MUST be in the SLIDE's own language: English
  notes for an English slide, Hindi for a Hindi slide. Keep Latin symbols/units
  exactly as printed (m/s, mol, N, J, Hz, °C).

WORKSPACE NOTES (make the board CLEAR, not crowded — this is MANDATORY):
- Include 2-4 short factual notes per question (write_note or annotate_word) — at
  least 2, and NEVER MORE THAN 4. COUNT THEM: write_note + annotate_word combined,
  INCLUDING the conclusion note, must total at most 4. A board with 6-8 notes is too
  cluttered for a student to follow; pick only the few facts that actually matter.
- NO REDUNDANCY: do not write two notes that make the same point (e.g. labelling
  three different options all as "कोशिकीय" — state the rule ONCE). The reason the
  CORRECT answer is right is the single most important note; keep at most one short
  note for the key wrong option, not one per option.
- DO NOT DUPLICATE THE DIAGRAM: if a fact is already shown as a node/arrow in your
  draw_diagram (e.g. the diagram shows FSH → सर्टोली कोशिका and LH → लीडिग कोशिका),
  do NOT also write it as a note. The diagram IS that explanation; a note repeating
  it just clutters the board. Notes should add facts the diagram does not show.
- During the EXPLANATION (not while merely reading the question), add these short
  factual notes in the empty board space that capture the KEY facts the teacher
  states — the things a student should jot: hormone names, the site/where, the
  cause, and especially the RULE or REASON behind the answer.
- Each note MUST be a CONCISE factual anchor of only a few words — NOT a sentence —
  written in the SLIDE's own language (English notes for an English slide).
  GOOD (Hindi biology): "LH शिखर - 14वाँ दिन", "FSH - सर्टोली कोशिका".
  GOOD (English, any subject): "KE = ½mv²", "frequency ∝ 1/√(LC)",
        "SN1 → racemic product", "F ∝ 1/r²", "discriminant < 0 → no real roots".
  BAD: a full explanatory sentence, or vague filler.
- Use `annotate_word` with a `target` when the fact explains a SPECIFIC slide term
  (an arrow will connect the note to that word); use `write_note` for a standalone
  fact or a one-line summary of why the answer is correct.
- Spread the notes across the explanation and give EACH a `spoken_cue` so it
  appears exactly when the teacher states that fact. Do not dump them all at once.
- Keep them few and meaningful (quality over quantity) — every note must earn its
  place; never write a note that does not carry a real fact.

SPOKEN CUE (CRITICAL FOR SYNC):
- For EVERY action, also include a field "spoken_cue": the SHORT exact phrase
  (about 3-8 words) the teacher SAYS in the AUDIO at the moment of that action,
  transcribed in the audio's own language/script exactly as you hear it
  (Devanagari for Hindi). This is the phrase that, when spoken, triggers the
  annotation — NOT the printed slide text and NOT the written meaning. It is used
  to time the action to the precise second the teacher utters it.
  Example: {{"time": 50.0, "action": "fill_placeholder", "label": "A",
            "text": "FSH", "box_2d": [..],
            "spoken_cue": "यहाँ पर ए के स्थान पर एफ एस एच आएगा"}}

OUTPUT REQUIREMENTS (your "actions" array MUST satisfy ALL of these):
1. 2-4 short underlines on key stem terms (reading phase).
2. 2-4 workspace notes (write_note / annotate_word) with concise facts — never more
   than 4, and none repeating another's point. EXCEPTION: for matching questions,
   use NO `annotate_word` table-cell notes and normally NO `write_note`.
3. EXACTLY ONE draw_diagram that visually teaches this question's answer. For a
   matching question this is FORBIDDEN: the printed table and match lines are the
   visual answer.
   NEVER for "flowchart_fill"/"diagram_label" — those already show a printed figure
   to fill/label, so a drawn diagram would just duplicate it.
4. The correct-answer mark (mark_answer) and, for assertion/count questions, a
   verdict_mark (✓/✗) on each statement.
5. Everything spread across the audio with a spoken_cue, never bunched at the start.
A timeline with no notes and no diagram is INCOMPLETE.

RULES:
- "time" is seconds from the START of the audio. Keep actions in increasing time order.
- Spacing between writing actions must be >= 1.5s.
- Return ONLY the raw JSON object {{"transcript": [...], "actions": [...]}}.
  No markdown fences, no comments.
"""


def _media_part(client, path, mime, inline_limit=15 * 1024 * 1024):
    """
    Return a content part for `path`. Small files are inlined as bytes; large
    files (e.g. long lectures) go through the Files API, waiting until the
    uploaded file is ACTIVE before use.
    """
    from google.genai import types
    if os.path.getsize(path) <= inline_limit:
        with open(path, "rb") as f:
            return types.Part.from_bytes(data=f.read(), mime_type=mime)

    import time
    uploaded = client.files.upload(file=path)
    for _ in range(30):
        state = getattr(getattr(uploaded, "state", None), "name", None) or getattr(uploaded, "state", None)
        if str(state) == "ACTIVE":
            break
        if str(state) == "FAILED":
            raise RuntimeError(f"Gemini file processing failed for {path}")
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    return uploaded


def _call_gemini_with_backoff(client, contents):
    """Try each model in turn; retry a model on transient (503/500/overload)
    errors with exponential backoff, and move to the next model on hard errors
    like 429 quota exhaustion (a different model may have separate quota)."""
    import time
    from google.genai import types
    response, last_err = None, None
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2,
    )
    for model in GEMINI_MODELS:
        for attempt in range(3):
            try:
                print(f"  Multimodal generation with {model} (audio + image)...")
                response = client.models.generate_content(
                    model=model, contents=contents, config=config)
                print(f"    (model: {model})")
                return response
            except Exception as e:
                last_err = e
                es = str(e)
                transient = ("503" in es or "500" in es or "UNAVAILABLE" in es
                             or "overloaded" in es.lower())
                if transient and attempt < 2:
                    wait = 4 * (attempt + 1)
                    print(f"    transient error, retrying {model} in {wait}s...")
                    time.sleep(wait)
                    continue
                print(f"    model {model} unavailable ({es[:70]}), trying next...")
                break
    raise RuntimeError(f"All Gemini models failed; last error: {last_err}")


def _parse_response(response):
    """Extract the first JSON value from a Gemini response (ignoring any trailing
    prose the model sometimes appends, which trips json.loads with 'Extra data')."""
    raw = (response.text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    start = min((i for i in (raw.find("{"), raw.find("[")) if i != -1), default=0)
    try:
        data, _ = json.JSONDecoder().raw_decode(raw[start:])
        return data
    except json.JSONDecodeError as e:
        os.makedirs("output", exist_ok=True)
        debug_path = os.path.join("output", "last_bad_gemini_response.txt")
        try:
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(raw)
        except Exception:
            debug_path = "<could not save>"
        raise RuntimeError(
            f"Gemini returned malformed JSON at line {e.lineno}, column {e.colno}. "
            f"Raw response saved to {debug_path}"
        ) from e


def generate_annotations_multimodal(audio_path, image_path, output_path,
                                    question_text="", image_size=(1280, 720),
                                    duration_hint=None, use_cache=True,
                                    refresh_cache=False):
    """
    Generate a synced annotation timeline from the audio + slide image via Gemini.

    Returns the parsed list of annotation dicts and writes them to output_path.
    Raises on failure so the caller can fall back to the Whisper text pipeline.

    Responses are cached by a hash of (audio + image + prompt): identical inputs
    reuse the cached result with no API call (saves quota, makes re-runs instant).
    """
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("No GEMINI_API_KEY / GOOGLE_API_KEY set for multimodal generation")

    W, H = image_size
    prompt = _build_prompt(question_text, W, H, duration_hint)

    # ── Cache lookup ────────────────────────────────────────────────────
    cache_file = None
    if use_cache:
        try:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            cache_file = os.path.join(_CACHE_DIR, _cache_key(audio_path, image_path, prompt) + ".json")
        except Exception:
            cache_file = None

    data = None
    if cache_file and os.path.exists(cache_file) and not refresh_cache:
        try:
            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)
            print(f"  Cache hit — reusing Gemini response ({os.path.basename(cache_file)}); no API call")
        except Exception:
            data = None

    # ── Live call (cache miss) ──────────────────────────────────────────
    if data is None:
        client = genai.Client(api_key=api_key)
        audio_mime = _AUDIO_MIME.get(os.path.splitext(audio_path)[1].lower(), "audio/mpeg")
        image_mime = _IMAGE_MIME.get(os.path.splitext(image_path)[1].lower(), "image/png")
        contents = [
            _media_part(client, audio_path, audio_mime),
            _media_part(client, image_path, image_mime),
            prompt,
        ]
        response = _call_gemini_with_backoff(client, contents)
        data = _parse_response(response)
        if cache_file:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
            except Exception:
                pass

    # Response is {"transcript": [...], "actions": [...]} (new) or a bare actions
    # array (older). Pull both apart and persist the transcript for sync.
    if isinstance(data, dict):
        annotations = data.get("actions") or data.get("annotations") or []
        gemini_transcript = data.get("transcript") or []
        # SALVAGE: Gemini sometimes INTERLEAVES the action objects directly into the
        # transcript array instead of emitting a separate "actions" key. That left
        # `actions` empty → 0 annotations → a completely blank video. Pull any
        # transcript entry that carries an "action" field back out as an annotation
        # (and drop it from the transcript so sync sees only spoken text).
        embedded = [e for e in gemini_transcript
                    if isinstance(e, dict) and e.get("action")]
        if embedded:
            gemini_transcript = [e for e in gemini_transcript
                                 if not (isinstance(e, dict) and e.get("action"))]
            annotations = (list(annotations) + embedded) if annotations else embedded
            print(f"  Salvaged {len(embedded)} action(s) interleaved in the transcript")
        qtype = data.get("question_type")
        if qtype:
            print(f"  Question type: {qtype}")
    else:
        annotations, gemini_transcript = data, []

    if gemini_transcript:
        gt_path = os.path.splitext(output_path)[0] + ".gtrans.json"
        try:
            with open(gt_path, "w", encoding="utf-8") as f:
                json.dump(gemini_transcript, f, indent=2, ensure_ascii=False)
            print(f"  Saved Gemini timestamped transcript ({len(gemini_transcript)} "
                  f"entries) -> {gt_path}")
        except Exception:
            pass

    # Shared timeline hygiene: order, spread (anti front-loading), space, clamp.
    from timing_utils import normalize_timeline
    annotations = normalize_timeline(annotations, duration_hint)

    # Canonicalise action names (tick_answer -> mark_answer, write_equation ->
    # write_step, ...) so the saved set uses the one clean schema. See action_schema.
    from action_schema import normalize_actions
    annotations = normalize_actions(annotations)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(annotations)} annotations -> {output_path}")

    # Persist the question type alongside the annotations so the pre-render
    # validation gate (validate_annotations.py) can check that the action set
    # matches the type. Sidecar, not inline, to keep annotations.json a plain list.
    try:
        meta_path = os.path.splitext(output_path)[0] + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"question_type": data.get("question_type")
                       if isinstance(data, dict) else None}, f, ensure_ascii=False)
    except Exception:
        pass
    return annotations


if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else "input/7.mp3"
    image = sys.argv[2] if len(sys.argv) > 2 else "output/analysis/pdf_pages/page_7.png"
    out = sys.argv[3] if len(sys.argv) > 3 else "output/annotations.json"
    generate_annotations_multimodal(audio, image, out)
