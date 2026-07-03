# PW Automated Annotation System

An automated pipeline that transforms a **question image** and **audio narration** into a fully annotated educational video — complete with character-by-character text animation, synced timestamps, and visual highlights for questions and answer options.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Pipeline Breakdown](#pipeline-breakdown)
  - [Step 1 — Audio Transcription](#step-1--audio-transcription-transcribepy)
  - [Step 2 — OCR Question Extraction](#step-2--ocr-question-extraction-ocr_questionpy)
  - [Step 3 — Annotation Generation](#step-3--annotation-generation-generate_annotationspy)
  - [Step 4 — Video Rendering](#step-4--video-rendering-render_videopy)
- [Image-only Hinglish video generation](#image-only-hinglish-video-generation---auto-audio)
- [Bonus: Rename Questions Utility](#bonus-rename-questions-utility)
- [Evaluation Harness](#evaluation-harness-scriptseval_corpuspy)
- [Output Files](#output-files)
- [Configuration](#configuration)
- [Tech Stack](#tech-stack)

---

## How It Works

```
input/question.png + input/narration.mp3
              │
              ▼
   ┌─────────────────────┐
   │  1. Transcribe Audio │  (Whisper) → word-level timestamps
   └──────────┬──────────┘
              ▼
   ┌─────────────────────┐
   │  2. OCR Question     │  (EasyOCR) → question text + option positions
   └──────────┬──────────┘
              ▼
   ┌─────────────────────┐
   │  3. Generate         │  (Gemini API / regex fallback)
   │     Annotations      │  → timestamped solution steps
   └──────────┬──────────┘
              ▼
   ┌─────────────────────┐
   │  4. Render Video     │  (PIL + MoviePy) → animated video with audio
   └──────────┬──────────┘
              ▼
        output/final.mp4
```

The system reads a static question image and a teacher's audio explanation, then automatically produces a polished instructional video where solution steps appear on screen in sync with the narration — simulating a handwritten solution.

The pipeline is **language- and subject-aware**: it works for an English math problem and equally for a **Hindi (Devanagari)** biology question, automatically detecting the script and writing the solution in the same language.

---

## Multilingual (Hindi) Support

The pipeline handles questions whose **audio narration, question image, and written solution are all in Hindi**, in addition to English.

**Run a Hindi question:**

```bash
python main.py \
  --image input/hindi_question.png \
  --audio input/hindi_narration.mp3 \
  --output output/hindi_final.mp4 \
  --transcript output/hindi_transcript.json \
  --annotations output/hindi_annotations.json \
  --whisper-model medium \
  --language hi \
  --ink red
```

**What changes per stage (and what stays automatic):**

| Stage | Hindi handling |
|-------|----------------|
| **OCR (runs first)** | EasyOCR always runs with `["hi", "en"]`, so it reads Devanagari **and** the Latin tokens that appear in Hindi science MCQs (e.g. `hCG`, `HPL`). Runs before transcription so the question's terms can prime Whisper. |
| **Transcribe** | Pass `--language hi` so Whisper emits **Devanagari Hindi** (not romanised/translated). The OCR'd question terms are passed as Whisper's `initial_prompt` to sharpen domain words and timing. Use `--whisper-model medium` (or `large`) — `base`/`small` are weak on Hindi. |
| **Annotations** | Language is auto-detected. The Gemini prompt is **audio-driven**: it only emits an action for something the teacher actually says, timestamped for sync, and writes meanings/notes in the question's script. Subject-agnostic (biology, math, …). |
| **Rendering** | Devanagari is drawn with the bundled **Kalam** handwriting font (falls back to Windows **Nirmala UI**), revealed **grapheme-cluster by cluster** so matras/conjuncts stay correct. The math `√`/subscript path is used only for non-Hindi equations. |

### Annotation engine: multimodal Gemini (default) vs Whisper

By default the pipeline uses **`--engine gemini`**: it sends the **audio + slide image directly to Gemini** in one multimodal call, and Gemini returns a **timestamped action timeline** synced to the narration. This avoids Whisper's tendency to drop/garble Hindi around intro music or silence, and lets the model *see* the slide (e.g. locate flowchart placeholders).

`--engine whisper` (or automatic fallback if Gemini is unavailable) uses the local Whisper transcription + text-only annotation path.

### Teacher-style annotation actions

Mirroring a real teacher's board, the model emits these timed actions (red ink by default, configurable via `--ink`):

| Action | What it does |
|--------|--------------|
| `underline_existing` | Solid underline under a key word/phrase in the question. |
| `circle_word` | **Hand-drawn ellipse** (with parametric noise) around a key term or diagram placeholder. Located via OCR text or Gemini `box_2d` coordinates. |
| `cross_out_word` | **Hand-drawn slash → X** over an incorrect term in an option. |
| `annotate_word` | Writes a word's **meaning beside/below it** in-place (e.g. सगर्भता → गर्भावस्था); if cramped, placed in free space with a **connecting arrow**. |
| `fill_placeholder` | Writes the answer term **next to a diagram blank** (A)/(B)/(C)/(D) with a connector arrow — for flowchart/figure questions. Blank positions are resolved **generically**, in priority order (see below). No per-question hardcoding. |
| `draw_arrow` | Hand-drawn arrow connecting two targets. |
| `write_note` | A short **working note in empty space** (e.g. `hCG = कॉर्पस ल्यूटियम`), including an optional multi-line **summary block** (`A = … / B = …`). |
| `mark_answer` | Solid diagonal line marking the correct option. |

Notes are placed by a **scatter engine** that keeps them in blank areas (never overlapping the printed question/options/diagram), with **dynamic font sizing**. Hindi text is **pre-rendered once and revealed with a left-to-right crop wipe**, so matras/conjuncts always shape correctly. Layout is seeded, so re-runs are stable but look hand-placed.

**Diagram blank resolution (generic, no hardcoding).** A flowchart blank's position is found by trying, in order:
1. **OCR full + band re-OCR** — locate `(A)`/`(B)`/… tokens; an upscaled crop of the diagram band recovers tiny labels the full-image pass misses.
2. **Targeted gap recovery** — if the detected labels form a sequence with a hole (e.g. `A, B, D` ⇒ `C` missing), re-OCR a tight, heavily-upscaled strip around each detected column and accept that *specific* letter even at low confidence (knowing which letter to look for makes a low-confidence hit safe). This recovered `(C)` at its true node in the Q7 flowchart.
3. **Geometric inference** — if a referenced blank is still unplaced, fit `cy(index)` through the found blanks (rows are monotonic), predict the column and **snap it to the nearest detected column** (robust to zig-zag layouts). Used only when the row-fit is clean.
4. **Gemini `box_2d`** — the model's vision coordinate, used last because it can hallucinate a blank far from the figure.

Candidate blanks are validated by **size** (a real `(A)` token is small) and a **watermark guard** (ignores the corner PW logo), so table labels/logos don't masquerade as fillable blanks.

**Timeline hygiene (engine-agnostic).** Whatever produced the actions, the final timeline is ordered, **stretched to fill the audio if the model front-loaded everything** (preserving intended order/pacing), minimally spaced, and **clamped so nothing is scheduled past the narration**.

**Generic figure understanding.** Diagram handling is not limited to lettered `(A)/(B)` flowchart blanks. The model can target **any part of any figure** (a labelled biology diagram, a non-lettered blank line/box, a `?`) by giving a `box_2d`, and the renderer **snaps that approximate coordinate onto the printed label it overlaps** (Gemini's vision boxes are roughly right but off by tens of pixels) — so circles, underlines, cross-outs and connector arrows land precisely on the real label. Genuine blanks (empty space) don't snap, so fills still write *beside* the blank. No diagrams are generated; the existing on-slide figure is annotated in place.

**Audio sync.** Action timing is re-derived from when things are actually said: Gemini returns a timestamped transcript of the whole audio (it transcribes Hindi far better than Whisper on noisy lecture audio), each action's `spoken_cue` is fuzzy-matched into that transcript, and the timeline is re-sorted to match — so annotations appear exactly when the teacher speaks them, not bunched at the start. Broken model timestamps are sanitised (kept in order, regridded), and a sanity gate falls back to even spacing if the transcript is too poor. Whisper word-timestamps are used as a fallback sync source. Disable with `--no-sync`.

**Worked numerical solutions.** For `numerical` questions the teacher's working is written out step-by-step (`write_step`): given values → formula → substitution → result, each line revealed in sync as it is spoken. The lines **stack as a tidy column in the largest empty region** and the font/spacing **auto-size (vertically and horizontally) so the whole solution fits** on screen. Steps may mix Latin/maths and Hindi (rendered via the same crop-reveal as notes).

**OCR-error tolerance.** Resolving an action's `target` to a box no longer needs an exact OCR match: text is normalised (case, punctuation, Devanagari danda, zero-width joiners) and scored by a blend of substring containment, token recall, and character ratio — so a small misread (e.g. `लीडिग` vs `लीडिंग`) still matches. A target whose printed phrase was **split across several OCR boxes** is recovered by merging consecutive same-line boxes. If nothing clears the bar, resolution returns nothing (the renderer skips rather than misplaces) and Gemini's `box_2d` (snapped) is the fallback.

**Question-type awareness.** The model first classifies the question (`mcq`, `assertion_reason`, `matching`, `flowchart_fill`, `diagram_label`, `numerical`) and annotates accordingly. Notably, **"match the following" questions draw real connector lines** (`match_pair`) from each List-I item to its correct List-II item as the teacher states the pairing — not just generic notes — then mark the option that lists all correct pairs.

**Reliability.** Gemini responses are **cached** by a hash of (audio + image + prompt) — identical inputs reuse the result with no API call (`--no-cache`/`--refresh-cache` to control). Transient errors retry with **exponential backoff**; quota errors fall through to the next model, then to the Whisper + rule-based path.

> **Note on fonts:** Hindi must be drawn as whole words/clusters — never one codepoint at a time — or matras and conjuncts break. The renderer handles this automatically. `fonts/Kalam-Regular.ttf` is committed with the project; no extra install is needed.

---

## Project Structure

```
PW-Automated-Annotation-System/
├── main.py                          # Entry point — runs the full pipeline (OCR → annotate → sync → validate → render)
├── run_new_question.py              # Render a new question reusing existing annotations (no Gemini)
├── requirements.txt                 # Pinned Python dependencies
│
├── scripts/
│   ├── transcribe.py                # Audio → timestamped transcript (Whisper)
│   ├── ocr_question.py              # Image → question text + option/placeholder boxes (EasyOCR)
│   ├── ocr_utils.py                 # OCR enrichment (element types, free-space regions)
│   ├── generate_annotations_multimodal.py  # PRIMARY: audio + slide → Gemini → timed actions
│   ├── generate_annotations.py      # Fallback: transcript (text) → Gemini annotations
│   ├── align_timeline.py            # Re-time actions to the moment each phrase is spoken
│   ├── timing_utils.py              # Timeline hygiene (order, spread, clamp to audio)
│   ├── action_schema.py             # Canonical action vocabulary (one source of truth)
│   ├── validate_annotations.py      # Pre-render validation gate (blocks bad/blank boards)
│   ├── validate.py                  # Pipeline validation harness (reference Hindi set)
│   ├── eval_corpus.py               # English multi-subject evaluation harness
│   ├── rename_questions.py          # Utility: bulk-rename images from ZIP + Excel metadata
│   ├── extract_pdf.py, render_pdf_pages.py, extract_*_frames.py   # dev/analysis helpers
│   ├── render_video.py              # Thin façade — re-exports render_video()
│   └── render/                      # Renderer package (split out of the old render_video.py)
│       ├── constants.py             #   ink palette, action sets, sub/superscript maps
│       ├── text_utils.py            #   script detection, grapheme split, wrap, math tokens
│       ├── fonts.py                 #   font location (bundled Kalam first), glyph fallback, sizing
│       ├── strokes.py               #   hand-drawn progressive pen primitives
│       ├── text_render.py           #   per-glyph draw, hand-drawn √ + stacked \frac, crop-reveal
│       ├── geometry.py              #   box overlap, slot finding, OCR/Gemini target-box resolution
│       ├── placeholders.py          #   diagram/flowchart blank inference
│       ├── matching.py              #   match-the-following connector routing
│       ├── verdicts.py              #   ✓/✗ verdict placement
│       ├── diagram.py               #   schematic diagram engine (flowchart / sequence / cycle)
│       ├── schedule.py              #   builds the timed, positioned draw schedule (the "brain")
│       └── frame.py                 #   renders each frame + assembles the video (render_video())
│
├── fonts/
│   ├── Kalam-Regular.ttf            # Bundled handwriting font — Hindi (Devanagari) + English (Latin/math)
│   └── Kalam-Bold.ttf               # Bold variant (titles / diagram headings)
│
├── input/                           # Question images + narration audio (e.g. question.png, narration.mp3)
├── output/                          # Generated artifacts: transcripts, annotations, videos, eval/validation reports
│
├── task02_brief.md                  # Writing style profile template (Task 02)
└── task03_explanation.md            # Explanation of the rename utility (Task 03)
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/PW-Automated-Annotation-System.git
cd PW-Automated-Annotation-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**

| Package          | Purpose                                      |
|------------------|----------------------------------------------|
| `openai-whisper` | Audio transcription with word-level timestamps |
| `easyocr`        | Optical character recognition on images       |
| `opencv-python`  | Image processing (EasyOCR dependency)         |
| `moviepy`        | Video composition and MP4 encoding            |
| `Pillow`         | Image drawing and text rendering              |
| `google-genai`   | Google Gemini API for smart annotation gen     |
| `numpy`          | Numerical operations for image arrays         |
| `openpyxl`       | Excel file parsing (rename utility)           |
| `pandas`         | DataFrame operations (rename utility)         |

### 3. Set up API key (optional but recommended)

For intelligent annotation generation that works with **any** question:

```bash
export GEMINI_API_KEY="your-google-gemini-api-key"

```

Without an API key, the system falls back to regex-based pattern matching (only works for specific hardcoded problem types).

### 4. Place your input files

```
input/question.png   ← your MCQ question image
input/narration.mp3  ← teacher's audio explanation
```

Once you have placed your files in the `input/` folder, run the pipeline command pointing to these paths:
```bash
python main.py --image input/question.png --audio input/narration.mp3 --output output/final.mp4
```

---

## Usage

### Run the full pipeline

```bash
python main.py
```

### Custom input/output paths

```bash
python main.py --image input/question.png --audio input/narration.mp3 --output output/final.mp4
```

### Skip re-transcription (reuse existing transcript)

```bash
python main.py --skip-transcribe
```

### Use a larger Whisper model for better accuracy

```bash
python main.py --whisper-model medium
```

### All CLI options

| Flag                | Default                  | Description                                    |
|---------------------|--------------------------|------------------------------------------------|
| `--image`           | `input/question.png`     | Path to the question background image          |
| `--audio`           | `input/narration.mp3`    | Path to the audio narration                    |
| `--output`          | `output/final.mp4`       | Output video path                              |
| `--transcript`      | `output/transcript.json` | Where to save/read the transcript              |
| `--annotations`     | `output/annotations.json`| Where to save generated annotations            |
| `--whisper-model`   | `base`                   | Whisper model size: tiny/base/small/medium/large |
| `--skip-transcribe` | `false`                  | Reuse existing transcript instead of re-transcribing |

### Running on New Inputs (Image & Audio)

To automate the whiteboard solving video generation for a completely **new question image** and **narration audio**:

#### 1. Setup your Google Gemini API Key
The system uses the Gemini API to analyze the question text and audio transcript dynamically to generate aligned solving annotations.
*   **On Windows (PowerShell)**:
    ```powershell
    $env:GEMINI_API_KEY="your_actual_gemini_api_key"
    ```
*   **On Windows (CMD)**:
    ```cmd
    set GEMINI_API_KEY=your_actual_gemini_api_key
    ```
*   **On Linux/macOS**:
    ```bash
    export GEMINI_API_KEY="your_actual_gemini_api_key"
    ```

#### 2. Run the command
If your Gemini API key is valid and has sufficient quota, simply copy your new question image and narration audio to the workspace, then run:
```bash
python main.py --image input/new_question.png --audio input/new_narration.mp3 --output output/new_final.mp4
```
*Note: Do not pass the `--skip-transcribe` flag since a new narration audio needs to be transcribed from scratch.*

#### 3. Handling API Key Rate Limits / Quota Exceeded (429 Fallback)
If your Gemini API key is rate-limited or exhausted:
1.  **Transcribe the new audio file**:
    ```bash
    python scripts/transcribe.py input/new_narration.mp3 output/new_transcript.json
    ```
2.  **Open the transcript file** (`output/new_transcript.json`), inspect what is said by the teacher, and write the timeline of solving steps inside a custom JSON file (e.g. `output/new_annotations.json`) using this schema:
    ```json
    [
      { "time": 10.0, "action": "underline_existing", "target": "coordinate_or_term" },
      { "time": 30.5, "action": "write_equation", "text": "y = mx + c" },
      { "time": 94.0, "action": "tick_answer", "target": "B" }
    ]
    ```
3.  **Compile the video** directly without calling the Gemini API using:
    ```bash
    python run_new_question.py
    ```
    *(You can update the image, audio, and annotations paths inside `run_new_question.py` as needed).*

---

## Pipeline Breakdown

### Step 1 — Audio Transcription (`transcribe.py`)

Converts the audio narration into a timestamped transcript using **OpenAI Whisper**.

- Loads the specified Whisper model (default: `base`)
- Transcribes with `word_timestamps=True` for precise timing
- Outputs JSON with segments and per-word timestamps

**Output format** (`output/transcript.json`):
```json
{
  "text": "Full transcript text...",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 7.34,
      "text": "Let us find the distance...",
      "words": [
        { "word": "Let", "start": 0.0, "end": 0.28, "probability": 0.89 },
        { "word": "us", "start": 0.28, "end": 0.52, "probability": 0.95 }
      ]
    }
  ]
}
```

### Step 2 — OCR Question Extraction (`ocr_question.py`)

Extracts text and option locations from the question image using **EasyOCR**.

- Scans the image for all text regions
- Identifies multiple-choice options (A, B, C, D) by matching patterns like `(a)`, `(b)`, etc.
- Computes bounding boxes for each option and the question region
- Returns: `(full_text, option_positions, question_bbox)`

**What it detects:**
- **Question text** — all OCR'd text concatenated
- **Option positions** — bounding box coordinates for each option (A/B/C/D)
- **Question bounding box** — region above the options

### Step 3 — Annotation Generation (`generate_annotations.py`)

Produces timestamped solution steps synced to the audio narration.

**Two modes:**

| Mode | When Used | How It Works |
|------|-----------|--------------|
| **Gemini API** (primary) | `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set | Sends transcript + question text to Gemini (tries `gemini-2.5-flash`, then `gemini-2.0-flash`, then `gemini-2.5-flash-lite` so a quota-exhausted model falls through to the next); generates intelligent step-by-step annotations for any question, in the question's language |
| **Regex fallback** | No API key available | Matches keywords in transcript against hardcoded patterns (limited to specific problem types) |

**Annotation types:**

| Action              | Description                          | Visual Style     |
|---------------------|--------------------------------------|------------------|
| `underline_existing`| Underline existing terms/coordinates | Jittery underline beneath OCR text |
| `write_equation` / `write_text` | Write a solution line (equation or, for Hindi/conceptual questions, a sentence) progressively | Black pen marker handwriting font (Kalam for Hindi) |
| `tick_answer`       | Select final correct option indicator| Diagonal slash crossing option indicator |

**Output format** (`output/annotations.json`):
```json
[
  { "time": 5.78, "action": "underline_existing", "target": "A (1, 2)" },
  { "time": 15.0, "action": "write_equation",     "text": "d = √((x₂−x₁)² + (y₂−y₁)²)" },
  { "time": 60.6, "action": "tick_answer",         "target": "C" }
]
```

### Step 4 — Video Rendering (`render_video.py`)

Composites the final annotated video using **Pillow** (drawing) and **MoviePy** (encoding).

`render_video.py` is a thin **façade** that re-exports `render_video()`; the renderer
itself lives in the **`scripts/render/` package** — focused modules for fonts, pen
strokes, text/equation rendering, geometry, the per-question-type resolvers
(`placeholders`, `matching`, `verdicts`, `diagram`), the scheduling "brain"
(`schedule.py`), and the frame compositor + video assembly (`frame.py`). See
[Project Structure](#project-structure) for the full module map.

**Canvas layout:**
All annotations are drawn directly on the original question image (resolution remains 1280x720). No bottom workspace panel or slide presentation is added, ensuring a natural board solving layout.

**Whiteboard drawing features:**
- **Handwriting Simulation**: Uses the **bundled Kalam handwriting font** for all annotations and math, so the board looks hand-written and **identical on every OS** (Windows / Linux / macOS). System handwriting fonts (`Ink Free`, `Segoe Print`, macOS Chalkboard) are only a fallback if the bundled font is unavailable.
- **Word-wise Progressive Reveal**: Reveals mathematical equations progressively token-by-token (word/symbol-wise) to simulate natural handwritten speed.
- **Proportional Underlining**: Computes the coordinates of targets dynamically and draws hand-drawn underlines exactly below coordinates (e.g. `A (1, 2)`).
- **Diagonal Option Slash**: Ticks the correct option letter indicator by drawing a hand-drawn diagonal slash crossing cleanly inside the option text box (e.g. `(C)`).
- **No Glow/Cursors**: Disables glows, colors, or highlighted headers; uses clean black ink marker style `(0, 0, 0)` for all drawing steps.
- **Radical, Fraction & Subscript Rendering** (`render/text_render.py`): Draws the square root (`√`) sign as smooth hand-drawn lines spanning its argument; **stacks fractions** written as `\frac{num}{den}` (numerator over a bar over denominator); and maps sub/superscripts (`₂`, `²`) to smaller, shifted glyphs — so equations render correctly in the handwriting style without empty-box glyphs. Greek letters and operators the handwriting font lacks fall back to a symbol font per glyph.

**Output:** 24 FPS MP4 with synced audio (libx264 + AAC)

---

## Image-only Hinglish video generation (`--auto-audio`)

**Give the system ONLY a question image — it generates everything else:** the
correct solution, a Hinglish teacher-style explanation, Sarvam TTS narration,
step-by-step board writing, and the final MP4. The classic image + audio
pipeline above is unchanged; this is a separate, optional mode.

### API keys via `.env` (recommended)

Copy `.env.example` to `.env` in the project root and fill in your keys:

```ini
GEMINI_API_KEY=your_gemini_key
SARVAM_API_KEY=your_sarvam_key
```

The `.env` file is gitignored and loaded automatically by `main.py` (and by the
individual scripts when run standalone). Shell environment variables always
take precedence over `.env` values, so the explicit `$env:`/`export` commands
below also still work.

### Run it

PowerShell:

```powershell
$env:GEMINI_API_KEY="your_gemini_key"
$env:SARVAM_API_KEY="your_sarvam_key"
python main.py --image input/question.png --auto-audio --language hinglish --tts-provider sarvam --output output/final.mp4
```

CMD:

```bat
set GEMINI_API_KEY=your_gemini_key
set SARVAM_API_KEY=your_sarvam_key
python main.py --image input/question.png --auto-audio --language hinglish --tts-provider sarvam --output output/final.mp4
```

Linux/macOS:

```bash
export GEMINI_API_KEY="your_gemini_key"
export SARVAM_API_KEY="your_sarvam_key"
python main.py --image input/question.png --auto-audio --language hinglish --tts-provider sarvam --output output/final.mp4
```

### How the auto pipeline works (storyboard-based)

```
input/question.png
   │ 1. prepare_canvas        → 1280x720 whiteboard: question LEFT, solution area RIGHT   (layout.json)
   │ 2. understand_question   → Gemini Vision (semantics) + EasyOCR (exact text boxes)    (question_understanding.json)
   │ 3. solve_question        → canonical, clean solution (source of truth)               (canonical_solution.json)
   │ 4. verify_solution       → sympy arithmetic tripwire + independent Gemini re-solve   (solution_verification.json)
   │      ✗ disagreement → STOP (no video from an unverified answer; --allow-unverified for testing only)
   │ 5. generate_storyboard   → one short board_lines array + Hinglish narration per step   (storyboard.json)
   │ 5.5 layout_engine        → measure every line, detect overflow, auto-paginate           (layout_validation.json, layout_plan.json)
   │ 5.6 font_glyph_checker   → check every symbol can be rendered; fallback/substitute      (glyph_report.json)
   │ 5.7 pronunciation_manager→ upload custom dict to Sarvam; cache dict_id                  (config/sarvam_dict_id.json)
   │ 6. generate_audio_sarvam → Sarvam TTS PER STEP (bulbul:v3, hi-IN); concat segments     (auto_narration.wav + audio_manifest.json)
   │ 7. transcribe            → Whisper on generated audio — timing REFINEMENT only
   │ 8. build_timeline        → annotation times from audio segment starts/durations         (auto_annotations.json)
   │ 9. validate_output (pre) → narration/audio/timing/board_lines/glyph checks             (validation_report.json, quality_score.json)
   │10. render_video          → whiteboard mode: steps stack in solution zone, final
   │                            answer boxed, correct option ringed on the question
   │11. create_contact_sheet  → 8 sampled frames for visual QA                              (contact_sheet.jpg)
   └─→ output/final.mp4
```

Key properties:

- **Timing comes from audio, not fuzzy matching.** Each storyboard step gets its
  own Sarvam segment; the measured segment durations ARE the timeline. Whisper
  only nudges a step start within ±1 s. Annotations carry `"exact": true` so the
  renderer's pacing heuristics (built for fuzzy Gemini timelines) leave them alone.
- **Hinglish, twice.** Every step has `display_narration_roman` (Roman Hinglish
  for logs/subtitles) and `tts_narration_text` (mixed-script: Hindi words in
  Devanagari, English/science terms in Latin) — fully romanised Hindi degrades
  Indic TTS quality, so Sarvam gets the mixed-script field.
- **The board can't lie.** Board lines come only from the canonical solution;
  Gemini writes the narration around them, not the math itself.
- **Target duration** defaults to 75 s (60–90 s for a normal question; longer
  questions scale up, capped well under 3 min).

### Auto-mode CLI flags

| Flag                 | Default                     | Description                                          |
|----------------------|-----------------------------|------------------------------------------------------|
| `--auto-audio`       | off                         | Enable the image-only mode                           |
| `--language`         | `hinglish`                  | Teaching language for the narration                  |
| `--tts-provider`     | `sarvam`                    | TTS provider (only Sarvam supported)                 |
| `--sarvam-speaker`   | `shubh`                     | Sarvam voice. Run `voice_benchmark_sarvam.py` to compare |
| `--sarvam-dict-id`   | *(auto)*                    | Sarvam pronunciation dict ID (auto-created from `config/pronunciation_dictionary.json`) |
| `--tts-pace`         | `0.92`                      | Speaking pace (slightly slower = teacher-like)       |
| `--target-duration`  | `75`                        | Target narration length in seconds                   |
| `--allow-unverified` | off                         | Continue past a failed verification (testing only)   |
| `--auto-layout`      | on                          | Auto-detect layout mode (two_column / top_bottom / question_first) |
| `--resolution`       | `1280x720`                  | Output resolution (only 1280x720 fully supported)    |
| `--auto-script`      | `output/storyboard.json`    | Storyboard path                                      |
| `--auto-audio-path`  | `output/auto_narration.wav` | Combined narration path                              |
| `--layout`           | `output/layout.json`        | Whiteboard layout metadata path                      |
| `--contact-sheet`    | `output/contact_sheet.jpg`  | QA contact sheet path                                |

### Auto-mode output files

```
output/layout.json                  whiteboard zones (question/solution)
output/canvas.png                   composed 1280x720 board (render background)
output/question_understanding.json  Gemini Vision + OCR question analysis
output/canonical_solution.json      the verified solution (source of truth)
output/solution_verification.json   verifier verdict + confidence + issues
output/storyboard.json              per-step board actions + Hinglish narration
output/audio_segments/s*.wav        one Sarvam segment per storyboard step
output/audio_manifest.json          exact per-segment timings
output/auto_narration.wav           concatenated narration track
output/auto_transcript.json         Whisper transcript of the narration (refinement)
output/auto_annotations.json        final timed annotations
output/layout_validation.json       layout fit check (font size, pages, overflow)
output/layout_plan.json             per-page split (multi-page questions only)
output/glyph_report.json            symbol/font safety report
output/validation_report.json       pre- and post-render validation results
output/quality_score.json           heuristic quality rubric (correctness/readability/pacing)
output/contact_sheet.jpg            8-frame visual QA sheet
output/final.mp4                    the video
output/test_manifest.json           test case checklist for 7 question types
```

### Voice benchmark

Before your first production run, listen to a few voices and choose the best one:

```powershell
# Generate samples for all candidate voices
.venv\Scripts\python.exe scripts\voice_benchmark_sarvam.py --language hinglish --pace 0.92

# Generate for specific voices only
.venv\Scripts\python.exe scripts\voice_benchmark_sarvam.py --speakers shubh,ritu,priya --pace 0.92
```

Outputs to `output/voice_tests/<speaker>/sample_*.wav` + `report.md`. Fill in
`voice_score_template.csv` by ear and set your preferred speaker with `--sarvam-speaker`.

### Pronunciation dictionary

`config/pronunciation_dictionary.json` holds Sarvam-friendly pronunciations for
technical words (velocity → वेलॉसिटी, kinematics → काइनेमैटिक्स, etc.). The
pipeline uploads this to Sarvam automatically and caches the `dict_id` in
`config/sarvam_dict_id.json`. Edit the JSON to add subject-specific words; the
next run re-uploads if you pass `--sarvam-dict-id ""` or delete the cached file.

### Test cases

Run all seven question-type tests (fill in your own image paths):

```powershell
# Physics numerical
.venv\Scripts\python.exe main.py --image input\phytestimg\phytestimg3.png --auto-audio --output output\tc02_final.mp4

# Theory MCQ with --allow-unverified
.venv\Scripts\python.exe main.py --image input\phytestimg\phytestimg2.png --auto-audio --allow-unverified --output output\tc01_final.mp4

# Long question (increase target duration)
.venv\Scripts\python.exe main.py --image input\phytestimg\phytestimg4.png --auto-audio --target-duration 100 --output output\tc03_final.mp4
```

See `output/test_manifest.json` for all 7 test commands and result tracking fields.
```

### Environment variables (auto mode)

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Question understanding, solving, verification, storyboard narration |
| `SARVAM_API_KEY` | Sarvam TTS (bulbul:v3) — get one at https://dashboard.sarvam.ai |

---

## Bonus: Rename Questions Utility

`scripts/rename_questions.py` solves a separate workflow problem: renaming randomly-named images from a ZIP file to a clean `Q1.png`, `S1.png`, `Q2.png`, `S2.png` format using Excel metadata.

### Problem

ZIP files from content pipelines often contain images with hashed filenames (e.g., `a3f82b1c.png`) that can't be easily matched to their corresponding questions.

### Solution

A 3-stage process:

1. **Read Excel metadata** — auto-detects filename and question ID columns
2. **Extract & match images** — matches ZIP contents to metadata rows
3. **Rename & classify** — renames to `Q<n>` (question) or `S<n>` (solution) format

### Usage

```bash
python scripts/rename_questions.py --zip input/questions.zip --excel input/metadata.xlsx --output output/renamed/
```

**Handles edge cases:** column name variations, missing question numbers, duplicate names, macOS ZIP artifacts (`.DS_Store`, `__MACOSX/`), and mixed image formats.

---

## Evaluation Harness (`scripts/eval_corpus.py`)

A manifest-driven evaluator for checking output quality across **multiple subjects
and languages** (built for the English physics / chemistry / maths / biology push).
It runs each question through the **real `main.py` pipeline**, scores the result on
quality signals, writes a per-subject markdown report, and extracts sample frames
for a quick visual check.

### Signals scored

| Signal | Meaning |
|--------|---------|
| `english_script` | Written text is Latin, not Devanagari (an English slide should produce English notes) |
| `write_step` | Count of worked-solution lines (numerical questions should show their working) |
| `frac` | Count of stacked fractions (`\frac{}{}`) used |
| `answer_marked` | The correct option is marked |
| `within_audio` / `not_frontloaded` | Actions stay inside the audio and spread across it (not bunched at the start) |

### Usage

```bash
# Full pipeline — re-generates annotations per question (needs GEMINI_API_KEY)
python scripts/eval_corpus.py

# Render existing annotations only (no Gemini) — for iterating on rendering
python scripts/eval_corpus.py --reuse

# Custom manifest
python scripts/eval_corpus.py my_manifest.json
```

Outputs: `output/eval_report.md` (per-subject table + PASS/WARN/MISSING verdicts) and
three frames per question under `output/eval_frames/`.

### Manifest

On first run it auto-creates `output/eval_manifest.json`, seeded with the bundled
English question plus **physics / chemistry / maths / biology placeholders**. Add
your own English image + audio for each subject; entries whose files are missing are
reported `MISSING`, so the manifest doubles as a **checklist of subjects still to
cover**.

```json
[
  {
    "name": "phys_dimensional",
    "subject": "physics",
    "image": "input/eng_img_test.png",
    "audio": "input/eng_audio_test.mp3",
    "annotations": "output/eng_test_annotations.json"
  }
]
```

The `annotations` field is optional and only used by `--reuse` (to render a
pre-existing annotation file without calling Gemini).

> Use this to drive the **render → watch → fix** loop: run it, read the report, open
> the frames, fix, repeat. It is the validation tool for the bundled-font,
> subject-diversified-prompt, and stacked-fraction work.

---

## Output Files

| File | Description |
|------|-------------|
| `output/transcript.json` | Whisper transcript with word-level timestamps |
| `output/annotations.json` | Timestamped annotation actions (highlight, write, etc.) |
| `output/final.mp4` | Final rendered video with animated annotations + audio |
| `output/frame_*.png` | Sample frames (development/debugging only) |

---

## Configuration

| Environment Variable | Purpose |
|---------------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key for smart annotation generation |
| `GOOGLE_API_KEY` | Alternative key name (same purpose as above) |

**Whisper model sizes** (trade-off: accuracy vs speed):

| Model    | Parameters | Relative Speed | Best For            |
|----------|-----------|----------------|---------------------|
| `tiny`   | 39M       | Fastest        | Quick testing        |
| `base`   | 74M       | Fast           | Default, good balance|
| `small`  | 244M      | Medium         | Better accuracy      |
| `medium` | 769M      | Slow           | High accuracy        |
| `large`  | 1550M     | Slowest        | Best accuracy        |

---

## Tech Stack

| Technology | Role |
|-----------|------|
| **OpenAI Whisper** | Speech-to-text with word-level timestamps |
| **EasyOCR** | Extract text and positions from question images |
| **Google Gemini (2.5 / 2.0 Flash)** | LLM-powered, language-aware annotation generation |
| **Kalam (bundled TTF)** | Handwriting font for **all** rendered text — Hindi (Devanagari) **and** English (Latin/math) — so the board looks identical on every OS |
| **Pillow (PIL)** | Frame rendering, text drawing, image composition |
| **MoviePy** | Video assembly, audio sync, MP4 encoding |
| **OpenCV** | Image processing support |
| **Pandas + OpenPyXL** | Excel metadata parsing (rename utility) |


## How to run 

Set-Location -LiteralPath 'c:\Users\rohit\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\3B891E5801924AB09B9B456E9C2B369F829E44A6\transfers\2026-23\PW-Automated-Annotation-System\PW-Automated-Annotation-System'
if (!(Test-Path 'C:\pw-aas')) { New-Item -ItemType Directory -Path 'C:\pw-aas' | Out-Null }
robocopy . C:\pw-aas /MIR /NFL /NDL /NJH /NJS /NC /NS /NP
Set-Location -LiteralPath 'C:\pw-aas'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\main.py
