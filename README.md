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
- [Bonus: Rename Questions Utility](#bonus-rename-questions-utility)
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

---

## Project Structure

```
PW-Automated-Annotation-System/
├── main.py                        # Entry point — runs the full 4-step pipeline
├── requirements.txt               # Python dependencies
├── scripts/
│   ├── transcribe.py              # Step 1: Audio → timestamped transcript (Whisper)
│   ├── ocr_question.py            # Step 2: Image → question text + option boxes (EasyOCR)
│   ├── generate_annotations.py    # Step 3: Transcript → timed annotations (Gemini/regex)
│   ├── render_video.py            # Step 4: Compose final annotated video (PIL + MoviePy)
│   └── rename_questions.py        # Utility: bulk-rename images from ZIP + Excel metadata
├── input/
│   ├── question.png               # Source question image (MCQ with options A–D)
│   └── narration.mp3              # Teacher's audio explanation
├── output/
│   ├── transcript.json            # Whisper output with word-level timestamps
│   ├── annotations.json           # Generated timestamped annotations
│   └── final.mp4                  # Final rendered video
├── task02_brief.md                # Writing style profile template (Task 02)
└── task03_explanation.md          # Explanation of the rename utility (Task 03)
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
| **Gemini API** (primary) | `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set | Sends transcript + question text to Gemini 2.0 Flash; generates intelligent step-by-step annotations for any question |
| **Regex fallback** | No API key available | Matches keywords in transcript against hardcoded patterns (limited to specific problem types) |

**Annotation types:**

| Action              | Description                          | Visual Style     |
|---------------------|--------------------------------------|------------------|
| `highlight_question`| Title / main problem statement       | Gold text         |
| `write`             | Solution step (math, explanation)    | White text, animated character-by-character |
| `highlight_option`  | Final answer selection               | Green highlight   |

**Output format** (`output/annotations.json`):
```json
[
  { "time": 0.0,  "action": "highlight_question", "text": "Find Distance Between Two Points" },
  { "time": 15.7, "action": "write",              "text": "Step 1: Distance Formula\nd = \u221a((x\u2082\u2212x\u2081)\u00b2 + (y\u2082\u2212y\u2081)\u00b2)" },
  { "time": 60.6, "action": "highlight_option",   "text": "Option C" }
]
```

### Step 4 — Video Rendering (`render_video.py`)

Composites the final annotated video using **Pillow** (drawing) and **MoviePy** (encoding).

**Canvas layout:**
```
┌──────────────────────────┐
│                          │
│    Original question     │
│    image (from OCR)      │
│                          │
├──────────────────────────┤  ← divider line
│                          │
│    Workspace area        │
│    (420px, dark bg)      │
│    - Animated solution   │
│      steps appear here   │
│    - Pen cursor glow     │
│                          │
└──────────────────────────┘
```

**Animation features:**
- **Character-by-character text reveal** at 14 chars/sec (simulates handwriting)
- **Glowing pen cursor** follows the writing position
- **Question highlight** — semi-transparent yellow overlay on the question region
- **Option highlight** — green overlay + checkmark on the correct answer
- **Fade in/out** — 0.6s transitions at start and end

**Color scheme:**
- Title text: Gold `(255, 215, 0)`
- Step labels: Soft blue `(120, 180, 255)`
- Body text: Off-white `(240, 240, 240)`
- Answer text: Green `(80, 255, 130)`
- Workspace background: Dark blue-grey `(22, 22, 34)`

**Output:** 24 FPS MP4 with synced audio (libx264 + AAC)

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
| **Google Gemini 2.0 Flash** | LLM-powered annotation generation |
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
