#!/usr/bin/env python3
"""
PW Automated Annotation System
================================
Generates annotated educational videos from a question image + audio narration.

Usage:
    python main.py
    python main.py --image input/question.png --audio input/narration.mp3
    python main.py --skip-transcribe   # reuse existing transcript

Pipeline:
    1. Transcribe audio  (Whisper)    -> word-level timestamps
    2. OCR question image (EasyOCR)   -> question text + option positions
    3. Generate annotations (Claude)  -> timestamped actions synced to audio
    4. Render video (PIL + MoviePy)   -> final annotated video with audio

Set ANTHROPIC_API_KEY for smart LLM-based annotations (works for any question).
Without it the system falls back to regex-based keyword matching.
"""

import argparse
import os
import sys

# Let scripts be imported as modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

from transcribe import transcribe_audio
from ocr_question import extract_question_info
from generate_annotations import generate_annotations
from render_video import render_video


def main():
    parser = argparse.ArgumentParser(
        description="PW Automated Annotation System"
    )
    parser.add_argument("--image", default="input/question.png",
                        help="Path to the question background image")
    parser.add_argument("--audio", default="input/narration.mp3",
                        help="Path to the audio narration")
    parser.add_argument("--output", default="output/final.mp4",
                        help="Output video path")
    parser.add_argument("--transcript", default="output/transcript.json",
                        help="Where to save / read the transcript")
    parser.add_argument("--annotations", default="output/annotations.json",
                        help="Where to save the generated annotations")
    parser.add_argument("--whisper-model", default="base",
                        help="Whisper model size (tiny/base/small/medium/large)")
    parser.add_argument("--skip-transcribe", action="store_true",
                        help="Reuse an existing transcript file")
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    # ── Step 1: Transcribe ──────────────────────────────────────────────
    if args.skip_transcribe and os.path.exists(args.transcript):
        print(f"[1/4] Skipping transcription (reusing {args.transcript})")
    else:
        print("[1/4] Transcribing audio...")
        transcribe_audio(args.audio, args.transcript, args.whisper_model)

    # ── Step 2: OCR ─────────────────────────────────────────────────────
    print("[2/4] Running OCR on question image...")
    question_text, option_positions, question_bbox, enriched_ocr = extract_question_info(args.image)

    # ── Step 3: Annotations ─────────────────────────────────────────────
    print("[3/4] Generating annotations...")
    generate_annotations(args.transcript, question_text, args.annotations)

    # ── Step 4: Render ──────────────────────────────────────────────────
    print("[4/4] Rendering video...")
    render_video(
        args.image, args.annotations, args.audio,
        args.output, option_positions, question_bbox, enriched_ocr,
    )

    print(f"\nDone! Video saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
