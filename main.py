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
from generate_annotations_multimodal import generate_annotations_multimodal
from render_video import render_video


def _prefer_whisper_sync(question_text, language):
    """Use Whisper sync for English/Latin slides; Gemini transcript for Hindi."""
    if language:
        lang = str(language).lower()
        return lang.startswith("en")
    latin = sum(1 for ch in question_text if "a" <= ch.lower() <= "z")
    devanagari = sum(1 for ch in question_text if "\u0900" <= ch <= "\u097f")
    return latin >= 12 and latin >= devanagari * 2


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
    parser.add_argument("--whisper-model", default="small",
                        help="Whisper model size (tiny/base/small/medium/large). "
                             "Used only by the Whisper fallback; 'small' balances "
                             "Hindi accuracy and speed ('base' drops many segments).")
    parser.add_argument("--language", default=None,
                        help="Force the spoken language for transcription, e.g. "
                             "'hi' for Hindi or 'en' for English. Omit to let "
                             "Whisper auto-detect. OCR, annotation, and rendering "
                             "detect the script automatically.")
    parser.add_argument("--skip-transcribe", action="store_true",
                        help="Reuse an existing transcript file")
    parser.add_argument("--ink", default="red",
                        help="Annotation ink colour: red (default, matches the "
                             "reference teacher video), black, blue, green, or an "
                             "'r,g,b' triplet.")
    parser.add_argument("--engine", default="gemini", choices=["gemini", "whisper"],
                        help="Annotation engine. 'gemini' (default) sends the audio "
                             "+ slide image directly to Gemini for a synced timeline "
                             "(fixes Whisper gaps). 'whisper' uses local transcription "
                             "+ text-only annotation generation.")
    parser.add_argument("--reuse-annotations", action="store_true",
                        help="Skip annotation generation and reuse the existing "
                             "annotations file. Useful to iterate on rendering or "
                             "recover from a Gemini rate-limit without re-calling it.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the Gemini response cache (always call the "
                             "API). By default identical audio+image inputs reuse "
                             "the cached response with no API call.")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Ignore any cached Gemini response and overwrite it "
                             "with a fresh API call.")
    parser.add_argument("--no-sync", action="store_true",
                        help="Disable Whisper word-timestamp alignment of the "
                             "Gemini timeline. By default the Gemini path is "
                             "re-timed to the exact moment each action's phrase is "
                             "spoken (fixes annotations appearing before/after the "
                             "narration). Use this to skip Whisper for speed.")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip the pre-render validation gate. By default a bad "
                             "annotation set (blank, wrong-subject, no answer marked) "
                             "blocks the render before it starts.")
    parser.add_argument("--strict-validate", action="store_true",
                        help="Treat validation WARNINGS as blocking errors too.")
    parser.add_argument("--no-whisper-fallback", action="store_true",
                        help="If Gemini generation fails, stop instead of falling "
                             "back to local Whisper. Use this for faster production "
                             "runs where a slow CPU fallback looks like a hang.")
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    ink = args.ink
    if "," in str(ink):
        ink = tuple(int(c) for c in str(ink).split(","))

    # ── Step 1: OCR ─────────────────────────────────────────────────────
    # OCR runs first so its text can prime Whisper / anchor annotation targets.
    print("[1/3] Running OCR on question image...")
    question_text, option_positions, question_bbox, enriched_ocr = extract_question_info(args.image)

    from PIL import Image as _Image
    image_size = _Image.open(args.image).size
    try:
        from moviepy import AudioFileClip as _AFC
        _clip = _AFC(args.audio)
        audio_duration = _clip.duration
        _clip.close()
    except Exception:
        audio_duration = None

    # ── Step 2: Annotations (Gemini multimodal, or Whisper fallback) ────
    print("[2/3] Generating annotations...")
    used_engine = args.engine
    if args.reuse_annotations and os.path.exists(args.annotations):
        print(f"  Reusing existing annotations ({args.annotations})")
        used_engine = "reuse"
    elif args.engine == "gemini":
        try:
            generate_annotations_multimodal(
                args.audio, args.image, args.annotations,
                question_text=question_text, image_size=image_size,
                duration_hint=audio_duration,
                use_cache=not args.no_cache, refresh_cache=args.refresh_cache,
            )
        except Exception as e:
            print(f"  Multimodal generation failed ({str(e)[:120]})")
            if args.no_whisper_fallback:
                print("  Stopping because --no-whisper-fallback is set. Fix/retry "
                      "Gemini instead of running slow local Whisper.")
                sys.exit(3)
            print("  Falling back to Whisper engine...")
            used_engine = "whisper"

    if used_engine == "whisper":
        if args.skip_transcribe and os.path.exists(args.transcript):
            print(f"  Skipping transcription (reusing {args.transcript})")
        else:
            print("  Transcribing audio (Whisper)...")
            transcribe_audio(args.audio, args.transcript, args.whisper_model,
                             args.language, initial_prompt=question_text)
        generate_annotations(
            args.transcript,
            question_text,
            args.annotations,
            audio_path=args.audio,
            image_path=args.image,
            enriched_ocr=enriched_ocr
        )

    # ── Step 3.4: Audio sync (align Gemini actions to spoken words) ─────
    # Gemini guesses action timestamps, which drift (annotation appears before
    # the teacher says it). Whisper word-timestamps fix this: each action is
    # re-timed to the exact second its `spoken_cue` is uttered. Only runs on the
    # Gemini path (the Whisper path is already transcript-driven).
    if used_engine == "gemini" and not args.no_sync:
        import json as _json
        try:
            from align_timeline import load_words, load_gemini_words, align_annotations
            # Prefer Gemini's own timestamped transcript (it transcribes Hindi far
            # better than Whisper on noisy lecture audio); fall back to a Whisper
            # word-timestamp transcript only if Gemini didn't supply one.
            gt_path = os.path.splitext(args.annotations)[0] + ".gtrans.json"
            if _prefer_whisper_sync(question_text, args.language):
                if not (args.skip_transcribe and os.path.exists(args.transcript)):
                    print("  Transcribing for English sync (Whisper word timestamps)...")
                    transcribe_audio(args.audio, args.transcript, args.whisper_model,
                                     args.language or "en", initial_prompt=question_text)
                words = load_words(args.transcript)
            elif os.path.exists(gt_path):
                print("  Syncing to Gemini timestamped transcript...")
                words = load_gemini_words(gt_path, duration=audio_duration)
            else:
                if not (args.skip_transcribe and os.path.exists(args.transcript)):
                    print("  Transcribing for sync (Whisper word timestamps)...")
                    transcribe_audio(args.audio, args.transcript, args.whisper_model,
                                     args.language, initial_prompt=question_text)
                words = load_words(args.transcript)
            with open(args.annotations, encoding="utf-8") as f:
                _anns = _json.load(f)
            _anns = align_annotations(_anns, words, duration=audio_duration)
            with open(args.annotations, "w", encoding="utf-8") as f:
                _json.dump(_anns, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  audio sync skipped ({str(e)[:100]})")

    # ── Step 3.5: Timeline hygiene (engine-agnostic) ────────────────────
    # Whatever produced the annotations (Gemini multimodal already does this
    # internally, but the Whisper-text and rule-based fallbacks do not), make
    # sure the final timeline is ordered, spread across the lecture, and ends
    # inside the audio — no action scheduled past the narration.
    if used_engine != "reuse" and audio_duration:
        import json as _json
        from timing_utils import normalize_timeline
        try:
            with open(args.annotations, encoding="utf-8") as f:
                _anns = _json.load(f)
            _anns = normalize_timeline(_anns, audio_duration)
            with open(args.annotations, "w", encoding="utf-8") as f:
                _json.dump(_anns, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  timeline normalisation skipped ({str(e)[:80]})")

    # ── Step 3.6: Validation gate ───────────────────────────────────────
    # Reject a bad annotation set BEFORE spending minutes rendering (and before a
    # student watches) a blank, wrong-subject, or answer-less board. ERRORs block
    # the render; WARNINGs are advisory. Catches CLASS-1 failures (bad annotations);
    # render-side geometry bugs are guarded inside render_video.py.
    if not args.no_validate:
        print("[3.6] Validating annotations...")
        from validate_annotations import gate
        ok = gate(args.annotations, option_positions=option_positions,
                  question_text=question_text, strict=args.strict_validate)
        if not ok:
            print("\nRender blocked by the validation gate. Fix the annotations "
                  "(or re-run with --no-validate to override).")
            sys.exit(2)

    # ── Step 4: Render ──────────────────────────────────────────────────
    print("[4/4] Rendering video...")
    render_video(
        args.image, args.annotations, args.audio,
        args.output, option_positions, question_bbox, enriched_ocr,
        ink=ink,
    )

    print(f"\nDone! Video saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
