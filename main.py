#!/usr/bin/env python3
"""
PW Automated Annotation System
================================
Generates annotated educational videos from a question image + audio narration.

Usage:
    python main.py
    python main.py --image input/question.png --audio input/narration.mp3
    python main.py --skip-transcribe   # reuse existing transcript

    # IMAGE-ONLY mode (no audio input): auto-solve + Hinglish Sarvam narration
    python main.py --image input/question.png --auto-audio --language hinglish \
                   --tts-provider sarvam --output output/final.mp4

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

# Pick up API keys from the project-root .env (shell env vars still win).
from env_utils import load_dotenv
load_dotenv()

from transcribe import transcribe_audio
from ocr_question import extract_question_info
from generate_annotations import generate_annotations
from generate_annotations_multimodal import generate_annotations_multimodal
from render_video import render_video


def _prefer_whisper_sync(question_text, language, sync_source="auto"):
    """
    Decide whether to sync against Whisper word-timestamps (vs the Gemini
    timestamped transcript).

    `sync_source` is the explicit guard for the (A) routing prototype:
      - "whisper" / "gemini": force that source regardless of slide text.
      - "auto" (default): UNCHANGED legacy behaviour \u2014 Whisper for English/Latin
        slides, Gemini transcript for Devanagari slides. Note this judges by the
        SLIDE text, so an English slide with HINDI narration still picks Whisper;
        pass --sync-source gemini to override that for Hindi-audio clips.
    """
    if sync_source == "whisper":
        return True
    if sync_source == "gemini":
        return False
    if language:
        lang = str(language).lower()
        return lang.startswith("en")
    latin = sum(1 for ch in question_text if "a" <= ch.lower() <= "z")
    devanagari = sum(1 for ch in question_text if "\u0900" <= ch <= "\u097f")
    return latin >= 12 and latin >= devanagari * 2


def _whisper_disabled(args):
    """Whisper transcription is optional — it only REFINES step timing by ±1s on
    top of the Sarvam segment durations that already drive the timeline. Disable
    it (via --skip-whisper or the SKIP_WHISPER env var) to avoid loading the
    torch-backed model on memory-constrained / CPU-only hosts, where it is both
    the slowest step and a frequent apparent-hang. Env var lets a deployment
    force-skip without changing the launch command."""
    if getattr(args, "skip_whisper", False):
        return True
    return os.environ.get("SKIP_WHISPER", "").strip().lower() in ("1", "true", "yes", "on")


def run_auto_audio_pipeline(args, ink):
    """IMAGE-ONLY mode: question image in → fully narrated whiteboard video out.

    Storyboard-based flow: prepare board → understand → solve → verify →
    storyboard → layout plan → glyph check → pronunciation dict → Sarvam
    per-step audio → exact timeline from segment durations (Whisper refines
    only) → render → validate → contact sheet. The classic image+audio
    pipeline in main() is untouched.
    """
    from prepare_canvas import prepare_canvas
    from understand_question import understand_question
    from solve_question import solve_question
    from verify_solution import verify_solution
    from generate_storyboard import generate_storyboard
    from layout_engine import plan_and_validate as layout_plan_and_validate
    from font_glyph_checker import check_storyboard as glyph_check
    from pronunciation_manager import get_or_create_dict_id
    from generate_audio_sarvam import generate_storyboard_audio
    from build_timeline import build_timeline
    from validate_output import validate_output
    from create_contact_sheet import create_contact_sheet

    if args.tts_provider != "sarvam":
        raise SystemExit(f"Unsupported --tts-provider '{args.tts_provider}'")
    language = args.language or "hinglish"
    canvas_path = "output/canvas.png"
    manifest_path = "output/audio_manifest.json"
    # Auto mode keeps its artifacts separate from the classic pipeline's
    # defaults so the two modes never clobber each other's files.
    annotations_path = ("output/auto_annotations.json"
                        if args.annotations == "output/annotations.json"
                        else args.annotations)
    transcript_path = ("output/auto_transcript.json"
                       if args.transcript == "output/transcript.json"
                       else args.transcript)

    print("[1/10] Preparing whiteboard canvas...")
    # Pass layout_mode=None when auto-layout is on (prepare_canvas detects it)
    forced_mode = None if getattr(args, "auto_layout", True) else "two_column"
    layout = prepare_canvas(args.image, canvas_out=canvas_path,
                            layout_out=args.layout,
                            layout_mode=forced_mode)

    print("[2/10] Understanding question (EasyOCR on canvas + Gemini Vision)...")
    from ocr_question import extract_question_info
    canvas_ocr = extract_question_info(canvas_path)
    _, option_positions, question_bbox, enriched_ocr = canvas_ocr
    understanding = understand_question(
        image_path=args.image, layout=layout, language=language,
        canvas_ocr=canvas_ocr)

    print("[3/10] Solving question (canonical solution)...")
    solution = solve_question(image_path=args.image,
                              understanding=understanding, language=language)

    print("[4/10] Verifying solution...")
    verification = verify_solution(image_path=args.image,
                                   understanding=understanding,
                                   solution=solution)
    if verification["status"] != "verified":
        if not args.allow_unverified:
            print("\nSolution verification FAILED — no video will be made from an "
                  "unverified answer. Inspect output/solution_verification.json; "
                  "re-run with --allow-unverified only for testing.")
            sys.exit(4)
        print("  WARNING: continuing with UNVERIFIED solution (--allow-unverified)")

    print("[5/10] Generating storyboard...")
    storyboard = generate_storyboard(
        solution=solution, layout=layout, language="hinglish",
        target_duration=args.target_duration,
        output_path=args.auto_script)

    print("[5.5/10] Layout planning + glyph safety check...")
    try:
        layout_plan_and_validate(storyboard, layout)
    except Exception as e:
        print(f"  Layout engine error ({str(e)[:100]}); continuing")
    try:
        glyph_check(storyboard)
    except Exception as e:
        print(f"  Glyph check error ({str(e)[:100]}); continuing")

    print("[5.7/10] Getting/creating pronunciation dictionary...")
    dict_id = None
    try:
        cli_dict_id = getattr(args, "sarvam_dict_id", None)
        dict_id = cli_dict_id or get_or_create_dict_id()
    except Exception as e:
        print(f"  Pronunciation manager error ({str(e)[:100]}); no dict_id")

    print("[6/10] Generating Sarvam narration (per storyboard step)...")
    generate_storyboard_audio(
        storyboard_path=args.auto_script,
        output_dir="output/audio_segments",
        combined_output_path=args.auto_audio_path,
        target_language_code="hi-IN",
        speaker=args.sarvam_speaker,
        pace=args.tts_pace,
        manifest_path=manifest_path,
        dict_id=dict_id,
    )

    print("[7/10] Transcribing narration (Whisper, timing refinement only)...")
    if _whisper_disabled(args):
        print("  Whisper DISABLED (--skip-whisper / SKIP_WHISPER); using Sarvam "
              "segment timings as-is. Saves ~500 MB RAM + the model download — "
              "recommended on low-memory hosts (e.g. Streamlit Cloud).")
        transcript_path = None
    else:
        try:
            transcribe_audio(args.auto_audio_path, transcript_path,
                             args.whisper_model, None)
        except Exception as e:
            print(f"  Whisper refinement unavailable ({str(e)[:100]}); "
                  "segment timings from the manifest are used as-is")
            transcript_path = None

    print("[8/10] Building timeline from audio segment durations...")
    build_timeline(storyboard_path=args.auto_script,
                   audio_manifest_path=manifest_path,
                   transcript_path=transcript_path,
                   output_annotations_path=annotations_path,
                   option_positions=option_positions)

    print("[9/10] Validating before render...")
    ok, _ = validate_output(
        stage="pre",
        storyboard_path=args.auto_script,
        manifest_path=manifest_path,
        annotations_path=annotations_path,
        audio_path=args.auto_audio_path,
        layout_path=args.layout,
        layout_validation_path="output/layout_validation.json",
        glyph_report_path="output/glyph_report.json",
        video_path=args.output,
        contact_sheet_path=args.contact_sheet,
        allow_unverified=args.allow_unverified,
    )
    if not ok:
        print("\nPre-render validation failed — fix the issues above "
              "(see output/validation_report.json).")
        sys.exit(5)

    print("[10/10] Rendering whiteboard video...")
    render_video(
        canvas_path, annotations_path, args.auto_audio_path, args.output,
        option_positions, question_bbox, enriched_ocr,
        ink=ink, layout_path=args.layout, mode="whiteboard_storyboard",
    )

    print("  Creating contact sheet...")
    try:
        create_contact_sheet(args.output, args.contact_sheet)
    except Exception as e:
        print(f"  contact sheet skipped ({str(e)[:100]})")

    validate_output(
        stage="post",
        storyboard_path=args.auto_script,
        manifest_path=manifest_path,
        annotations_path=annotations_path,
        audio_path=args.auto_audio_path,
        layout_path=args.layout,
        layout_validation_path="output/layout_validation.json",
        glyph_report_path="output/glyph_report.json",
        video_path=args.output,
        contact_sheet_path=args.contact_sheet,
        allow_unverified=args.allow_unverified,
    )

    print(f"\nDone! Video saved to: {os.path.abspath(args.output)}")
    print(f"Contact sheet: {os.path.abspath(args.contact_sheet)}")


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
                        help="Classic mode: force the spoken language for "
                             "transcription, e.g. 'hi' or 'en' (omit to "
                             "auto-detect). With --auto-audio: the teaching "
                             "language, default 'hinglish'.")
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
    parser.add_argument("--sync-source", default="auto",
                        choices=["auto", "whisper", "gemini"],
                        help="Which transcript drives audio sync. 'auto' (default) "
                             "picks Whisper for English/Latin slides and the Gemini "
                             "timestamped transcript for Devanagari slides — current "
                             "behaviour, unchanged. 'gemini' FORCES the decoded Gemini "
                             "transcript (better for Hindi NARRATION even when the slide "
                             "is English; PROTOTYPE). 'whisper' forces Whisper.")
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
    parser.add_argument("--skip-whisper", action="store_true",
                        help="Skip the Whisper transcription refinement entirely "
                             "(--auto-audio mode). Whisper only nudges step timing "
                             "±1s on top of the Sarvam segment durations, so skipping "
                             "it changes the video negligibly while saving ~500 MB RAM, "
                             "the model download, and the slowest pipeline step. "
                             "Recommended on low-memory / CPU-only hosts. Can also be "
                             "forced with the SKIP_WHISPER=1 environment variable.")
    # ── Image-only (auto-audio) mode ─────────────────────────────────────
    parser.add_argument("--auto-audio", action="store_true",
                        help="IMAGE-ONLY mode: no narration file needed. The system "
                             "solves the question, writes a Hinglish storyboard, "
                             "generates teacher audio with Sarvam TTS, and renders a "
                             "1280x720 whiteboard video (question left, solution "
                             "right). --audio is ignored in this mode.")
    parser.add_argument("--tts-provider", default="sarvam", choices=["sarvam"],
                        help="TTS provider for --auto-audio (only 'sarvam').")
    parser.add_argument("--sarvam-speaker", default="shubh",
                        help="Sarvam TTS speaker voice (default: shubh). "
                             "Run voice_benchmark_sarvam.py to compare voices.")
    parser.add_argument("--sarvam-dict-id", default=None,
                        help="Sarvam pronunciation dictionary ID. If omitted, "
                             "the pronunciation_manager auto-creates one from "
                             "config/pronunciation_dictionary.json.")
    parser.add_argument("--tts-pace", type=float, default=0.92,
                        help="Sarvam TTS speaking pace (default 0.92 — "
                             "slightly slower than neutral, teacher-like).")
    parser.add_argument("--target-duration", type=int, default=75,
                        help="Target video length in seconds for --auto-audio "
                             "(default 75; the storyboard aims for 60-90s).")
    parser.add_argument("--allow-unverified", action="store_true",
                        help="Continue even if the solution verification pass "
                             "disagrees with the solver. TESTING ONLY — the video "
                             "may teach a wrong answer.")
    parser.add_argument("--auto-script", default="output/storyboard.json",
                        help="Where to save the generated storyboard.")
    parser.add_argument("--auto-audio-path", default="output/auto_narration.wav",
                        help="Where to save the combined Sarvam narration.")
    parser.add_argument("--layout", default="output/layout.json",
                        help="Where to save the whiteboard layout metadata.")
    parser.add_argument("--contact-sheet", default="output/contact_sheet.jpg",
                        help="Where to save the QA contact sheet.")
    parser.add_argument("--resolution", default="1280x720",
                        help="Output video resolution (default: 1280x720). "
                             "Only 1280x720 is fully supported currently.")
    parser.add_argument("--auto-layout", default=True, action=argparse.BooleanOptionalAction,
                        help="Auto-detect layout mode (two_column/top_bottom/etc.). "
                             "Enabled by default. Use --no-auto-layout to force "
                             "two_column.")
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    ink = args.ink
    if "," in str(ink):
        ink = tuple(int(c) for c in str(ink).split(","))

    if args.auto_audio:
        run_auto_audio_pipeline(args, ink)
        return

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
            from align_timeline import load_words, load_gemini_words, align_best
            # Prefer Gemini's own timestamped transcript (it transcribes Hindi far
            # better than Whisper on noisy lecture audio); fall back to a Whisper
            # word-timestamp transcript only if Gemini didn't supply one.
            gt_path = os.path.splitext(args.annotations)[0] + ".gtrans.json"
            sources = []
            if _prefer_whisper_sync(question_text, args.language, args.sync_source):
                if not (args.skip_transcribe and os.path.exists(args.transcript)):
                    print("  Transcribing for English sync (Whisper word timestamps)...")
                    transcribe_audio(args.audio, args.transcript, args.whisper_model,
                                     args.language or "en", initial_prompt=question_text)
                sources.append(("whisper", load_words(args.transcript)))
                # Fallback candidate (auto mode only): Gemini's own transcript. An
                # English SLIDE can still have HINDI narration, whose spoken_cues
                # don't match an English Whisper transcript and collapse into an
                # early burst. align_best keeps Whisper unless that alignment is
                # actually broken AND Gemini's transcript aligns clearly better — so
                # well-synced English videos are unchanged, at no extra ASR cost.
                if args.sync_source == "auto" and os.path.exists(gt_path):
                    sources.append(("gemini", load_gemini_words(gt_path, duration=audio_duration)))
            elif os.path.exists(gt_path):
                print("  Syncing to Gemini timestamped transcript...")
                sources.append(("gemini", load_gemini_words(gt_path, duration=audio_duration)))
            else:
                if not (args.skip_transcribe and os.path.exists(args.transcript)):
                    print("  Transcribing for sync (Whisper word timestamps)...")
                    transcribe_audio(args.audio, args.transcript, args.whisper_model,
                                     args.language, initial_prompt=question_text)
                sources.append(("whisper", load_words(args.transcript)))
            with open(args.annotations, encoding="utf-8") as f:
                _anns = _json.load(f)
            _anns = align_best(_anns, sources, duration=audio_duration)
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
