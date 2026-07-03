#!/usr/bin/env python3
"""Transcribe audio to JSON with word-level timestamps using OpenAI Whisper.

Whisper (and its torch backend + model weights) is imported LAZILY inside
transcribe_audio, not at module load. This lets the rest of the pipeline import
this module for free on memory-constrained hosts (e.g. Streamlit Cloud's ~1 GB
tier) when transcription is skipped — see main.py's --skip-whisper / SKIP_WHISPER.
"""

import json
import sys


def transcribe_audio(audio_path, output_path, model_size="base", language=None,
                     initial_prompt=None):
    """
    Transcribe audio file and save result with word-level timestamps.

    Args:
        audio_path: Path to audio file (mp3, wav, etc.)
        output_path: Path to save transcript JSON
        model_size: Whisper model size (tiny/base/small/medium/large)
        language: ISO language code to force (e.g. "hi" for Hindi). When None,
                  Whisper auto-detects the spoken language. Forcing "hi" makes
                  Whisper emit Devanagari Hindi instead of romanising/translating
                  it, and noticeably improves accuracy on Hindi narration.
        initial_prompt: Optional priming text (e.g. the question's key terms).
                  Whisper biases its vocabulary toward these words, which sharpens
                  domain terms (e.g. हार्मोन, अपरा, hCG) and their timing.
    """
    import whisper  # lazy: pulls in torch + weights only when actually transcribing
    print(f"  Loading Whisper '{model_size}' model...")
    model = whisper.load_model(model_size)

    if language:
        print(f"  Transcribing {audio_path} (language={language})...")
    else:
        print(f"  Transcribing {audio_path} (auto-detect language)...")
    if initial_prompt:
        print(f"  Priming with question terms ({len(initial_prompt)} chars)")
    result = model.transcribe(audio_path, word_timestamps=True, language=language,
                              initial_prompt=initial_prompt)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    n_segments = len(result.get("segments", []))
    print(f"  Saved {n_segments} segments -> {output_path}")
    return result


if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else "input/narration.mp3"
    output = sys.argv[2] if len(sys.argv) > 2 else "output/transcript.json"
    model = sys.argv[3] if len(sys.argv) > 3 else "base"
    language = sys.argv[4] if len(sys.argv) > 4 else None
    prompt = sys.argv[5] if len(sys.argv) > 5 else None
    transcribe_audio(audio, output, model, language, prompt)
