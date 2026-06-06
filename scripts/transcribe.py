#!/usr/bin/env python3
"""Transcribe audio to JSON with word-level timestamps using OpenAI Whisper."""

import whisper
import json
import sys


def transcribe_audio(audio_path, output_path, model_size="base"):
    """
    Transcribe audio file and save result with word-level timestamps.

    Args:
        audio_path: Path to audio file (mp3, wav, etc.)
        output_path: Path to save transcript JSON
        model_size: Whisper model size (tiny/base/small/medium/large)
    """
    print(f"  Loading Whisper '{model_size}' model...")
    model = whisper.load_model(model_size)

    print(f"  Transcribing {audio_path}...")
    result = model.transcribe(audio_path, word_timestamps=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    n_segments = len(result.get("segments", []))
    print(f"  Saved {n_segments} segments -> {output_path}")
    return result


if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else "input/narration.mp3"
    output = sys.argv[2] if len(sys.argv) > 2 else "output/transcript.json"
    model = sys.argv[3] if len(sys.argv) > 3 else "base"
    transcribe_audio(audio, output, model)
