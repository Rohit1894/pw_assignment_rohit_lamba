#!/usr/bin/env python3
"""Sarvam TTS audio generation for the storyboard — one segment per step.

Audio is generated PER STORYBOARD STEP (never one long file): the measured
duration of each segment IS the timeline, which is what makes writing/audio
sync exact. Segments are then concatenated (with a small silence gap) into the
final narration track, and an audio_manifest.json records the exact start/end
of every step inside the combined track.

Uses the official `sarvamai` SDK when installed; otherwise falls back to the
plain REST API via `requests`. Reads SARVAM_API_KEY from the environment.

Outputs:
    output/audio_segments/s1.wav ...
    output/auto_narration.wav
    output/audio_manifest.json
"""

import base64
import json
import os
import sys
import time
import wave

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SAMPLE_RATE = 22050
SEGMENT_GAP_SEC = 0.3          # breathing pause between steps (0.2-0.4s)
MAX_TTS_CHARS = 1400           # Sarvam per-request text limit safety margin


def _get_api_key():
    from env_utils import load_dotenv
    load_dotenv()  # pick up keys from .env for standalone script runs
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        raise RuntimeError(
            "SARVAM_API_KEY is not set. Get a key from https://dashboard.sarvam.ai "
            "and add it to the project's .env file (see .env.example), or set it "
            'in the shell, e.g. PowerShell:  $env:SARVAM_API_KEY="your_key"')
    return key


def _tts_call_sdk(text, target_language_code, speaker, model, pace,
                  dict_id=None):
    """Call Sarvam via the official SDK. Returns raw WAV bytes."""
    from sarvamai import SarvamAI
    client = SarvamAI(api_subscription_key=_get_api_key())
    kwargs = dict(
        text=text,
        target_language_code=target_language_code,
        speaker=speaker,
        model=model,
        pace=pace,
        speech_sample_rate=SAMPLE_RATE,
    )
    if dict_id:
        kwargs["dict_id"] = dict_id
    resp = client.text_to_speech.convert(**kwargs)
    audios = getattr(resp, "audios", None) or (resp.get("audios") if isinstance(resp, dict) else None)
    if not audios:
        raise RuntimeError(f"Sarvam SDK returned no audio for text: {text[:60]!r}")
    return base64.b64decode(audios[0])


def _tts_call_rest(text, target_language_code, speaker, model, pace,
                   dict_id=None):
    """REST fallback when the SDK is not installed. Returns raw WAV bytes."""
    import requests
    payload = {
        "text": text,
        "target_language_code": target_language_code,
        "speaker": speaker,
        "model": model,
        "pace": pace,
        "speech_sample_rate": SAMPLE_RATE,
    }
    if dict_id:
        payload["dict_id"] = dict_id
    resp = requests.post(
        SARVAM_TTS_URL,
        headers={"api-subscription-key": _get_api_key(),
                 "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    if resp.status_code != 200:
        # Surface the API's own message — a wrong speaker/model name comes back
        # here as a 400 with a helpful detail string.
        raise RuntimeError(f"Sarvam TTS HTTP {resp.status_code}: {resp.text[:300]}")
    audios = (resp.json() or {}).get("audios") or []
    if not audios:
        raise RuntimeError(f"Sarvam TTS returned no audio for text: {text[:60]!r}")
    return base64.b64decode(audios[0])


def _is_transient(err_str):
    s = err_str.lower()
    return any(k in s for k in ("429", "500", "502", "503", "504", "timeout",
                                "timed out", "connection", "temporarily",
                                "rate limit", "overload"))


def generate_sarvam_audio_segment(
    text: str,
    output_path: str,
    target_language_code: str = "hi-IN",
    speaker: str = "shubh",
    model: str = "bulbul:v3",
    pace: float = 0.92,
    dict_id: str | None = None,
) -> str:
    """Generate one WAV segment via Sarvam TTS. Returns output_path.

    Retries transient errors (429/5xx/network) with exponential backoff; hard
    errors (bad key, invalid speaker) are raised immediately with the API's
    message. Validates the written file exists and is non-empty."""
    text = (text or "").strip()
    if not text:
        raise ValueError("generate_sarvam_audio_segment: empty text")
    speaker = str(speaker).strip().lower()   # API speaker names are lowercase ("ritu")
    if len(text) > MAX_TTS_CHARS:
        raise ValueError(f"TTS text too long ({len(text)} chars > {MAX_TTS_CHARS}); "
                         "split the storyboard step into shorter sentences")

    try:
        import sarvamai  # noqa: F401
        call = _tts_call_sdk
    except ImportError:
        call = _tts_call_rest

    last_err = None
    for attempt in range(4):
        try:
            wav_bytes = call(text, target_language_code, speaker, model, pace,
                             dict_id=dict_id)
            break
        except Exception as e:
            last_err = e
            es = str(e)
            if _is_transient(es) and attempt < 3:
                wait = 2 ** (attempt + 1)
                print(f"    transient Sarvam error ({es[:80]}); retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Sarvam TTS failed for segment "
                               f"{os.path.basename(output_path)}: {es[:300]}") from e
    else:
        raise RuntimeError(f"Sarvam TTS failed after retries: {last_err}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(wav_bytes)
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 100:
        raise RuntimeError(f"Sarvam TTS produced an empty file: {output_path}")
    return output_path


def _read_wav(path):
    """Read a PCM WAV; return (params, frames)."""
    with wave.open(path, "rb") as w:
        return w.getparams(), w.readframes(w.getnframes())


def _wav_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def generate_storyboard_audio(
    storyboard_path: str,
    output_dir: str,
    combined_output_path: str,
    target_language_code: str = "hi-IN",
    speaker: str = "shubh",
    pace: float = 0.92,
    model: str = "bulbul:v3",
    manifest_path: str = "output/audio_manifest.json",
    gap_sec: float = SEGMENT_GAP_SEC,
    dict_id: str | None = None,
) -> dict:
    """Generate one Sarvam segment per storyboard step, concatenate them with a
    small silence gap, and write audio_manifest.json with exact timings.

    Concatenation uses the stdlib `wave` module (all segments share the same
    PCM format from the same API), so durations in the manifest are
    sample-exact — no re-encode drift. Returns the manifest dict."""
    with open(storyboard_path, encoding="utf-8") as f:
        storyboard = json.load(f)
    steps = storyboard.get("steps") or []
    if not steps:
        raise RuntimeError("generate_storyboard_audio: storyboard has no steps")

    from narration_normalizer import normalize_tts_text

    os.makedirs(output_dir, exist_ok=True)
    seg_paths = []
    for st in steps:
        sid = st.get("id")
        # Roman Hinglish (display_narration_roman) gives much better pronunciation
        # for English technical terms than Devanagari mixed-script (tts_narration_text).
        text = (st.get("display_narration_roman") or st.get("tts_narration_text") or "").strip()
        if not sid or not text:
            raise RuntimeError(f"Storyboard step {sid!r} has no tts_narration_text")
        # Expand digits and math symbols to English words so Sarvam hi-IN
        # says "four point six zero millimeters" rather than "chaar point chhah"
        text = normalize_tts_text(text)
        seg_path = os.path.join(output_dir, f"{sid}.wav")
        preview = text[:70].encode("ascii", errors="replace").decode("ascii")
        print(f"  Sarvam TTS [{sid}] ({len(text)} chars): {preview}")
        generate_sarvam_audio_segment(
            text, seg_path, target_language_code=target_language_code,
            speaker=speaker, model=model, pace=pace, dict_id=dict_id)
        seg_paths.append((sid, seg_path))

    # ── Concatenate with silence gaps (sample-exact) ─────────────────────
    params0, _ = _read_wav(seg_paths[0][1])
    gap_frames = int(gap_sec * params0.framerate)
    silence = b"\x00" * (gap_frames * params0.nchannels * params0.sampwidth)

    segments = []
    os.makedirs(os.path.dirname(combined_output_path) or ".", exist_ok=True)
    with wave.open(combined_output_path, "wb") as out:
        out.setnchannels(params0.nchannels)
        out.setsampwidth(params0.sampwidth)
        out.setframerate(params0.framerate)
        cursor_frames = 0
        for i, (sid, seg_path) in enumerate(seg_paths):
            params, frames = _read_wav(seg_path)
            if (params.nchannels, params.sampwidth, params.framerate) != \
                    (params0.nchannels, params0.sampwidth, params0.framerate):
                raise RuntimeError(f"Segment {sid} has mismatched audio format")
            start = cursor_frames / float(params0.framerate)
            out.writeframes(frames)
            cursor_frames += len(frames) // (params0.nchannels * params0.sampwidth)
            end = cursor_frames / float(params0.framerate)
            segments.append({
                "id": sid,
                "path": seg_path.replace("\\", "/"),
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "duration": round(end - start, 3),
            })
            if i < len(seg_paths) - 1:
                out.writeframes(silence)
                cursor_frames += gap_frames

    total = cursor_frames / float(params0.framerate)
    # Validate the combined file really has the expected length.
    actual = _wav_duration(combined_output_path)
    if abs(actual - total) > 0.05:
        raise RuntimeError(f"Combined narration duration mismatch: "
                           f"expected {total:.2f}s, file has {actual:.2f}s")

    manifest = {
        "combined_audio": combined_output_path.replace("\\", "/"),
        "total_duration_sec": round(total, 3),
        "sample_rate": params0.framerate,
        "speaker": speaker,
        "model": model,
        "pace": pace,
        "segments": segments,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Narration: {len(segments)} segments, {total:.1f}s total "
          f"-> {combined_output_path}")
    print(f"  Manifest -> {manifest_path}")
    return manifest


if __name__ == "__main__":
    sb = sys.argv[1] if len(sys.argv) > 1 else "output/storyboard.json"
    generate_storyboard_audio(sb, "output/audio_segments",
                              "output/auto_narration.wav")
