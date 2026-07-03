#!/usr/bin/env python3
"""Shared Gemini helpers for the image-only (auto-audio) pipeline.

One place for: client creation, the model-fallback/backoff loop, and robust
JSON extraction — the same battle-tested behaviour generate_annotations_multimodal
uses, factored out so the new storyboard-pipeline stages (understand / solve /
verify / storyboard) don't each reimplement it.
"""

import json
import os
import re

# Models tried in order (free-tier quota for one model is sometimes disabled;
# the next model may still have quota). All accept image + text input.
GEMINI_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
)

_IMAGE_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".webp": "image/webp"}


def get_client():
    """Create a google-genai client from GEMINI_API_KEY / GOOGLE_API_KEY."""
    from google import genai
    from env_utils import load_dotenv
    load_dotenv()  # pick up keys from .env for standalone script runs
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY / GOOGLE_API_KEY is not set. Add it to the "
            "project's .env file (see .env.example) or set it in the shell, "
            'e.g. PowerShell:  $env:GEMINI_API_KEY="your_key"')
    return genai.Client(api_key=api_key)


def image_part(path):
    """Inline image content part for a Gemini call."""
    from google.genai import types
    mime = _IMAGE_MIME.get(os.path.splitext(path)[1].lower(), "image/png")
    with open(path, "rb") as f:
        return types.Part.from_bytes(data=f.read(), mime_type=mime)


def parse_json_response(response):
    """Extract the first JSON value from a Gemini response (tolerates markdown
    fences and trailing prose, which trip a plain json.loads)."""
    raw = (getattr(response, "text", None) or "").strip()
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
            f"Gemini returned malformed JSON (line {e.lineno}, col {e.colno}). "
            f"Raw response saved to {debug_path}") from e


def call_gemini_json(contents, temperature=0.2, label="Gemini call"):
    """Call Gemini expecting a JSON response; model fallback + backoff on
    transient errors. Returns the parsed JSON value."""
    import time
    from google.genai import types
    client = get_client()
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=temperature,
    )
    last_err = None
    for model in GEMINI_MODELS:
        for attempt in range(3):
            try:
                print(f"  {label} with {model}...")
                response = client.models.generate_content(
                    model=model, contents=contents, config=config)
                return parse_json_response(response)
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
                print(f"    model {model} failed ({es[:90]}), trying next...")
                break
    raise RuntimeError(f"All Gemini models failed for {label}; last error: {last_err}")
