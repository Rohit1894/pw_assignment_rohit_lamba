#!/usr/bin/env python3
"""Manage Sarvam pronunciation dictionaries to fix common TTS mispronunciations.

Reads config/pronunciation_dictionary.json and tries to upload it to the
Sarvam /dictionaries endpoint (if the API supports it). Stores the returned
dict_id in config/sarvam_dict_id.json for reuse across runs.

If the upload fails (endpoint not available on this Sarvam plan), the module
returns None and the pipeline continues without a custom dictionary — the
pronunciation_dictionary.json is then used only as documentation.

Usage:
    from pronunciation_manager import get_or_create_dict_id
    dict_id = get_or_create_dict_id()   # may be None
"""

import json
import os


_DICT_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "pronunciation_dictionary.json")
_DICT_ID_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "sarvam_dict_id.json")

SARVAM_DICT_URL = "https://api.sarvam.ai/dictionaries"


def _load_pronunciations(language_code: str = "hi-IN") -> dict:
    """Load the pronunciation map for a given language code."""
    try:
        with open(_DICT_JSON, encoding="utf-8") as f:
            data = json.load(f)
        return (data.get("pronunciations") or {}).get(language_code, {})
    except Exception:
        return {}


def _upload_to_sarvam(pronunciations: dict, language_code: str,
                      api_key: str) -> str | None:
    """Try to create/update a Sarvam pronunciation dictionary.

    Returns the dict_id on success, None if the endpoint is not supported."""
    try:
        import requests
        payload = {
            "language_code": language_code,
            "pronunciations": [
                {"word": word, "pronunciation": pron}
                for word, pron in pronunciations.items()
            ],
        }
        resp = requests.post(
            SARVAM_DICT_URL,
            headers={"api-subscription-key": api_key,
                     "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        if resp.status_code in (200, 201):
            body = resp.json()
            return body.get("dict_id") or body.get("id")
        if resp.status_code == 404:
            print("  Sarvam dictionaries endpoint not available on this plan "
                  "(404). Continuing without custom pronunciation dictionary.")
            return None
        print(f"  Sarvam dictionary upload returned {resp.status_code}: "
              f"{resp.text[:200]}. Continuing without dictionary.")
        return None
    except Exception as e:
        print(f"  Could not upload pronunciation dictionary ({str(e)[:100]}). "
              "Continuing without it.")
        return None


def _load_cached_dict_id() -> str | None:
    """Return the previously uploaded dict_id if it exists."""
    try:
        with open(_DICT_ID_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("dict_id") or None
    except Exception:
        return None


def _save_dict_id(dict_id: str):
    os.makedirs(os.path.dirname(_DICT_ID_FILE), exist_ok=True)
    with open(_DICT_ID_FILE, "w", encoding="utf-8") as f:
        json.dump({"dict_id": dict_id}, f, indent=2)


def get_or_create_dict_id(language_code: str = "hi-IN",
                          force_refresh: bool = False) -> str | None:
    """Return a Sarvam dict_id for the pronunciation dictionary.

    Returns cached id if available; otherwise uploads the dictionary and
    caches the returned id. Returns None if upload is not supported or fails.
    """
    from env_utils import load_dotenv
    load_dotenv()
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        print("  No SARVAM_API_KEY — skipping pronunciation dictionary upload.")
        return None

    if not force_refresh:
        cached = _load_cached_dict_id()
        if cached:
            print(f"  Using cached Sarvam dict_id: {cached}")
            return cached

    pronunciations = _load_pronunciations(language_code)
    if not pronunciations:
        print("  Pronunciation dictionary is empty — no upload needed.")
        return None

    print(f"  Uploading {len(pronunciations)} pronunciation entries to Sarvam...")
    dict_id = _upload_to_sarvam(pronunciations, language_code, api_key)
    if dict_id:
        _save_dict_id(dict_id)
        print(f"  Pronunciation dictionary uploaded, dict_id={dict_id}")
    return dict_id


if __name__ == "__main__":
    import sys
    force = "--refresh" in sys.argv
    did = get_or_create_dict_id(force_refresh=force)
    print(f"dict_id = {did!r}")
