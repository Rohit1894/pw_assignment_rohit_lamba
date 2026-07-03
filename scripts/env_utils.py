#!/usr/bin/env python3
"""Tiny zero-dependency .env loader.

Loads KEY=VALUE pairs from the project-root `.env` file into os.environ so API
keys don't have to be exported in every shell session. Values already present
in the environment ALWAYS win — a shell `$env:GEMINI_API_KEY=...` overrides the
file, and an empty value in the file never blanks a real key.

Called automatically by main.py and by the modules that read API keys
(gemini_utils, generate_audio_sarvam), so standalone script runs also pick up
the .env. Safe to call repeatedly.
"""

import os

# Project root = parent of the scripts/ directory this module lives in.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dotenv(path=None):
    """Load `.env` into os.environ (existing env vars are never overridden).

    Looks in the project root by default, then the current working directory.
    Returns the number of variables actually set. Missing file is fine.
    """
    candidates = ([path] if path else
                  [os.path.join(_PROJECT_ROOT, ".env"),
                   os.path.join(os.getcwd(), ".env")])
    loaded = 0
    for cand in candidates:
        if not cand or not os.path.isfile(cand):
            continue
        try:
            with open(cand, encoding="utf-8-sig") as f:
                lines = f.readlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key.lower().startswith("export "):
                key = key[7:].strip()
            value = value.strip().strip('"').strip("'")
            # Never override a real environment variable, and never set an
            # empty placeholder (so `KEY=` lines in the template are inert).
            if key and value and not os.environ.get(key):
                os.environ[key] = value
                loaded += 1
        break  # first .env found wins; don't merge multiple files
    return loaded


if __name__ == "__main__":
    n = load_dotenv()
    print(f"Loaded {n} variable(s) from .env")
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "SARVAM_API_KEY"):
        v = os.environ.get(k)
        print(f"  {k}: {'set (' + v[:6] + '...)' if v else 'NOT SET'}")
