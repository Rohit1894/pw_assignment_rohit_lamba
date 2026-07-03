#!/usr/bin/env python3
"""Streamlit front door for the PW AI Video Generator.

This is a thin wrapper around the SAME pipeline the Flask server runs: it saves
the uploaded question image, invokes `main.py --auto-audio ...` as a subprocess,
streams the `[n/10]` step markers into a live progress bar, then plays the
finished MP4. Nothing about the pipeline itself changes — this is only a new UI
that Streamlit Community Cloud can host (it runs `streamlit run`, not Flask), so
you get a free, always-on, shareable link without keeping your PC on.

Deploy notes:
  * API keys come from Streamlit **secrets** (Settings → Secrets), NOT the repo.
    Set GEMINI_API_KEY (or GOOGLE_API_KEY) and SARVAM_API_KEY there. They are
    bridged into os.environ below so the pipeline subprocess inherits them.
  * Whisper is force-skipped by default (SKIP_WHISPER) — it needs ~500 MB extra
    RAM that the free tier doesn't have, and only refines timing by ±1s.
  * System package ffmpeg is provided via packages.txt.

Run locally:  streamlit run streamlit_app.py
"""

import os
import sys
import uuid
import subprocess

import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))


# ── Secrets → environment ─────────────────────────────────────────────────────
# On Streamlit Cloud the API keys live in st.secrets; locally they live in .env.
# Bridge secrets into os.environ (without overriding a real shell/.env value) so
# the pipeline subprocess — which reads os.environ — picks them up either way.
def _bridge_secrets():
    try:
        secrets = dict(st.secrets)
    except Exception:
        secrets = {}
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "SARVAM_API_KEY", "SKIP_WHISPER"):
        val = secrets.get(key)
        if val and not os.environ.get(key):
            os.environ[key] = str(val)
    # Whisper is not viable on the free tier — force-skip unless a secret says otherwise.
    os.environ.setdefault("SKIP_WHISPER", "1")


_bridge_secrets()

# Load .env too (harmless on the cloud where the file is absent; helps local runs).
try:
    from env_utils import load_dotenv
    load_dotenv()
    os.environ.setdefault("SKIP_WHISPER", "1")
except Exception:
    pass


# ── Pipeline step markers → (label, progress fraction) ────────────────────────
STEP_MAP = [
    ("[1/10]",   "Preparing whiteboard canvas",        0.08),
    ("[2/10]",   "Reading & understanding question",   0.18),
    ("[3/10]",   "Solving step-by-step",               0.28),
    ("[4/10]",   "Verifying solution",                 0.36),
    ("[5/10]",   "Generating storyboard",              0.46),
    ("[5.5/10]", "Layout planning + glyph check",      0.52),
    ("[5.7/10]", "Building pronunciation dictionary",  0.56),
    ("[6/10]",   "Generating narration audio (TTS)",   0.66),
    ("[7/10]",   "Timing refinement",                  0.74),
    ("[8/10]",   "Building timeline",                  0.82),
    ("[9/10]",   "Validating before render",           0.88),
    ("[10/10]",  "Rendering whiteboard video",         0.96),
    ("Done!",    "Finalising MP4",                     1.00),
]

SPEAKERS = ["shubh", "manisha", "vidya", "arya", "karun", "hitesh"]


def _keys_present():
    gemini = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return bool(gemini), bool(os.environ.get("SARVAM_API_KEY"))


def _run_pipeline(input_path, output_path, job_dir, language, speaker, pace,
                  progress_bar, status_box, log_box):
    """Run main.py as a subprocess, streaming step progress into the UI.
    Returns (ok, tail_log)."""
    cmd = [
        sys.executable, "-u", os.path.join(BASE_DIR, "main.py"),
        "--image",           input_path,
        "--auto-audio",
        "--language",        language,
        "--tts-provider",    "sarvam",
        "--sarvam-speaker",  speaker,
        "--tts-pace",        str(pace),
        "--output",          output_path,
        "--auto-script",     os.path.join(job_dir, "storyboard.json"),
        "--auto-audio-path", os.path.join(job_dir, "auto_narration.wav"),
        "--layout",          os.path.join(job_dir, "layout.json"),
        "--annotations",     os.path.join(job_dir, "annotations.json"),
        "--transcript",      os.path.join(job_dir, "transcript.json"),
        "--contact-sheet",   os.path.join(job_dir, "contact_sheet.jpg"),
        "--skip-whisper",
        "--allow-unverified",
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    log_lines = []
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        cwd=BASE_DIR, env=env,
    )
    for raw in proc.stdout:
        line = raw.rstrip()
        if not line:
            continue
        log_lines.append(line)
        for marker, label, frac in STEP_MAP:
            if marker in line:
                progress_bar.progress(frac, text=label)
                status_box.info(f"⏳ {label}…")
                break
        # Keep the on-screen log to a readable tail.
        log_box.code("\n".join(log_lines[-14:]), language="text")
    proc.wait()

    ok = proc.returncode == 0 and os.path.exists(output_path)
    return ok, "\n".join(log_lines[-12:])


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PW AI Video Generator", page_icon="🎬",
                   layout="centered")

st.title("🎬 PW AI Video Generator")
st.caption("Upload a question image → get a narrated whiteboard explanation video.")

gemini_ok, sarvam_ok = _keys_present()
if not (gemini_ok and sarvam_ok):
    missing = []
    if not gemini_ok:
        missing.append("GEMINI_API_KEY (or GOOGLE_API_KEY)")
    if not sarvam_ok:
        missing.append("SARVAM_API_KEY")
    st.error(
        "Missing API key(s): " + ", ".join(missing) +
        ".\n\nOn Streamlit Cloud, add them under **Settings → Secrets**. "
        "Locally, put them in a `.env` file at the project root."
    )

with st.form("generate"):
    uploaded = st.file_uploader(
        "Question image", type=["png", "jpg", "jpeg", "webp"],
        help="A clear photo/screenshot of a single question.",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        language = st.selectbox("Teaching language", ["hinglish", "hindi", "english"], index=0)
    with col2:
        speaker = st.selectbox("Voice", SPEAKERS, index=0)
    with col3:
        pace = st.slider("Pace", 0.7, 1.2, 0.92, 0.02,
                         help="Lower = slower, more teacher-like.")
    submitted = st.form_submit_button("Generate video", type="primary",
                                      use_container_width=True)

if submitted:
    if uploaded is None:
        st.warning("Please upload a question image first.")
        st.stop()
    if not (gemini_ok and sarvam_ok):
        st.warning("Add the missing API key(s) above before generating.")
        st.stop()

    job_id = uuid.uuid4().hex[:10]
    upload_dir = os.path.join(BASE_DIR, "input", "uploads")
    job_dir = os.path.join(BASE_DIR, "output", "jobs", job_id)
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(job_dir, exist_ok=True)

    ext = os.path.splitext(uploaded.name)[1].lower() or ".png"
    input_path = os.path.join(upload_dir, f"{job_id}{ext}")
    output_path = os.path.join(job_dir, "final.mp4")
    with open(input_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.info("Generating… this takes a few minutes (rendering is CPU-bound). "
            "Keep this tab open.")
    progress_bar = st.progress(0.0, text="Starting pipeline…")
    status_box = st.empty()
    with st.expander("Live log", expanded=False):
        log_box = st.empty()

    try:
        ok, tail = _run_pipeline(input_path, output_path, job_dir, language,
                                 speaker, pace, progress_bar, status_box, log_box)
    except Exception as exc:  # noqa: BLE001
        ok, tail = False, str(exc)

    if ok:
        progress_bar.progress(1.0, text="Done")
        status_box.success("✅ Video ready!")
        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download MP4", f, file_name=f"pw_explanation_{job_id}.mp4",
                               mime="video/mp4", use_container_width=True)
    else:
        status_box.error("❌ Generation failed.")
        st.code(tail or "No output captured.", language="text")

st.divider()
st.caption("Powered by the PW pipeline (Gemini · Sarvam TTS · PIL/MoviePy render). "
           "Whisper timing-refinement is skipped on this host for speed/memory.")
