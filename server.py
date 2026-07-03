#!/usr/bin/env python3
"""Flask API server for the PW AI Video Generator frontend.

Endpoints:
  GET  /                      → serve frontend/index.html
  POST /api/generate          → upload file + options, start pipeline, return job_id
  GET  /api/status/<job_id>   → poll current progress (JSON)
  GET  /api/video/<job_id>    → stream final MP4 for in-browser playback
  GET  /api/download/<job_id> → download final MP4 as attachment
  GET  /api/log/<job_id>      → raw pipeline stdout log (JSON)

Run:
  .venv\\Scripts\\python.exe server.py
  Then open http://localhost:5000
"""

import os
import sys
import uuid
import threading
import subprocess
import mimetypes

from flask import Flask, request, jsonify, send_file, send_from_directory

# ── Bootstrap ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

from env_utils import load_dotenv
load_dotenv()

app = Flask(__name__, static_folder="frontend", static_url_path="/static")

# In-memory job store  {job_id -> dict}
jobs: dict[str, dict] = {}

# ── Pipeline step → (display label, progress %) ───────────────────────────────
STEP_MAP = [
    ("[1/10]",    "Preparing whiteboard canvas",        8),
    ("[2/10]",    "Reading & understanding question",   18),
    ("[3/10]",    "Solving step-by-step",               28),
    ("[4/10]",    "Verifying solution",                 36),
    ("[5/10]",    "Generating storyboard",              46),
    ("[5.5/10]",  "Layout planning + glyph check",     52),
    ("[5.7/10]",  "Building pronunciation dictionary",  56),
    ("[6/10]",    "Generating narration audio (TTS)",   66),
    ("[7/10]",    "Transcribing narration (Whisper)",   74),
    ("[8/10]",    "Building timeline",                  82),
    ("[9/10]",    "Validating before render",           88),
    ("[10/10]",   "Rendering whiteboard video",         96),
    ("Done!",     "Finalising MP4",                    100),
]

ALLOWED_IMAGE_EXT  = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_DOC_EXT    = {".pdf", ".ppt", ".pptx"}

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file uploaded"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXT | ALLOWED_DOC_EXT:
        return jsonify({"error": f"Unsupported file type '{ext}'"}), 400
    if ext in ALLOWED_DOC_EXT:
        return jsonify({
            "error": "PDF/PPT support is coming soon. Please upload an image (JPG/PNG) for now."
        }), 400

    language  = request.form.get("language",  "hinglish")
    speaker   = request.form.get("speaker",   "shubh")
    pace      = float(request.form.get("pace", "0.92"))

    job_id = uuid.uuid4().hex[:10]

    # Per-job upload + output dirs (prevents concurrent runs clobbering each other)
    upload_dir = os.path.join(BASE_DIR, "input", "uploads")
    job_dir    = os.path.join(BASE_DIR, "output", "jobs", job_id)
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(job_dir,    exist_ok=True)

    input_path  = os.path.join(upload_dir, f"{job_id}{ext}")
    output_path = os.path.join(job_dir, "final.mp4")
    file.save(input_path)

    jobs[job_id] = {
        "status":       "running",
        "progress":     0,
        "current_step": "Starting pipeline…",
        "completed_steps": [],
        "output_path":  output_path,
        "error":        None,
        "log":          [],
    }

    t = threading.Thread(
        target=_run_pipeline,
        args=(job_id, input_path, output_path, job_dir, language, speaker, pace),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status":           job["status"],
        "progress":         job["progress"],
        "current_step":     job["current_step"],
        "completed_steps":  job["completed_steps"],
        "error":            job["error"],
    })


@app.route("/api/video/<job_id>")
def api_video(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "done":
        return jsonify({"error": "Video not ready yet"}), 202
    path = job["output_path"]
    if not os.path.exists(path):
        return jsonify({"error": "Video file missing on server"}), 500
    return send_file(path, mimetype="video/mp4", conditional=True)


@app.route("/api/download/<job_id>")
def api_download(job_id):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Video not ready"}), 404
    path = job["output_path"]
    return send_file(
        path, mimetype="video/mp4",
        as_attachment=True,
        download_name=f"pw_explanation_{job_id}.mp4",
    )


@app.route("/api/log/<job_id>")
def api_log(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    return jsonify({"log": job["log"]})


# ── Pipeline runner (background thread) ───────────────────────────────────────

def _run_pipeline(job_id, input_path, output_path, job_dir, language, speaker, pace):
    job = jobs[job_id]

    cmd = [
        sys.executable, "-u", os.path.join(BASE_DIR, "main.py"),
        "--image",           input_path,
        "--auto-audio",
        "--language",        language,
        "--tts-provider",    "sarvam",
        "--sarvam-speaker",  speaker,
        "--tts-pace",        str(pace),
        "--output",          output_path,
        # Per-job intermediate artifact paths so concurrent runs don't clash
        "--auto-script",     os.path.join(job_dir, "storyboard.json"),
        "--auto-audio-path", os.path.join(job_dir, "auto_narration.wav"),
        "--layout",          os.path.join(job_dir, "layout.json"),
        "--annotations",     os.path.join(job_dir, "annotations.json"),
        "--transcript",      os.path.join(job_dir, "transcript.json"),
        "--contact-sheet",   os.path.join(job_dir, "contact_sheet.jpg"),
        "--allow-unverified",
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            cwd=BASE_DIR, env=env,
        )

        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            job["log"].append(line)

            for marker, label, pct in STEP_MAP:
                if marker in line:
                    if job["current_step"] != "Starting pipeline…":
                        job["completed_steps"].append(job["current_step"])
                    job["progress"]     = pct
                    job["current_step"] = label
                    break

        proc.wait()

        if proc.returncode == 0 and os.path.exists(output_path):
            job["status"]       = "done"
            job["progress"]     = 100
            job["current_step"] = "Complete"
            job["completed_steps"].append("Rendering whiteboard video")
        else:
            # Grab last few lines of log for a readable error
            tail = "\n".join(job["log"][-6:])
            job["status"] = "error"
            job["error"]  = f"Pipeline exited with code {proc.returncode}.\n{tail}"

    except Exception as exc:
        job["status"] = "error"
        job["error"]  = str(exc)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("PW AI Video Generator — backend server")
    print("Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
