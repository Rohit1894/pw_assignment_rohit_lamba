# Deploy to Streamlit Community Cloud (free, always-on, shareable link)

This app now has a Streamlit front door (`streamlit_app.py`) that wraps the exact
same pipeline the Flask server runs. Streamlit Community Cloud hosts it for free
with a public URL — **no need to keep your PC on**.

## What's already set up for you
| File | Purpose |
|------|---------|
| `streamlit_app.py` | The UI Streamlit Cloud runs (`streamlit run`). Uploads an image, runs the pipeline, plays the video. |
| `requirements.txt` | **Slim deploy set** — no Whisper (saves ~500 MB RAM). Uses `opencv-python-headless`. |
| `requirements-local.txt` | Full local set **with** Whisper (`pip install -r requirements-local.txt`). |
| `packages.txt` | System `ffmpeg` (needed by the renderer). |
| `.streamlit/config.toml` | Upload limit + theme. |
| `.streamlit/secrets.toml.example` | Template for your API keys (real keys go in the Cloud UI, never committed). |
| Whisper skip | `SKIP_WHISPER=1` is forced by the app + `main.py --skip-whisper`. |

## Steps

### 1. Push the repo to GitHub
The repo already `.gitignore`s `.env`, `input/`, `output/`, and the real
`.streamlit/secrets.toml`. Make sure `config/` and `fonts/` **are** committed
(the pipeline needs them):
```bash
git add -A
git status            # confirm config/ and fonts/ are staged; .env is NOT
git commit -m "Add Streamlit deployment"
git push
```

### 2. Create the app on Streamlit Cloud
1. Go to https://share.streamlit.io → **Sign in with GitHub**.
2. **Create app** → **Deploy a public app from GitHub**.
3. Pick your repo/branch, set **Main file path** = `streamlit_app.py`.
4. Click **Deploy**.

### 3. Add your API keys as Secrets
In the app → **⋮ / Settings → Secrets**, paste (from
`.streamlit/secrets.toml.example`):
```toml
GEMINI_API_KEY = "your-gemini-key"
SARVAM_API_KEY = "your-sarvam-key"
SKIP_WHISPER = "1"
```
Save — the app reboots and picks them up. **Share the app URL** with anyone.

## Test it locally first (optional)
```bash
.venv\Scripts\python.exe -m pip install streamlit
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```
Keys come from your existing `.env` locally.

## ⚠️ The one real risk: RAM
Streamlit's free tier is **~1 GB RAM**. This app needs `torch` (via **EasyOCR**
for reading the question) — that's the heavy part, and it stays even with Whisper
removed. Skipping Whisper buys headroom, but OCR + render may still approach the
limit on a big image.

If the app crashes/reboots mid-run with an out-of-memory or "app went over its
resource limits" message, the realistic fixes are, in order:
1. **Upload smaller images** (resize to ~1000px wide) — less OCR memory.
2. **Move to a bigger free-ish tier** — Hugging Face Spaces (16 GB RAM, also free)
   runs this comfortably; the same `streamlit_app.py` works there in a Docker Space.
3. **Swap EasyOCR for a lighter OCR** (e.g. the Gemini vision OCR path) to drop torch.

Rendering is CPU-only on any free tier, so each video still takes a few minutes —
that's expected, not a hang.
