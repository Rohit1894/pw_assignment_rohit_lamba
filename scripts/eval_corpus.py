#!/usr/bin/env python3
"""
English multi-subject evaluation harness.

Runs a manifest of questions (ideally English: physics / chemistry / maths /
biology) through the REAL `main.py` pipeline, then scores each result on quality
signals — including the things the recent English/all-subject work targets:

  * english_script  — written text is Latin (not Devanagari) on an English slide
  * write_step      — numerical questions show a worked solution
  * frac            — stacked fractions (\\frac{}{}) are used where useful
  * answer_marked / within_audio / not_frontloaded — core correctness/timing

It writes a per-subject markdown report and extracts 3 frames per question for a
quick visual check. Use this to validate the prompt + font + fraction changes and
to drive the render->watch->fix loop once a GEMINI_API_KEY is available.

Usage:
    python scripts/eval_corpus.py                      # default manifest
    python scripts/eval_corpus.py my_manifest.json     # custom manifest
    python scripts/eval_corpus.py --reuse              # render existing
                                                       # annotations (no Gemini)

Manifest = JSON array of objects:
    {"name": "phys_dim", "subject": "physics",
     "image": "input/eng_img_test.png", "audio": "input/eng_audio_test.mp3",
     "annotations": "output/eng_test_annotations.json"}   # optional; for --reuse

Full mode (no --reuse) needs GEMINI_API_KEY and re-runs generation per question.
Entries whose image/audio are missing are SKIPPED with a note (so the default
manifest doubles as a checklist of subject inputs still to add).
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join("output", "eval_manifest.json")

# Shipped starter manifest: the one English question we have + placeholders for
# the other subjects (fill in your own English image+audio and they light up).
STARTER = [
    {"name": "phys_dimensional", "subject": "physics",
     "image": "input/eng_img_test.png", "audio": "input/eng_audio_test.mp3",
     "annotations": "output/eng_test_annotations.json"},
    {"name": "chem_example", "subject": "chemistry",
     "image": "input/chem_en.png", "audio": "input/chem_en.mp3"},
    {"name": "maths_example", "subject": "maths",
     "image": "input/maths_en.png", "audio": "input/maths_en.mp3"},
    {"name": "bio_example", "subject": "biology",
     "image": "input/bio_en.png", "audio": "input/bio_en.mp3"},
]

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _ensure_manifest(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(STARTER, f, indent=2, ensure_ascii=False)
    print(f"  created starter manifest -> {path} (edit it to add your subjects)")
    return STARTER


def _audio_duration(path):
    try:
        from moviepy import AudioFileClip
        c = AudioFileClip(path)
        d = c.duration
        c.close()
        return d
    except Exception:
        return None


def _run_pipeline(q, reuse):
    """Run main.py for one question; return (log, annotations_path, video_path)."""
    ann = q.get("annotations") or os.path.join("output", f"eval_{q['name']}_ann.json")
    vid = os.path.join("output", f"eval_{q['name']}.mp4")
    cmd = [sys.executable, "main.py",
           "--image", q["image"], "--audio", q["audio"],
           "--output", vid, "--annotations", ann,
           "--language", "en", "--ink", "red"]
    cmd += ["--reuse-annotations"] if reuse else ["--engine", "gemini"]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=1800)
    return (p.stdout or "") + (p.stderr or ""), ann, vid


def _analyze(ann_path, duration):
    """Load annotations + meta sidecar and compute quality signals."""
    sig = {}
    anns = []
    if os.path.exists(ann_path):
        try:
            with open(ann_path, encoding="utf-8") as f:
                anns = json.load(f)
        except Exception:
            anns = []
    meta_path = os.path.splitext(ann_path)[0] + ".meta.json"
    qtype = None
    try:
        with open(meta_path, encoding="utf-8") as f:
            qtype = (json.load(f) or {}).get("question_type")
    except Exception:
        pass

    actions = [a.get("action") for a in anns if isinstance(a, dict)]
    texts = " ".join(str(a.get("text", "")) for a in anns if isinstance(a, dict))
    times = [float(a.get("time", 0)) for a in anns if isinstance(a, dict)]

    sig["n_actions"] = len(anns)
    sig["qtype"] = qtype
    sig["answer_marked"] = "mark_answer" in actions or qtype == "diagram_label"
    # English slide should yield Latin text, not Devanagari.
    deva = len(_DEVANAGARI.findall(texts))
    latin = sum(1 for c in texts if "a" <= c.lower() <= "z")
    sig["english_script"] = (deva == 0) if (latin + deva) else None
    # feature signals from the recent work
    sig["n_write_step"] = actions.count("write_step")
    sig["n_frac"] = texts.count("\\frac") + texts.count("\\dfrac") + texts.count("\\tfrac")
    sig["n_diagram"] = actions.count("draw_diagram")
    sig["n_underline"] = actions.count("underline_existing")
    if duration and times:
        sig["within_audio"] = max(times) <= duration + 0.5
        sig["not_frontloaded"] = (max(times) - min(times)) >= 0.4 * duration
    else:
        sig["within_audio"] = sig["not_frontloaded"] = None
    return sig


def _extract_frames(video, name, duration):
    if not duration or not os.path.exists(video):
        return []
    out, fdir = [], os.path.join("output", "eval_frames")
    os.makedirs(fdir, exist_ok=True)
    for frac in (0.3, 0.6, 0.9):
        t = max(1, int(duration * frac))
        fp = os.path.join(fdir, f"{name}_{int(frac*100)}.png")
        try:
            subprocess.run(["ffmpeg", "-y", "-ss", str(t), "-i", video,
                            "-frames:v", "1", fp, "-loglevel", "error"],
                           cwd=ROOT, timeout=120)
            if os.path.exists(fp):
                out.append(fp)
        except Exception:
            pass
    return out


def _verdict(sig, ran):
    if not ran or sig["n_actions"] < 3:
        return "FAIL"
    bad = (sig["answer_marked"] is False
           or sig["english_script"] is False
           or sig["within_audio"] is False)
    return "WARN" if bad else "PASS"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    reuse = "--reuse" in sys.argv
    manifest_path = args[0] if args else DEFAULT_MANIFEST
    os.chdir(ROOT)
    manifest = _ensure_manifest(manifest_path)
    if not reuse and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("  NOTE: no GEMINI_API_KEY set — full generation will fail. Run with "
              "--reuse to render existing annotations, or set the key.\n")

    rows = []
    for q in manifest:
        name, subject = q["name"], q.get("subject", "?")
        print(f"\n=== {subject}/{name} ===")
        if not (os.path.exists(q["image"]) and os.path.exists(q["audio"])):
            print(f"  SKIP — add input files: image={q['image']} audio={q['audio']}")
            rows.append((subject, name, "MISSING", {}, []))
            continue
        if reuse and not (q.get("annotations") and os.path.exists(q["annotations"])):
            print(f"  SKIP (--reuse) — no existing annotations for {name}")
            rows.append((subject, name, "MISSING", {}, []))
            continue
        try:
            dur = _audio_duration(q["audio"])
            log, ann, vid = _run_pipeline(q, reuse)
            ran = "Done!" in log or os.path.exists(vid)
            sig = _analyze(ann, dur)
            frames = _extract_frames(vid, name, dur)
            verdict = _verdict(sig, ran)
            rows.append((subject, name, verdict, sig, frames))
            print(f"  -> {verdict}  type={sig['qtype']} actions={sig['n_actions']} "
                  f"english={sig['english_script']} write_step={sig['n_write_step']} "
                  f"frac={sig['n_frac']} answer={sig['answer_marked']}")
        except subprocess.TimeoutExpired:
            print("  FAIL — pipeline timed out")
            rows.append((subject, name, "FAIL", {}, []))
        except Exception as e:
            print(f"  FAIL — {e}")
            rows.append((subject, name, "FAIL", {}, []))

    _write_report(rows, reuse)


def _write_report(rows, reuse):
    cols = ["english_script", "n_actions", "qtype", "n_write_step", "n_frac",
            "n_diagram", "answer_marked", "within_audio", "not_frontloaded"]
    lines = ["# English multi-subject eval report",
             f"_mode: {'reuse (no Gemini)' if reuse else 'full pipeline'}_", ""]
    by_subject = {}
    for subject, name, verdict, sig, frames in rows:
        by_subject.setdefault(subject, []).append((name, verdict, sig, frames))
    # summary table
    lines.append("| Subject | Question | Verdict | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (3 + len(cols)))
    for subject in sorted(by_subject):
        for name, verdict, sig, frames in by_subject[subject]:
            row = [subject, name, verdict] + [str(sig.get(c, "-")) for c in cols]
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    for subject in sorted(by_subject):
        lines.append(f"### {subject}")
        for name, verdict, sig, frames in by_subject[subject]:
            lines.append(f"- **{name}** — {verdict}" + (f" (actions={sig.get('n_actions')}, "
                         f"write_step={sig.get('n_write_step')}, frac={sig.get('n_frac')})" if sig else ""))
            for fp in frames:
                lines.append(f"  - frame: `{fp}`")
        lines.append("")
    report = os.path.join("output", "eval_report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    npass = sum(1 for r in rows if r[2] == "PASS")
    nwarn = sum(1 for r in rows if r[2] == "WARN")
    nbad = sum(1 for r in rows if r[2] in ("FAIL", "MISSING"))
    print(f"\n==== {npass} PASS / {nwarn} WARN / {nbad} FAIL|MISSING — report: {report} ====")


if __name__ == "__main__":
    main()
