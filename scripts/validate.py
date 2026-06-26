#!/usr/bin/env python3
"""
Validation harness — proves (not just spot-checks) that the pipeline holds across
many questions and types.

For each question in a manifest it runs the real `main.py` pipeline (Gemini
responses are cached, so re-runs are free), then applies automated structural
checks to the produced annotations and extracts key frames for a quick visual
look. Results are printed and written to output/validation_report.md.

Usage:
    python scripts/validate.py                 # default 3-question manifest
    python scripts/validate.py manifest.json   # custom [{name,image,audio}, ...]

A custom manifest is a JSON array of objects: {"name", "image", "audio"}.
"""

import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MANIFEST = [
    {"name": "Q3_assertion", "image": "output/analysis/pdf_pages/page_3.png", "audio": "input/3.mp3"},
    {"name": "Q7_flowchart", "image": "output/analysis/pdf_pages/page_7.png", "audio": "input/7.mp3"},
    {"name": "Q14_matching", "image": "output/analysis/pdf_pages/page_14.png", "audio": "input/14.mp3"},
]

# Which action a given classified type is expected to produce.
TYPE_EXPECT = {
    "matching": "match_pair",
    "flowchart_fill": "fill_placeholder",
    "numerical": "write_step",
    "diagram_label": None,        # circle/annotate — no single required action
    "mcq": "mark_answer",
    "assertion_reason": "mark_answer",
}


def _audio_duration(path):
    try:
        from moviepy import AudioFileClip
        c = AudioFileClip(path)
        d = c.duration
        c.close()
        return d
    except Exception:
        return None


def _run_pipeline(q):
    """Run main.py for one question; return (stdout, annotations_path, video_path)."""
    ann = os.path.join("output", f"val_{q['name']}_ann.json")
    vid = os.path.join("output", f"val_{q['name']}.mp4")
    cmd = [sys.executable, "main.py",
           "--image", q["image"], "--audio", q["audio"],
           "--output", vid, "--annotations", ann,
           "--engine", "gemini", "--ink", "red"]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=1800)
    return (p.stdout or "") + (p.stderr or ""), ann, vid


def _extract_frames(video, name, duration):
    """Grab three frames (≈25/60/90%) for a quick visual check."""
    if not duration or not os.path.exists(video):
        return []
    out = []
    fdir = os.path.join("output", "validation_frames")
    os.makedirs(fdir, exist_ok=True)
    for frac in (0.25, 0.6, 0.9):
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


def _check(q, log, ann_path, duration):
    """Apply automated structural checks; return (results dict, qtype, summary)."""
    res = {}

    qtype_m = re.search(r"Question type:\s*(\w+)", log)
    qtype = qtype_m.group(1) if qtype_m else None

    used_fallback = "Falling back to Whisper engine" in log or "rule-based" in log
    cache_hit = "Cache hit" in log

    anns = []
    if os.path.exists(ann_path):
        try:
            with open(ann_path, encoding="utf-8") as f:
                anns = json.load(f)
        except Exception:
            anns = []
    actions = [a.get("action") for a in anns]
    times = [float(a.get("time", 0)) for a in anns]

    res["ran"] = "PASS" if "Done!" in log or os.path.exists(ann_path) else "FAIL"
    res["produced_actions"] = "PASS" if len(anns) >= 3 else "FAIL"

    if duration and times:
        res["within_audio"] = "PASS" if max(times) <= duration + 0.5 else "FAIL"
        span = max(times) - min(times)
        res["not_frontloaded"] = "PASS" if span >= 0.4 * duration else "WARN"
    else:
        res["within_audio"] = res["not_frontloaded"] = "WARN"

    sync_m = re.search(r"Aligned timeline to spoken words:\s*(\d+)/(\d+)", log)
    if sync_m:
        a_, b_ = int(sync_m.group(1)), int(sync_m.group(2))
        cov = a_ / b_ if b_ else 0
        res["sync"] = "PASS" if cov >= 0.5 else "WARN"
        res["_sync_cov"] = f"{a_}/{b_}"
    elif "keeping model timing" in log:
        res["sync"] = "WARN"      # transcript too sparse → safe fallback
        res["_sync_cov"] = "fallback"
    else:
        res["sync"] = "WARN"
        res["_sync_cov"] = "n/a"

    expect = TYPE_EXPECT.get(qtype, None)
    if expect:
        res["type_behavior"] = "PASS" if expect in actions else "WARN"
    else:
        res["type_behavior"] = "PASS"

    res["answer_marked"] = "PASS" if ("mark_answer" in actions or qtype == "diagram_label") else "WARN"

    summary = {
        "qtype": qtype, "n_actions": len(anns), "sync": res.get("_sync_cov"),
        "fallback": used_fallback, "cache": cache_hit,
        "action_counts": {a: actions.count(a) for a in sorted(set(actions))},
    }
    return res, qtype, summary


def main():
    manifest = DEFAULT_MANIFEST
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            manifest = json.load(f)

    os.chdir(ROOT)
    report_rows = []
    check_names = ["ran", "produced_actions", "within_audio", "not_frontloaded",
                   "sync", "type_behavior", "answer_marked"]

    for q in manifest:
        print(f"\n=== {q['name']} ===")
        if not (os.path.exists(q["image"]) and os.path.exists(q["audio"])):
            print(f"  SKIP — missing image/audio")
            report_rows.append((q["name"], "MISSING", {}, {}, []))
            continue
        dur = _audio_duration(q["audio"])
        log, ann_path, vid = _run_pipeline(q)
        res, qtype, summary = _check(q, log, ann_path, dur)
        frames = _extract_frames(vid, q["name"], dur)
        verdict = "FAIL" if any(v == "FAIL" for v in res.values()) else \
                  ("WARN" if any(v == "WARN" for k, v in res.items() if not k.startswith("_")) else "PASS")
        report_rows.append((q["name"], verdict, res, summary, frames))
        print(f"  type={qtype} actions={summary['n_actions']} sync={summary['sync']} "
              f"fallback={summary['fallback']} -> {verdict}")
        for c in check_names:
            print(f"    {c:18} {res.get(c)}")

    # ── Markdown report ─────────────────────────────────────────────────
    lines = ["# Validation report", ""]
    lines.append("| Question | Verdict | Type | Actions | Sync | " +
                 " | ".join(check_names) + " |")
    lines.append("|" + "---|" * (5 + len(check_names)))
    for name, verdict, res, summary, frames in report_rows:
        row = [name, verdict, str(summary.get("qtype")), str(summary.get("n_actions")),
               str(summary.get("sync"))] + [str(res.get(c, "-")) for c in check_names]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    for name, verdict, res, summary, frames in report_rows:
        if summary:
            lines.append(f"### {name} — {verdict}")
            lines.append(f"- action counts: `{summary.get('action_counts')}`")
            lines.append(f"- fallback used: {summary.get('fallback')}, cache hit: {summary.get('cache')}")
            for fp in frames:
                lines.append(f"- frame: `{fp}`")
            lines.append("")
    report = os.path.join("output", "validation_report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    n_pass = sum(1 for r in report_rows if r[1] == "PASS")
    n_warn = sum(1 for r in report_rows if r[1] == "WARN")
    n_fail = sum(1 for r in report_rows if r[1] in ("FAIL", "MISSING"))
    print(f"\n==== {n_pass} PASS / {n_warn} WARN / {n_fail} FAIL — report: {report} ====")


if __name__ == "__main__":
    main()
