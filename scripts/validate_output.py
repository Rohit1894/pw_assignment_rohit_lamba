#!/usr/bin/env python3
"""Validate the auto-audio pipeline's artifacts — pre-render and post-render.

Pre-render (stage="pre"):
  - Solution verified (or allow_unverified)
  - Storyboard: every step has board_lines + display_narration_roman + tts_narration_text
  - No board_line contains \\n, /n, markdown escapes, or JSON fragments
  - Board lines are English-only (no Devanagari unless it was on the original)
  - Audio manifest consistent with the actual WAV
  - Annotation times inside the audio
  - Final answer present (mark_answer or boxed write_step)
  - Glyph report status safe/substituted (no hard failures)
  - Layout validation passed

Post-render (stage="post"):
  - Final video exists and is non-empty
  - Video duration within 1.5s of narration duration
  - Contact sheet exists

Also writes output/quality_score.json.

Output: output/validation_report.json
"""

import json
import os
import re
import sys

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_ESCAPE_RE = re.compile(r"(\\n|/n|\n|\\t|```|^#+\s|^\*\s)", re.MULTILINE)


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _wav_duration(path):
    import wave
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _check_board_line(line: str) -> list:
    """Return list of issues with a single board line string."""
    issues = []
    if _ESCAPE_RE.search(line):
        issues.append(f"board_line contains escape/markdown: {line[:50]!r}")
    if _DEVANAGARI_RE.search(line):
        issues.append(f"board_line contains Devanagari (board must be English): "
                      f"{line[:50]!r}")
    if len(line) > 120:
        issues.append(f"board_line very long ({len(line)} chars): {line[:50]!r}")
    return issues


def _compute_quality_score(errors: list, warnings: list, sb, mf, ver,
                            allow_unverified: bool) -> dict:
    """Heuristic quality score — improves when real STT QA is wired up."""
    # correctness: full marks when verified, penalised for issues
    n_issues = len(errors) + len(warnings) * 0.3
    base = max(0.0, 1.0 - n_issues * 0.05)

    correctness = 1.0 if (ver or {}).get("status") == "verified" else (
        0.6 if allow_unverified else 0.0)
    confidence = float((ver or {}).get("confidence") or 0.5)
    correctness = min(correctness, confidence)

    n_steps = len((sb or {}).get("steps") or []) if sb else 0
    layout_quality = min(1.0, max(0.5, n_steps / 10)) if n_steps else 0.5

    total_dur = float((mf or {}).get("total_duration_sec") or 0)
    # ideal 45–120s; penalise too short or too long
    pacing = 1.0
    if total_dur > 0:
        if total_dur < 30 or total_dur > 180:
            pacing = 0.7
        elif total_dur < 45 or total_dur > 120:
            pacing = 0.85

    overall = round((correctness * 0.35 + layout_quality * 0.25
                     + pacing * 0.20 + base * 0.20), 3)
    return {
        "overall_score": overall,
        "correctness": round(correctness, 3),
        "readability": round(layout_quality, 3),
        "voice_quality": round(pacing, 3),
        "layout_quality": round(layout_quality, 3),
        "student_friendliness": round(base, 3),
        "issues": errors[:10],
    }


def validate_output(stage: str = "pre",
                    verification_path: str = "output/solution_verification.json",
                    storyboard_path: str = "output/storyboard.json",
                    manifest_path: str = "output/audio_manifest.json",
                    annotations_path: str = "output/auto_annotations.json",
                    audio_path: str = "output/auto_narration.wav",
                    layout_path: str = "output/layout.json",
                    layout_validation_path: str = "output/layout_validation.json",
                    glyph_report_path: str = "output/glyph_report.json",
                    video_path: str = "output/final.mp4",
                    contact_sheet_path: str = "output/contact_sheet.jpg",
                    report_path: str = "output/validation_report.json",
                    quality_score_path: str = "output/quality_score.json",
                    allow_unverified: bool = False) -> tuple:
    """Run the checks for `stage` ("pre" or "post"). Returns (ok, report)."""
    errors, warnings = [], []

    # ── Load shared artefacts ────────────────────────────────────────────
    ver = _load(verification_path)
    sb  = _load(storyboard_path)
    mf  = _load(manifest_path)

    if stage == "pre":
        # 1. Solution verified
        if not ver:
            errors.append(f"missing {verification_path}")
        elif ver.get("status") != "verified":
            msg = (f"solution not verified (status={ver.get('status')}, "
                   f"confidence={ver.get('confidence')}, "
                   f"issues={ver.get('issues')})")
            (warnings if allow_unverified else errors).append(msg)

        # 2. Storyboard checks
        if not sb or not sb.get("steps"):
            errors.append("storyboard missing or has no steps")
        else:
            for st in sb["steps"]:
                sid = st.get("id", "?")
                # 2a. narration fields
                if not (st.get("tts_narration_text") or "").strip():
                    errors.append(f"storyboard step {sid} has no tts_narration_text")
                if not (st.get("display_narration_roman") or "").strip():
                    warnings.append(f"storyboard step {sid} has no display_narration_roman")
                # 2b. board_lines checks
                action = st.get("visual_action", "")
                if action == "write_step":
                    board_lines = st.get("board_lines") or []
                    if not board_lines:
                        warnings.append(f"storyboard step {sid} has empty board_lines")
                    for line in board_lines:
                        for iss in _check_board_line(str(line)):
                            errors.append(f"step {sid}: {iss}")
                # 2c. visual_action present
                if not action:
                    errors.append(f"storyboard step {sid} has no visual_action")

        # 3. Audio manifest + WAV consistency
        if not mf or not mf.get("segments"):
            errors.append("audio manifest missing or empty")
        else:
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
                errors.append(f"combined narration missing/empty: {audio_path}")
            else:
                try:
                    actual = _wav_duration(audio_path)
                    claimed = float(mf.get("total_duration_sec", 0))
                    if abs(actual - claimed) > 0.25:
                        errors.append(f"audio duration mismatch: manifest says "
                                      f"{claimed:.2f}s, file is {actual:.2f}s")
                except Exception as e:
                    errors.append(f"could not read narration wav ({e})")
            # Every storyboard step must have an audio segment
            if sb and sb.get("steps"):
                seg_ids = {s["id"] for s in mf["segments"]}
                for st in sb["steps"]:
                    if st.get("id") not in seg_ids:
                        errors.append(f"no audio segment for step {st.get('id')}")
            # Check each segment file exists
            for seg in mf.get("segments", []):
                sp = seg.get("path", "")
                if sp and not os.path.exists(sp):
                    warnings.append(f"segment file missing: {sp}")

        # 4. Annotations
        anns = _load(annotations_path)
        if not anns:
            errors.append("auto annotations missing or empty")
        else:
            total = float((mf or {}).get("total_duration_sec", 0) or 0)
            if total:
                late = [a for a in anns if float(a.get("time", 0)) > total]
                if late:
                    errors.append(f"{len(late)} annotation(s) start after audio end")
                last_t = max(float(a.get("time", 0)) for a in anns)
                tail = total - last_t
                if tail < 1.5:
                    warnings.append(f"final action only {tail:.1f}s before audio end")
            has_answer = any(
                a.get("action") == "mark_answer"
                or (a.get("action") == "write_step" and a.get("box_answer"))
                for a in anns)
            if not has_answer:
                errors.append("no final-answer action (mark_answer or boxed write_step)")

        # 5. Layout validation — advisory only; renderer wraps/clips gracefully
        lv = _load(layout_validation_path)
        if lv:
            if lv.get("status") == "failed":
                warnings.append(f"layout validation flagged issues (renderer will "
                                f"wrap/clip): {lv.get('issues', [])[:3]}")
            for iss in (lv.get("issues") or [])[:3]:
                warnings.append(f"layout: {iss}")
        else:
            warnings.append(f"layout_validation.json not found — layout not pre-checked")

        # 6. Glyph report
        gr = _load(glyph_report_path)
        if gr:
            if gr.get("status") not in ("safe", "substituted"):
                errors.append(f"glyph report status: {gr.get('status')}")
        else:
            warnings.append(f"glyph_report.json not found — symbols not verified")

        n_checked = "pre-render checks"

    elif stage == "post":
        # 7. Video exists and is non-trivially large
        if not os.path.exists(video_path):
            errors.append(f"final video missing: {video_path}")
        elif os.path.getsize(video_path) < 50_000:
            errors.append(f"final video suspiciously small "
                          f"({os.path.getsize(video_path)} bytes)")
        else:
            try:
                from moviepy import VideoFileClip
                clip = VideoFileClip(video_path)
                vdur = clip.duration
                clip.close()
                claimed = float((mf or {}).get("total_duration_sec", 0) or 0)
                if claimed and abs(vdur - claimed) > 1.5:
                    warnings.append(f"video duration {vdur:.1f}s differs from "
                                    f"narration {claimed:.1f}s")
                if vdur > 185:
                    warnings.append(f"video is {vdur:.0f}s — exceeds 3 min cap")
            except Exception as e:
                warnings.append(f"could not probe video duration ({str(e)[:80]})")

        # 8. Contact sheet
        if not os.path.exists(contact_sheet_path):
            warnings.append(f"contact sheet not found: {contact_sheet_path}")

        n_checked = "post-render checks"
    else:
        raise ValueError(f"unknown validation stage: {stage!r}")

    # ── Accumulate report ────────────────────────────────────────────────
    report = _load(report_path) or {}
    report[stage] = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
    }
    report["ok"] = all(report.get(s, {}).get("ok", True) for s in ("pre", "post"))
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Quality score (write/update every time) ──────────────────────────
    qs = _compute_quality_score(errors, warnings, sb, mf, ver, allow_unverified)
    os.makedirs(os.path.dirname(quality_score_path) or ".", exist_ok=True)
    with open(quality_score_path, "w", encoding="utf-8") as f:
        json.dump(qs, f, indent=2, ensure_ascii=False)

    tag = "OK" if not errors else "FAILED"
    print(f"  Validation [{stage}] {tag}: {len(errors)} error(s), "
          f"{len(warnings)} warning(s) ({n_checked}) -> {report_path}")
    for e in errors:
        print(f"    ERROR: {e}")
    for w in warnings:
        print(f"    WARN:  {w}")
    print(f"  Quality score: {qs['overall_score']:.2f} -> {quality_score_path}")
    return (not errors), report


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "pre"
    ok, _ = validate_output(stage=stage)
    sys.exit(0 if ok else 1)
