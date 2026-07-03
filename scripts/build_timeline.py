#!/usr/bin/env python3
"""Build the final timed annotation list from storyboard + audio manifest.

TIMING PRINCIPLE: the measured Sarvam segment durations ARE the timeline.
Each storyboard step's annotation fires at its own segment's start (plus a
small pen-up delay); the renderer stretches the writing across the segment.
Whisper (if a transcript of the generated narration is provided) only REFINES
a step's start within a small window — it is never the primary timing source,
and it can never reorder steps.

Annotations carry "exact": true so the renderer's pacing heuristics (built for
fuzzy Gemini timelines) leave these audio-derived timings untouched.

Output: output/auto_annotations.json
"""

import json
import os
import sys

PEN_UP_DELAY = 0.2       # writing starts a beat after the teacher starts the sentence
ANSWER_MIN_TAIL = 2.0    # final answer must be visible at least this long before the end
REFINE_WINDOW = 1.0      # Whisper may move a step start by at most ± this many seconds


def _whisper_refine(start, transcript_segments):
    """Snap `start` to the nearest Whisper segment boundary within
    REFINE_WINDOW. Conservative: manifest timing wins when Whisper has no
    boundary nearby (Whisper on TTS audio is only a cross-check)."""
    best = None
    for seg in transcript_segments:
        s = seg.get("start")
        if s is None:
            continue
        d = abs(float(s) - start)
        if d <= REFINE_WINDOW and (best is None or d < abs(best - start)):
            best = float(s)
    return best if best is not None else start


def build_timeline(storyboard_path, audio_manifest_path,
                   transcript_path=None,
                   output_annotations_path="output/auto_annotations.json",
                   option_positions=None):
    """Combine storyboard + manifest (+ optional Whisper transcript) into the
    renderer's annotation schema. Returns the annotation list."""
    with open(storyboard_path, encoding="utf-8") as f:
        storyboard = json.load(f)
    with open(audio_manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    seg_by_id = {s["id"]: s for s in manifest.get("segments", [])}
    total = float(manifest.get("total_duration_sec", 0.0))
    if not seg_by_id or total <= 0:
        raise RuntimeError("build_timeline: audio manifest has no segments")

    transcript_segments = []
    if transcript_path and os.path.exists(transcript_path):
        try:
            with open(transcript_path, encoding="utf-8") as f:
                transcript_segments = (json.load(f) or {}).get("segments") or []
            print(f"  Whisper refinement: {len(transcript_segments)} transcript "
                  f"segments available")
        except Exception as e:
            print(f"  Whisper refinement skipped ({str(e)[:80]})")

    annotations = []
    steps = storyboard.get("steps", [])
    for st in steps:
        sid = st.get("id")
        seg = seg_by_id.get(sid)
        if seg is None:
            raise RuntimeError(f"build_timeline: no audio segment for step {sid!r}")
        start = float(seg["start_time"]) + PEN_UP_DELAY
        if transcript_segments:
            start = max(float(seg["start_time"]),
                        _whisper_refine(start, transcript_segments))
        action = st.get("visual_action", "write_step")

        if action == "pause":
            continue  # a pause is just narration with nothing new on the board

        ann = {
            "time": round(start, 2),
            "action": action,
            "step_id": sid,
            "exact": True,
            "narration": st.get("display_narration_roman", ""),
            # Carry the storyboard page on EVERY annotation (not just write_steps).
            # The renderer wipes the board between pages; an annotation missing its
            # page would default to page 1 and — if it is a workspace write (e.g.
            # the mark_answer→write_step fallback when the option can't be located)
            # — get wiped the moment a later page begins. Setting it here keeps a
            # last-page answer on the last page, so it is never wiped.
            "page": st.get("page", 1),
        }
        if action == "mark_answer":
            letter = str(st.get("target", "")).strip().upper()[:1]
            known = bool(option_positions) and letter in (option_positions or {})
            if known:
                ann["target"] = letter
            else:
                # Option marker not found on the canvas — write the answer in
                # the solution area instead of ringing a spot that isn't there.
                ann["action"] = "write_step"
                answer_val = st.get("answer_text", "")
                if answer_val:
                    ann["text"] = f"Answer: Option ({letter}) = {answer_val}"
                    ann["board_lines"] = [f"Answer: Option ({letter}) = {answer_val}"]
                else:
                    ann["text"] = f"Correct option: {letter}" if letter else "Answer"
                    ann["board_lines"] = [ann["text"]]
                ann["zone"] = "solution"
                ann["box_answer"] = True
                print(f"  mark_answer fallback: option '{letter}' not located on "
                      f"canvas; writing it in the solution area")
        elif action in ("underline_existing", "circle_word", "cross_out_word"):
            ann["target"] = st.get("target", st.get("text", ""))
        elif action == "match_pair":
            ann["from_target"] = st.get("from_target", "")
            ann["to_target"] = st.get("to_target", "")
        elif action == "draw_arrow":
            ann["start_target"] = st.get("start_target", "")
            ann["end_target"] = st.get("end_target", "")
        elif action == "fill_placeholder":
            ann["label"] = st.get("target", "")
            ann["text"] = st.get("text", "")
        else:  # write_step / annotate_word and friends
            board_lines = st.get("board_lines") or []
            if board_lines:
                # Include both board_lines (spec schema) and joined text
                # (renderer compatibility — schedule.py uses ann["text"])
                ann["board_lines"] = board_lines
                ann["text"] = "\n".join(str(l) for l in board_lines)
            else:
                ann["text"] = str(st.get("text") or "")
            ann["zone"] = st.get("zone", "solution")
            if st.get("target"):
                ann["target"] = st["target"]
        annotations.append(ann)

    if not annotations:
        raise RuntimeError("build_timeline: storyboard produced no annotations")

    # ── Box/highlight the final answer line ──────────────────────────────
    # The last write_step before (or instead of) the answer mark is the final
    # answer statement — box it so the conclusion visibly pops.
    write_steps = [a for a in annotations if a["action"] == "write_step"]
    if write_steps:
        write_steps[-1]["box_answer"] = True

    # ── Timing invariants ────────────────────────────────────────────────
    # 1. No action may start after the audio ends.
    # 2. The final action must leave >= ANSWER_MIN_TAIL of visible tail.
    # 3. Times strictly increase (Whisper refinement must not reorder).
    last_allowed = max(0.5, total - ANSWER_MIN_TAIL)
    for a in annotations:
        a["time"] = min(a["time"], last_allowed)
    for i in range(1, len(annotations)):
        if annotations[i]["time"] <= annotations[i - 1]["time"]:
            annotations[i]["time"] = round(annotations[i - 1]["time"] + 0.5, 2)
    if annotations[-1]["time"] > last_allowed:
        annotations[-1]["time"] = last_allowed

    os.makedirs(os.path.dirname(output_annotations_path) or ".", exist_ok=True)
    with open(output_annotations_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    print(f"  Timeline: {len(annotations)} annotations over {total:.1f}s "
          f"-> {output_annotations_path}")
    return annotations


if __name__ == "__main__":
    sb = sys.argv[1] if len(sys.argv) > 1 else "output/storyboard.json"
    mf = sys.argv[2] if len(sys.argv) > 2 else "output/audio_manifest.json"
    tr = sys.argv[3] if len(sys.argv) > 3 else None
    build_timeline(sb, mf, tr)
