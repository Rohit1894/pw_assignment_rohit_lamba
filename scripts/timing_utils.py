#!/usr/bin/env python3
"""
Shared timeline hygiene for annotation actions, applied no matter which engine
produced them (Gemini multimodal, Whisper + Gemini text, or the rule-based
fallback). Guarantees every action is ordered, minimally spaced, spread across
the lecture (not bunched at the start), and finishes inside the audio.
"""


def normalize_timeline(annotations, duration_hint=None, spacing=1.2, tail=3.0):
    """
    Clean an annotation list in place-safe fashion and return it.

    Steps:
      1. Drop malformed entries; coerce `time` to a non-negative float; sort.
      2. Anti front-loading: if the actions cover far less of the audio than they
         should, linearly stretch the timeline to fill it. This preserves the
         INTENDED order and relative pacing while using the whole duration, and
         is a no-op when actions are already spread out.
      3. Forward pass: enforce a minimum gap between consecutive actions.
      4. Backward pass: pull any tail that ran past the audio (minus a finishing
         `tail`) back inside it, preserving order — so the final answer/summary
         still appear on screen and nothing is scheduled after the audio ends.
    """
    annotations = [a for a in annotations if isinstance(a, dict) and "action" in a]
    for a in annotations:
        a["time"] = max(0.0, float(a.get("time", 0.0)))
    annotations.sort(key=lambda x: x["time"])

    if duration_hint and len(annotations) >= 3:
        first = annotations[0]["time"]
        last = annotations[-1]["time"]
        actual_span = last - first
        target_span = max(1.0, duration_hint - tail - first)
        if 1e-3 < actual_span < 0.6 * target_span:
            factor = target_span / actual_span
            for a in annotations:
                a["time"] = first + (a["time"] - first) * factor
            print(f"  Timeline was front-loaded ({actual_span:.0f}s of "
                  f"{duration_hint:.0f}s); stretched x{factor:.1f} to fill the audio")

    for i in range(1, len(annotations)):
        if annotations[i]["time"] - annotations[i - 1]["time"] < spacing:
            annotations[i]["time"] = annotations[i - 1]["time"] + spacing

    if duration_hint:
        max_end = max(1.0, duration_hint - tail)
        for i in range(len(annotations) - 1, -1, -1):
            cap = max_end if i == len(annotations) - 1 else annotations[i + 1]["time"] - spacing
            if annotations[i]["time"] > cap:
                annotations[i]["time"] = max(0.0, cap)

    return annotations
