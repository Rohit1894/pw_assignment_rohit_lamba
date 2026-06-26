#!/usr/bin/env python3
"""Thin facade for the render pipeline.

The renderer was split into the scripts/render/ package (Steps 1-5 of the
render_video refactor); render_video() and the frame renderer now live in
render.frame. This module re-exports render_video() so existing imports
(`from render_video import render_video`, used by main.py / run_new_question.py)
and the CLI (`python render_video.py img ann aud out`) keep working unchanged."""

import sys

from render.frame import render_video, get_substring_bounds  # noqa: F401  (re-exported)


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "input/question.png"
    ann = sys.argv[2] if len(sys.argv) > 2 else "output/annotations.json"
    aud = sys.argv[3] if len(sys.argv) > 3 else "input/narration.mp3"
    out = sys.argv[4] if len(sys.argv) > 4 else "output/final.mp4"
    render_video(img, ann, aud, out)
