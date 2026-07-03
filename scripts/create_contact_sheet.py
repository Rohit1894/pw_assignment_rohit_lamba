#!/usr/bin/env python3
"""Extract sample frames from the final video into a contact sheet for quick
visual QA (is the board laid out right? does the answer appear?).

Frames at 0%, 15%, 30%, 50%, 70%, 90% and 100% of the runtime, tiled into
output/contact_sheet.jpg with timestamps.
"""

import os
import sys

FRACTIONS = (0.0, 0.10, 0.25, 0.40, 0.60, 0.80, 0.95, 1.0)


def create_contact_sheet(video_path, output_path="output/contact_sheet.jpg",
                         columns=4):
    """Render the contact sheet; returns output_path."""
    from moviepy import VideoFileClip
    from PIL import Image, ImageDraw

    clip = VideoFileClip(video_path)
    dur = clip.duration
    frames = []
    for frac in FRACTIONS:
        # Sample just inside the ends: t=duration can be past the last frame.
        t = min(max(frac * dur, 0.05), dur - 0.05)
        frames.append((frac, t, Image.fromarray(clip.get_frame(t))))
    clip.close()

    thumb_w = 480
    tw, th = frames[0][2].size
    thumb_h = int(th * thumb_w / tw)
    rows = (len(frames) + columns - 1) // columns
    pad = 8
    sheet = Image.new("RGB", (columns * (thumb_w + pad) + pad,
                              rows * (thumb_h + pad + 18) + pad), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    for i, (frac, t, im) in enumerate(frames):
        r, c = divmod(i, columns)
        x = pad + c * (thumb_w + pad)
        y = pad + r * (thumb_h + pad + 18)
        sheet.paste(im.resize((thumb_w, thumb_h)), (x, y))
        draw.text((x + 4, y + thumb_h + 3),
                  f"{int(frac * 100)}%  t={t:.1f}s", fill=(230, 230, 235))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sheet.save(output_path, quality=88)
    print(f"  Contact sheet ({len(frames)} frames) -> {output_path}")
    return output_path


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "output/final.mp4"
    out = sys.argv[2] if len(sys.argv) > 2 else "output/contact_sheet.jpg"
    create_contact_sheet(video, out)
