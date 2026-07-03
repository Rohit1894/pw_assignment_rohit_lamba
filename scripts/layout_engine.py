#!/usr/bin/env python3
"""Layout engine: measure board content and validate / repair before rendering.

Before any pixel is drawn the engine:
1. Measures every board line in the storyboard using PIL's text metrics.
2. Estimates total vertical space needed in the solution zone.
3. If content overflows the zone, tries repairs in order:
      a. Wrap long lines
      b. Reduce font size (down to MIN_FONT_PX)
      c. Split into pages (writes layout_plan.json)
4. Writes output/layout_validation.json with final verdict.

The renderer in render/schedule.py already handles overflow via _fit_steps;
this engine gives an EARLY WARNING before TTS audio is generated so the
storyboard can be trimmed if needed, and produces layout_plan.json so
multi-page rendering knows what goes where.

Outputs:
    output/layout_validation.json
    output/layout_plan.json (only for multi-page layouts)
"""

import json
import math
import os
import sys

# Geometry constants must match prepare_canvas.py / frame.py defaults.
CANVAS_W, CANVAS_H = 1280, 720
DEFAULT_SOLUTION_ZONE = (610, 70, 1200, 650)   # x1, y1, x2, y2
MIN_FONT_PX = 24        # below this text is unreadable on 720p
DEFAULT_FONT_PX = 30    # starting font size
LINE_SPACING_FACTOR = 1.45
SECTION_PAD_PX = 10     # extra pixels between solution sections
MAX_CHARS_PER_LINE = 55  # wrap board lines longer than this


def _sol_zone(layout: dict) -> tuple:
    z = (layout or {}).get("solution_zone")
    if z and len(z) == 4:
        return tuple(z)
    return DEFAULT_SOLUTION_ZONE


def _measure_text_height(lines: list, font_size: int) -> int:
    """Estimate total pixel height for a list of text lines at given font_size."""
    line_h = math.ceil(font_size * LINE_SPACING_FACTOR)
    return len(lines) * line_h


def _wrap_lines(lines: list, max_chars: int = MAX_CHARS_PER_LINE) -> list:
    """Wrap lines that exceed max_chars at a word boundary."""
    result = []
    for line in lines:
        if len(line) <= max_chars:
            result.append(line)
            continue
        words = line.split()
        current = ""
        for w in words:
            if current and len(current) + 1 + len(w) > max_chars:
                result.append(current)
                current = w
            else:
                current = (current + " " + w).strip()
        if current:
            result.append(current)
    return result


def _collect_board_lines(storyboard: dict) -> list:
    """Collect all board_lines from storyboard steps (preserving order)."""
    all_lines = []
    for step in storyboard.get("steps", []):
        lines = step.get("board_lines") or []
        if not lines and step.get("text"):
            lines = [step["text"]]
        all_lines.extend([str(l) for l in lines if l])
    return all_lines


def _group_by_page(storyboard: dict) -> dict:
    """Group storyboard steps by their page number."""
    pages = {}
    for step in storyboard.get("steps", []):
        p = step.get("page", 1)
        pages.setdefault(p, []).append(step)
    return pages


def plan_and_validate(storyboard: dict, layout: dict,
                      validation_out: str = "output/layout_validation.json",
                      plan_out: str = "output/layout_plan.json") -> dict:
    """Measure the storyboard content, validate fit, and write layout artefacts.

    Returns the validation dict (always succeeds — at worst it splits pages).
    """
    x1, y1, x2, y2 = _sol_zone(layout)
    zone_w = x2 - x1
    zone_h = y2 - y1
    issues = []

    # 1. Collect all board lines
    all_lines = _collect_board_lines(storyboard)
    if not all_lines:
        issues.append("No board_lines found in storyboard")

    # 2. Try default font size
    font_size = DEFAULT_FONT_PX
    wrapped = _wrap_lines(all_lines)

    # Check line widths (rough: assume avg character ~0.6 * font_size wide).
    # Wide lines are a WARNING only — the renderer wraps them automatically.
    char_w = font_size * 0.6
    wide_lines = [line for line in wrapped if len(line) * char_w > zone_w]
    if wide_lines:
        for line in wide_lines[:5]:
            issues.append(f"Line too wide for solution zone: {line[:50]!r}")
        # Attempt to re-wrap with a tighter limit before reporting failure
        wrapped = _wrap_lines(all_lines, max_chars=max(20, int(zone_w / max(1, char_w)) - 2))
        total_h = _measure_text_height(wrapped, font_size)

    total_h = _measure_text_height(wrapped, font_size)
    lines_per_page = max(1, zone_h // math.ceil(font_size * LINE_SPACING_FACTOR))

    # 3. Try wrapping repair
    if total_h > zone_h:
        wrapped = _wrap_lines(all_lines, max_chars=MAX_CHARS_PER_LINE - 10)
        total_h = _measure_text_height(wrapped, font_size)

    # 4. Try reducing font size
    while total_h > zone_h and font_size > MIN_FONT_PX:
        font_size -= 2
        total_h = _measure_text_height(wrapped, font_size)
        lines_per_page = max(1, zone_h // math.ceil(font_size * LINE_SPACING_FACTOR))

    # 5. If still overflowing → multi-page
    pages_needed = 1
    repair_action = None
    if total_h > zone_h:
        issues.append(f"Content overflows solution zone even at {font_size}px font. "
                      f"Switching to multi-page layout.")
        pages_needed = math.ceil(len(wrapped) / max(1, lines_per_page))
        repair_action = "split_to_multi_page"

    # 6. Build per-page plan when multi-page
    plan = None
    if pages_needed > 1 or repair_action:
        step_groups = _group_by_page(storyboard)
        if len(step_groups) <= 1:
            # Auto-assign pages: solution steps split evenly across pages
            page_plans = []
            steps = storyboard.get("steps", [])
            per_page = max(1, math.ceil(len(steps) / pages_needed))
            for i in range(pages_needed):
                chunk = steps[i * per_page:(i + 1) * per_page]
                titles = [s.get("title", s.get("id", "")) for s in chunk]
                page_plans.append({"page": i + 1, "contains": titles})
        else:
            page_plans = [{"page": p, "contains": [s.get("id") for s in ss]}
                          for p, ss in sorted(step_groups.items())]

        plan = {
            "layout_type": "multi_page",
            "reason": (f"Solution requires ~{len(wrapped)} rendered lines but "
                       f"one page can fit ~{lines_per_page} at {font_size}px."),
            "pages": page_plans,
        }
        os.makedirs(os.path.dirname(plan_out) or ".", exist_ok=True)
        with open(plan_out, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        print(f"  Layout plan: {pages_needed} pages -> {plan_out}")

    # "Line too wide" issues are advisory — the renderer wraps automatically.
    # Only mark truly failed if height overflows AND no repair resolved it.
    hard_issues = [i for i in issues if "overflows" in i.lower()]
    status = "failed" if (hard_issues and not repair_action) else "passed"

    validation = {
        "status": status,
        "pages": pages_needed,
        "font_size": font_size,
        "total_lines": len(wrapped),
        "lines_per_page": lines_per_page,
        "solution_zone": [x1, y1, x2, y2],
        "issues": issues,
        "repair_action": repair_action,
    }

    os.makedirs(os.path.dirname(validation_out) or ".", exist_ok=True)
    with open(validation_out, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2, ensure_ascii=False)
    tag = "PASSED" if status == "passed" else "FAILED"
    print(f"  Layout [{tag}]: {len(wrapped)} lines, {font_size}px font, "
          f"{pages_needed} page(s) -> {validation_out}")
    if issues:
        for iss in issues[:5]:
            print(f"    {iss}")

    return validation


if __name__ == "__main__":
    sb_path = sys.argv[1] if len(sys.argv) > 1 else "output/storyboard.json"
    ly_path = sys.argv[2] if len(sys.argv) > 2 else "output/layout.json"
    with open(sb_path, encoding="utf-8") as f:
        sb = json.load(f)
    layout_data = {}
    try:
        with open(ly_path, encoding="utf-8") as f:
            layout_data = json.load(f)
    except Exception:
        pass
    plan_and_validate(sb, layout_data)
