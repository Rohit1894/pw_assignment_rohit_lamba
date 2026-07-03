#!/usr/bin/env python3
"""Convert canonical_solution.json into the storyboard — the exact sequence of
(board action, Hinglish narration) steps that drives audio and video.

Each storyboard step is ONE small unit: one or a few board lines + one or two
short narration sentences. Sarvam generates audio PER STEP, so short steps are
what make audio-to-writing sync exact.

Narration comes in two forms per step:
  display_narration_roman — Roman Hinglish (logs / subtitles / debugging)
  tts_narration_text      — mixed-script Hinglish for Sarvam: Hindi words in
                            Devanagari, English/science/math terms in English
                            (fully romanised Hindi degrades Indic TTS quality).

The visual skeleton (board_lines arrays, order, final answer mark) is built
deterministically from the canonical solution — the board NEVER shows content
not in the verified solution. Gemini writes ONLY the narration around it.

Output: output/storyboard.json
"""

import json
import math
import os
import re
import sys

from gemini_utils import call_gemini_json

ALLOWED_ACTIONS = {"write_step", "write_math", "underline_existing", "circle_word",
                   "cross_out_word", "annotate_word", "fill_placeholder",
                   "draw_arrow", "match_pair", "mark_answer", "pause"}

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def _board_lines_from_step(s: dict) -> list:
    """Extract board_lines from a solution step (new array or legacy text)."""
    lines = s.get("board_lines")
    if isinstance(lines, list) and lines:
        return [str(l).strip() for l in lines if str(l).strip()]
    text = str(s.get("text") or "").strip()
    return [text] if text else []


def _build_visual_skeleton(solution: dict) -> list:
    """Deterministic board plan: one write_step per canonical solution step, then
    mark_answer. The narration Gemini writes can be different for each but it
    must NOT add or remove board steps — only narrate them."""
    steps = []
    n = 0

    # Prepend a compact "Given" step so students see what's known before solving
    given = solution.get("given") or []
    if given:
        n += 1
        given_board = ["Given:"] + [str(g)[:60] for g in given[:4]]
        steps.append({
            "id": f"s{n}",
            "visual_action": "write_step",
            "zone": "solution",
            "board_lines": given_board,
            "title": "Given Values",
            "step_id": "given",
        })

    for s in solution.get("solution_steps", []):
        n += 1
        board_lines = _board_lines_from_step(s)
        if not board_lines:
            continue
        steps.append({
            "id": f"s{n}",
            "visual_action": "write_step",
            "zone": "solution",
            "board_lines": board_lines,
            "title": s.get("title", ""),
            "step_id": s.get("id", ""),
        })
    option = solution.get("final_answer", {}).get("option", "")
    if option:
        n += 1
        ans_text = solution.get("final_answer", {}).get("text", "")
        answer_board = []
        if ans_text:
            answer_board = [f"Answer: Option ({option}) = {ans_text}"]
        steps.append({
            "id": f"s{n}",
            "visual_action": "mark_answer",
            "target": option,
            "title": "Answer",
            "board_lines": answer_board,
            "answer_text": ans_text,
        })
    return steps


def _narration_prompt(solution: dict, skeleton: list, target_duration: int) -> str:
    def _lines_repr(bl):
        return " | ".join(bl) if bl else "(mark answer)"

    step_desc = "\n".join(
        f'  {sk["id"]}: [{sk.get("title", "")}] '
        + (f'writes [{_lines_repr(sk.get("board_lines", []))}]'
           if sk["visual_action"] == "write_step"
           else f'marks option ({sk.get("target", "")}) as correct')
        for sk in skeleton
    )
    per_step = max(4, min(10, target_duration // max(len(skeleton), 1)))
    ans_option = solution.get("final_answer", {}).get("option", "")
    ans_text = solution.get("final_answer", {}).get("text", "")
    return f"""You are a friendly Indian teacher (Physics Wallah style) explaining a solution
in natural HINGLISH — conversational Hindi mixed with English technical terms.

The verified solution:
  CONCEPT: {solution.get('concept', '')}
  FORMULA: {solution.get('formula', '')}
  FINAL ANSWER: option {ans_option} — {ans_text}

The video will show these board steps IN THIS ORDER (do not add/remove/reorder):
{step_desc}

For EACH step id, write the teacher's narration for the moment that step
appears on the board. Return ONLY a JSON object:

{{
  "steps": [
    {{
      "id": "s1",
      "display_narration_roman": "<Roman Hinglish, e.g. 'Ab hum given values likhte hain. Yahan initial velocity 20 m/s di hui hai.'>",
      "tts_narration_text": "<SAME sentence mixed-script: Hindi words in Devanagari, English/math terms in Latin, e.g. 'अब हम given values लिखते हैं। यहाँ initial velocity 20 m/s दी हुई है।'>",
      "duration_hint_sec": <int, expected speaking time>
    }},
    ...
  ]
}}

Narration rules (IMPORTANT):
- Natural teacher Hinglish: "dekhiye", "ab hum", "toh", "yahan par", warm tone.
- TWO to THREE sentences per step (~{per_step} seconds spoken).
- Both fields say the SAME thing — one romanised, one mixed-script.
- In tts_narration_text keep ALL technical/English terms (velocity, formula,
  option, numbers, units like m/s) in English/Latin; ONLY genuine Hindi words go
  in Devanagari. Numbers stay as digits.
- Read formulas naturally: "v square equal to u square plus 2 a s".
- COMPLETE every sentence — NEVER end a narration mid-formula, mid-calculation, or
  mid-word. If you write "CR = " you MUST finish with the value.
- First step (Given Values): say what the question asks, name all given values clearly.
- When an abbreviation first appears (LC, ZE, OR, CR, etc.) expand it in the narration.
- Second step onward: explain WHY each formula or step is used, not just what.
- Last (answer) step: say "isliye correct answer option {ans_option} hai, yaani {ans_text}".
- Total speaking time should be about {target_duration} seconds across all steps.
- Do NOT use "as an AI", "the model will now", or any robotic phrasing.
Return the raw JSON object only."""


def _fallback_narration(step: dict, solution: dict):
    """Template narration when Gemini is unavailable."""
    title = (step.get("title") or "").lower()
    lines = step.get("board_lines") or []
    text_preview = " ".join(lines)[:60]
    if step["visual_action"] == "mark_answer":
        opt = step.get("target", "")
        ans = solution.get("final_answer", {}).get("text", "")
        return (f"Therefore, correct answer option {opt} hai. {ans}.",
                f"Therefore, correct answer option {opt} है। {ans}।")
    if "given" in title:
        return ("Ab hum given values likhte hain. " + text_preview + ".",
                "अब हम given values लिखते हैं। " + text_preview + "।")
    if "formula" in title:
        return ("Ab hum formula likhte hain. " + text_preview + ".",
                "अब हम formula लिखते हैं। " + text_preview + "।")
    if "substitu" in title:
        return ("Ab values substitute karte hain. " + text_preview + ".",
                "अब values substitute करते हैं। " + text_preview + "।")
    if "answer" in title:
        return ("Toh final answer aata hai " + text_preview + ".",
                "तो final answer आता है " + text_preview + "।")
    return ("Ab hum likhte hain: " + text_preview + ".",
            "अब हम लिखते हैं: " + text_preview + "।")


def _estimate_duration(tts_text: str) -> int:
    """Rough speaking-time estimate for Hinglish (~13 chars/sec at pace 0.92)."""
    return max(3, min(15, round(len(tts_text) / 13)))


def _is_truncated_narration(tts: str) -> bool:
    """True when the narration is obviously cut off mid-sentence."""
    s = tts.strip()
    if len(s) < 20:
        return True
    # Ends with an operator/equals before punctuation — sentence not completed
    if re.search(r'[=+\-×÷/]\s*[।.।,]?\s*$', s):
        return True
    return False


# ── Board-fit pagination ──────────────────────────────────────────────────
# A page is paginated by how many board lines physically FIT the solution zone
# at a legible font — NOT by narration seconds. Splitting by time let a single
# page accumulate far more lines than the board could hold, which overflowed
# into the overflow column (that collides with the primary column in top/bottom
# layouts) and produced the heavy text overlap. The renderer clears the board
# between pages (page-turn wipe), so each page starts on a clean zone.
# Keep these constants in step with render/schedule.py + layout_engine.py.
_LINE_SPACING = 1.45
_LEGIBLE_FONT_PX = 24      # comfortable floor on 720p (matches layout_engine MIN_FONT_PX)
_CANVAS_H = 720
_ZONE_BOTTOM_PAD = 12      # render/schedule.py _BOTTOM_PAD


def _page_line_budget(layout: dict) -> int:
    """How many board lines fit one page's primary column at a legible font.

    Mirrors the renderer's usable height: it writes from the solution-zone top
    down to (canvas height - bottom pad) — a little past the nominal zone bottom
    — in a single column before it would spill into the overflow column.
    """
    z = (layout or {}).get("solution_zone")
    y1, y2 = (z[1], z[3]) if z and len(z) == 4 else (360, 680)
    usable_h = (_CANVAS_H - _ZONE_BOTTOM_PAD) - (int(y1) + 10)
    line_h = math.ceil(_LEGIBLE_FONT_PX * _LINE_SPACING)
    return max(3, int(usable_h // line_h))


def _step_line_weight(step: dict) -> int:
    """Approximate rendered line count for a step's board content. A stacked
    fraction renders taller than a plain line, so it counts for more."""
    lines = step.get("board_lines") or []
    if not lines:
        return 0                       # mark_answer / no-write steps take ~no column height
    weight = 0.0
    for ln in lines:
        s = str(ln)
        weight += 2.0 if ("\\frac" in s or "\\dfrac" in s) else 1.0
    return max(1, math.ceil(weight))


def _assign_pages(steps: list, layout: dict = None) -> list:
    """Assign page numbers by BOARD FIT, not narration time.

    Each page must hold its steps' board lines in the solution zone at a legible
    font; when the next step would overflow, a new page begins (the renderer
    clears the board between pages). Steps are atomic — a derivation step is
    never split across a page break — and mark_answer rides the last page with
    the conclusion it marks.
    """
    budget = _page_line_budget(layout)
    page = 1
    used = 0
    for st in steps:
        w = _step_line_weight(st)
        is_answer = st.get("visual_action") == "mark_answer"
        # Start a fresh page when this step's lines won't fit the current one.
        # (A zero-weight mark_answer never forces a break — it stays on the
        # current page. A lone step bigger than a whole page still gets its own
        # page, best-effort, and the renderer shrinks it to fit.)
        if not is_answer and used > 0 and used + w > budget:
            page += 1
            used = 0
        st["page"] = page
        used += w
    return steps


def generate_storyboard(solution: dict, layout: dict = None,
                        language: str = "hinglish",
                        target_duration: int = 75,
                        output_path: str = "output/storyboard.json") -> dict:
    """Build and save the storyboard. Returns the storyboard dict."""
    skeleton = _build_visual_skeleton(solution)
    if not skeleton:
        raise RuntimeError("Storyboard: canonical solution produced no steps")

    # ── Fetch narration from Gemini ──────────────────────────────────────
    narration = {}
    try:
        data = call_gemini_json(
            [_narration_prompt(solution, skeleton, target_duration)],
            temperature=0.6, label="Storyboard narration")
        for st in (data.get("steps") or []) if isinstance(data, dict) else []:
            if not isinstance(st, dict):
                continue
            roman = str(st.get("display_narration_roman") or "").strip()
            tts = str(st.get("tts_narration_text") or "").strip()
            if roman and tts and not _is_truncated_narration(tts):
                narration[str(st.get("id"))] = (roman, tts, st.get("duration_hint_sec"))
            elif roman and tts:
                print(f"  note: step {st.get('id')} narration appears truncated; using fallback")
    except Exception as e:
        print(f"  Gemini narration failed ({str(e)[:100]}); using template narration")

    # ── Build storyboard steps ───────────────────────────────────────────
    steps = []
    for sk in skeleton:
        roman, tts, hint = None, None, None
        if sk["id"] in narration:
            roman, tts, hint = narration[sk["id"]]
            if not _DEVANAGARI_RE.search(tts):
                print(f"  note: {sk['id']} tts_narration_text has no Devanagari")
        if not roman or not tts:
            roman, tts = _fallback_narration(sk, solution)
        try:
            hint = int(hint) if hint else _estimate_duration(tts)
        except (TypeError, ValueError):
            hint = _estimate_duration(tts)
        hint = max(3, min(15, hint))

        step = {
            "id": sk["id"],
            "page": 1,                       # assigned below
            "visual_action": sk["visual_action"],
            "zone": sk.get("zone", "solution"),
            "board_lines": sk.get("board_lines", []),
            "display_narration_roman": roman,
            "tts_narration_text": tts,
            "duration_hint_sec": hint,
            "title": sk.get("title", ""),
        }
        if sk["visual_action"] == "mark_answer":
            step["target"] = sk.get("target", "")
        steps.append(step)

    # ── Assign pages ─────────────────────────────────────────────────────
    steps = _assign_pages(steps, layout)

    # ── Validate ─────────────────────────────────────────────────────────
    for st in steps:
        if st["visual_action"] not in ALLOWED_ACTIONS:
            raise RuntimeError(f"Storyboard: invalid visual_action "
                               f"'{st['visual_action']}' in {st['id']}")
        if not (st.get("tts_narration_text") or "").strip():
            raise RuntimeError(f"Storyboard: step {st['id']} has no narration")
        if st["visual_action"] == "write_step" and not st.get("board_lines"):
            print(f"  WARN: step {st['id']} has no board_lines (write_step)")

    storyboard = {
        "target_duration_sec": target_duration,
        "style": {
            "board": "white",
            "writing": "clean-handwritten",
            "board_language": "english",
            "voice_language": "hinglish",
            "question_position": "auto",
            "solution_position": "auto",
        },
        "steps": steps,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(storyboard, f, indent=2, ensure_ascii=False)
    est = sum(s["duration_hint_sec"] for s in steps)
    n_pages = max((s["page"] for s in steps), default=1)
    print(f"  Storyboard: {len(steps)} steps, {n_pages} page(s), "
          f"~{est}s estimated narration -> {output_path}")
    return storyboard


if __name__ == "__main__":
    with open(sys.argv[1] if len(sys.argv) > 1
              else "output/canonical_solution.json", encoding="utf-8") as f:
        sol = json.load(f)
    generate_storyboard(sol)
