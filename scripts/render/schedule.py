"""Animation schedule: turn the action list into timed, positioned draw steps —
note pruning, slot/layout assignment, option-strike geometry, diagram compute.
The orchestration brain. Extracted from render_video.py (Step 4 refactor)."""

import math
import random
import re

from PIL import Image, ImageDraw

from render.constants import (
    PEN_COLOR, PEN_WIDTH, ANSWER_INK, MATCH_INK,
    WRITE_ACTIONS, NOTE_ACTIONS, ANSWER_ACTIONS, VERDICT_ACTIONS, TEXT_ACTIONS,
    _SUPERSCRIPT_MAP, _SUBSCRIPT_MAP,
)
from render.text_utils import (
    _contains_devanagari, _sanitize_text, _is_formula_like_text,
    split_grapheme_clusters, wrap_text_to_width, split_into_math_tokens,
)
from render.fonts import (
    _find_hindi_font, _find_font, _font_has_glyph, _glyph_fallback_fonts,
    _resolve_glyph_font, _sized_sub_font, _sized_variant, _note_font_size,
)
from render.strokes import (
    _draw_handwritten_line, _draw_progressive_polyline, _draw_progressive_underline,
    _draw_progressive_circle, _draw_progressive_arrow, _draw_progressive_diagonal_slash,
    _draw_progressive_ellipse, _draw_progressive_cross,
)
from render.text_render import (
    draw_custom_text, get_custom_text_width, draw_math_equation_with_radicals,
    _measure_block, _layout_text_lines, _render_text_layer, _build_text_layers,
    _paste_text_reveal, _frac_nesting_depth,
)
from render.geometry import (
    _boxes_overlap, _segment_hits_rect, _arrow_crosses_text, _underline_for_box,
    _next_clear_y, _find_slot, _snap_box_to_ocr, _resolve_box, _coords_box,
)
from render.verdicts import _resolve_verdict_box, _verdict_row_positions
from render.matching import (
    _target_cue, _route_match_pairs, _is_matching_timeline,
    _matching_table_bounds, _option_box_or_infer,
)
from render.placeholders import (
    _label_index, _fit_linear, _snap_to_column,
    infer_missing_placeholders, _placeholder_box,
)
from render.diagram import _layout_diagram, _render_diagram


def _is_workspace_write_action(ann):
    """Treat generated formula notes as worked-solution lines."""
    action = ann.get("action")
    return action in WRITE_ACTIONS or (
        action == "write_note" and _is_formula_like_text(ann.get("text", ""))
    )


_VERDICT_WORDS = ("गलत", "सही", "सत्य", "असत्य", "true", "false",
                  "correct", "incorrect", "right", "wrong")


_STMT_LABELS = ("कथन", "अभिकथन", "कारण", "statement", "reason", "assertion")


def _prune_notes(annotations):
    """Defensive cleanup that mirrors the prompt's note rules, so even a stale or
    over-eager Gemini response renders clean:

      1. Drop write_notes that merely restate a verdict mark (the ✓/✗ already says
         "कथन I: गलत") — keep the final conclusion note that selects the option.
      2. Cap free notes (write_note + annotate_word) at 4; keep the conclusion plus
         the earliest informative notes, in original order.
    """
    def norm(s):
        return re.sub(r"\s+", " ", str(s or "")).strip().lower()

    def note_words(s):
        return [w for w in re.findall(r"[^\s,/():;=+\-]+", norm(s)) if len(w) >= 3]

    def near_duplicate(a, b):
        aw, bw = note_words(a.get("text")), note_words(b.get("text"))
        if not aw or not bw:
            return False
        aset, bset = set(aw), set(bw)
        overlap = len(aset & bset) / max(1, min(len(aset), len(bset)))
        an, bn = " ".join(aw), " ".join(bw)
        return overlap >= 0.82 or an in bn or bn in an

    verdict_targets = [norm(a.get("target")) for a in annotations
                       if a.get("action") == "verdict_mark"]

    def is_conclusion(note):
        txt = note.get("text", "")
        nt = norm(txt)
        if "→" in txt or "->" in txt:
            return True
        if sum(nt.count(s) for s in _STMT_LABELS) >= 2:
            return True
        return len(nt.split()) > 5

    def is_redundant_verdict(note):
        if note.get("action") != "write_note":
            return False
        nt = norm(note.get("text"))
        if not any(v in nt for v in _VERDICT_WORDS):
            return False
        if is_conclusion(note):
            return False
        # short single-statement judgement → already shown as a ✓/✗ mark
        if any(vt and vt in nt for vt in verdict_targets):
            return True
        return sum(nt.count(s) for s in _STMT_LABELS) == 1 and len(nt.split()) <= 4

    # Drop a note that merely RESTATES the diagram. The prompt forbids duplicating the
    # figure, but Gemini sometimes writes the same fact as both a node and a note → the
    # board says it twice. Only a STRONG overlap is pruned (>=2/3 of the note's content
    # words already appear as diagram labels), so notes that add a NEW fact survive.
    # Hindi inflects heavily, so words match by a shared 4+ char prefix, not equality.
    diagram_words = set()
    for a in annotations:
        if a.get("action") == "draw_diagram":
            dia = a.get("diagram") or {}
            labels = [dia.get("title", "")]
            labels += [n.get("label", "") for n in (dia.get("nodes") or [])]
            labels += [e.get("label", "") for e in (dia.get("edges") or [])]
            for lab in labels:
                diagram_words.update(w for w in re.findall(r"[^\s,/()]+", norm(lab))
                                     if len(w) >= 4)

    def _answer_conclusion(note):           # the option-selecting note — never prune it
        txt = note.get("text", "")
        return ("→" in txt or "->" in txt
                or sum(norm(txt).count(s) for s in _STMT_LABELS) >= 2)

    def duplicates_diagram(note):
        if not diagram_words or _answer_conclusion(note):
            return False
        words = [w for w in re.findall(r"[^\s,/()]+", norm(note.get("text")))
                 if len(w) >= 4]
        # Only prune a full sentence-like FACT (>=4 content words) the diagram already
        # states. A short note (a 2-word label/translation like the Sertoli-cell name,
        # or a fragment) earns its place even if its words appear in the figure — generic
        # domain words (कोशिका/शुक्राणु) recur across nodes and would otherwise false-match.
        if len(words) < 4:
            return False
        hits = sum(1 for w in words
                   if any(w[:4] == d[:4] and (w.startswith(d) or d.startswith(w))
                          for d in diagram_words))
        return hits / len(words) >= 0.67

    notes = [a for a in annotations if a.get("action") in ("write_note", "annotate_word")]
    survivors = [n for n in notes
                 if not is_redundant_verdict(n) and not duplicates_diagram(n)]

    deduped = []
    for n in survivors:
        if any(near_duplicate(n, old) for old in deduped):
            continue
        deduped.append(n)
    survivors = deduped

    concl = [n for n in survivors if is_conclusion(n)][:2]
    reg = [n for n in survivors if n not in concl]
    keep_ids = {id(n) for n in concl + reg[:max(0, 4 - len(concl))]}

    pruned = [a for a in annotations
              if a.get("action") not in ("write_note", "annotate_word") or id(a) in keep_ids]

    # Matching tables are a special case: the printed table is the primary visual.
    # Extra explainers beside the table compete with the pair strokes and can make a
    # correct mapping feel uncertain. Keep the table clean: underlines, match_pair
    # connectors, and the final answer mark are enough.
    if _is_matching_timeline(pruned):
        cleaned = []
        matching_dropped = 0
        for a in pruned:
            act = a.get("action")
            if act in ("draw_diagram", "annotate_word", "write_note"):
                matching_dropped += 1
                continue
            cleaned.append(a)
        pruned = cleaned
        if matching_dropped:
            print(f"  Matching cleanup dropped {matching_dropped} table note/duplicate diagram action(s)")
    dropped = len(annotations) - len(pruned)
    if dropped:
        print(f"  Pruned {dropped} redundant/excess note(s) (verdict-duplicate or >4 cap)")
    return pruned


# Effective rendered height of a derivation line, in "plain-line" units. A \frac
# stacks 3-6x taller than a text line depending on nesting depth and how many
# fraction tokens it holds. ONE shared definition so the step-font auto-fit and the
# overflow prune below agree on exactly how tall the worked solution is.
_FRAC_WEIGHTS = {0: 1.0, 1: 3.0, 2: 5.0}


def _step_weight(text):
    st = _sanitize_text(text or "")
    if "\\frac" in st:
        depth = min(2, _frac_nesting_depth(st))
        nf = st.count("\\frac")
        return _FRAC_WEIGHTS[depth] * max(1.0, nf / 2.0)
    return 1.0


def _step_lhs(text):
    """Normalised left-hand side (everything before the first '='), used to group a
    derivation that re-states the same quantity line after line."""
    st = _sanitize_text(text or "")
    lhs = st.split("=", 1)[0]
    return re.sub(r"\s+", "", lhs).lower()


def _dedup_steps(annotations):
    """Drop a write_step that EXACTLY repeats the previous derivation line (a
    generation glitch — the same line emitted twice). Always safe: progressive
    derivations differ line to line, so only byte-identical repeats are removed.
    Anything resembling a real next step is left for the overflow-gated _fit_steps,
    which only trims when the board physically can't hold the lines."""
    drop = set()
    last = None
    for i, a in enumerate(annotations):
        if a.get("action") != "write_step":
            continue
        cur = re.sub(r"\s+", "", _sanitize_text(a.get("text") or ""))
        if cur and cur == last:
            drop.add(i)
        last = cur
    if not drop:
        return annotations
    print(f"  Dropped {len(drop)} duplicate derivation step(s)")
    return [a for k, a in enumerate(annotations) if k not in drop]


def _fit_steps(annotations, avail_h, floor_fs=10, line_factor=1.4, pad_per_line=12):
    """Guarantee the derivation FITS the board — the hard backstop that the old
    'clamp the last line into frame with a few px overlap' fallback relied on.

    GATED ON REAL OVERFLOW: if the full set already stacks inside the two workspace
    columns at the floor font, it is returned UNCHANGED (so derivations that ship
    fine — v2/v4 etc. — are byte-identical). Only when the lines genuinely can't fit
    do we drop, in this priority:
      1. Middle lines of a re-stated run (same LHS on >=3 lines): Gemini plugging
         numbers in one at a time — the redundant substitutions, dropped from the
         middle outward. The symbolic first line and final result are protected.
      2. If still overflowing, any other non-protected line, heaviest first.
    Protected throughout: the first and last step, and the first (symbolic) line of
    every re-stated quantity — so a real progressive simplification is never gutted
    unless space leaves no choice, and even then its formula + answer survive."""
    idx = [i for i, a in enumerate(annotations) if a.get("action") == "write_step"]
    if len(idx) < 3 or avail_h <= 0:
        return annotations
    per = floor_fs * line_factor + pad_per_line
    weight = {i: _step_weight(annotations[i].get("text")) for i in idx}

    def height(drop):
        return sum(weight[i] for i in idx if i not in drop) * per

    if height(set()) <= avail_h:
        return annotations                      # fits at the floor font — leave it

    # Group the steps into same-LHS runs (a quantity re-stated line after line).
    runs, prev = [], object()
    for i in idx:
        lhs = _step_lhs(annotations[i].get("text"))
        if lhs != prev:
            runs.append([])
            prev = lhs
        runs[-1].append(i)

    protected = {idx[0], idx[-1]}
    for members in runs:
        protected.add(members[0])               # symbolic formula of each quantity

    tier1 = []                                  # redundant substitution middles
    for members in runs:
        if len(members) >= 3:
            mid = (members[0] + members[-1]) / 2.0
            tier1 += sorted(members[1:-1], key=lambda i: abs(i - mid))
    tier2 = [i for i in idx if i not in protected and i not in tier1]
    tier2.sort(key=lambda i: weight[i], reverse=True)

    drop = set()
    for i in tier1 + tier2:
        if height(drop) <= avail_h:
            break
        drop.add(i)
    if not drop:
        return annotations
    print(f"  Trimmed {len(drop)} intermediate derivation step(s) to fit the board")
    return [a for k, a in enumerate(annotations) if k not in drop]


# ── Animation schedule ──────────────────────────────────────────────────────
def _build_schedule(annotations, total_duration, enriched_ocr, option_positions,
                    fonts=None, image_size=(1280, 720), pen=PEN_COLOR):
    """
    Pre-compute geometry, layout and timing for every annotation.

    New behaviours vs. the original single-column writer:
      - `annotate_word`/`write_note` are placed in scattered empty space (never
        overlapping printed text), with dynamic font size and optional arrows.
      - underlines / answer marks are solid lines anchored to OCR'd text.
    Legacy `write_equation`/`write_text` keep the simple column layout so the
    English math flow is unchanged.
    """
    schedule = []
    W, H = image_size

    annotations = _prune_notes(annotations)
    annotations = _dedup_steps(annotations)

    # Storyboard timelines (auto-audio mode) carry "exact": true — their times
    # come from measured per-step audio segment durations, so the pacing
    # heuristics below (phase gate, blank-hold re-spacing, elimination
    # re-ordering — all built for fuzzy Gemini timelines) must not move them.
    exact_mode = bool(annotations) and all(
        a.get("exact") for a in annotations if isinstance(a, dict))

    font_body = fonts[0] if fonts else _find_font("body", 28)
    hindi_font = fonts[1] if fonts and len(fonts) > 1 else _find_hindi_font(30)
    _measure_img = Image.new("RGB", (10, 10))
    _measure_draw = ImageDraw.Draw(_measure_img)

    ocr_index = enriched_ocr.get("index") if enriched_ocr else None
    free_spaces = enriched_ocr.get("free_spaces", []) if enriched_ocr else []
    placeholders = enriched_ocr.get("placeholders", {}) if enriched_ocr else {}
    # Fill in any blank the timeline references that neither OCR nor Gemini
    # located, by extrapolating from the regular layout of the detected blanks.
    inferred_ph = infer_missing_placeholders(placeholders, annotations, W, H)
    if inferred_ph:
        desc = ", ".join(f"{k}{'*' if v[1] else '?'}" for k, v in sorted(inferred_ph.items()))
        print(f"  Geometrically inferred placeholder(s): {desc}  (* = confident)")
    rng = random.Random(20240623)  # seeded → reproducible but scattered layout

    # Occupancy map: every printed OCR box is an obstacle, plus the PW logo band.
    occupied = []
    for el in (enriched_ocr.get("elements") or []):
        b = el.get("bounds")
        if b:
            occupied.append(tuple(b))
    occupied.append((W - 175, 0, W, 150))  # top-right watermark logo
    protected_table = _matching_table_bounds(
        annotations, enriched_ocr.get("elements") or [], option_positions, W, H)
    if protected_table:
        occupied.append(protected_table)

    # Soft block over the flowchart/diagram node cluster so scattered NOTES don't
    # land on the figure. (fill_placeholder still writes right next to its blank.)
    opt_top = min((min(p[1] for p in v) for v in option_positions.values()), default=H)

    # Statements/options divider for verdict marks. Deriving it from option markers
    # is fragile: if EasyOCR misreads one marker (e.g. "(A)" → garbage), opt_top drops
    # INTO the option block and a "कथन I" inside an option would be mistaken for the
    # statement heading. The instruction line ("...विकल्पों में से सही उत्तर चुनिए" /
    # "...कूट का प्रयोग कर...") reliably sits between the statements and the options —
    # use its top as the ceiling so a verdict only ever lands on a statement heading.
    _INSTR_KEYS = ("विकल्प", "चुनिए", "कूट", "निम्नलिखित में से सही", "सही उत्तर")
    instr_tops = [b[1] for el in (enriched_ocr.get("elements") or [])
                  for b in [el.get("bounds")]
                  if b and b[1] < opt_top and any(k in (el.get("text") or "") for k in _INSTR_KEYS)]
    stmt_ceiling = min(instr_tops) if instr_tops else opt_top

    dxs, dys = [], []
    for el in (enriched_ocr.get("elements") or []):
        b = el.get("bounds")
        txt = (el.get("text") or "").strip()
        if b and 120 < (b[1] + b[3]) / 2 < opt_top - 10 and len(txt) < 25 and not re.search(r"\d{4}", txt):
            dxs += [b[0], b[2]]
            dys += [b[1], b[3]]
    diagram_block = (min(dxs) - 6, min(dys) - 6, max(dxs) + 6, max(dys) + 6) if dxs else None
    note_block = [b for b in (diagram_block, protected_table) if b]

    # Workspace zone: the clear right-margin column to the RIGHT of the printed
    # question text, where free factual notes stack tidily (legible, teacher-like)
    # instead of scattering over the slide. Derived from where the text ends.
    text_rights = [b[2] for b in occupied if b[2] < W - 50]   # exclude watermark band
    text_right = max(text_rights) if text_rights else int(0.5 * W)
    ws_zone_x0 = int(max(0.5 * W, min(W - 280, text_right + 24)))

    # Legacy single-column cursor (used only by write_equation/write_text).
    if free_spaces:
        rx1, ry1, rx2, ry2 = free_spaces[0]["bounds"]
        print(f"  Writing layout region selected: {free_spaces[0]['position']} bounds: {free_spaces[0]['bounds']}")
    else:
        rx1, ry1, rx2, ry2 = 680, 100, W - 40, H - 40
    wx = rx1 + 25
    wy = max(ry1 + 30, 150)
    region_w = max(200, (rx2 - rx1) - 50)

    # Whiteboard-storyboard mode: the caller supplies the solution zone
    # explicitly (layout.json) instead of deriving the workspace from OCR.
    # All step writing then stays inside that zone. Absent the key, nothing
    # changes for the classic image+audio mode.
    forced_zone = (enriched_ocr or {}).get("workspace_zone")
    if forced_zone and len(forced_zone) == 4:
        rx1, ry1, rx2, ry2 = (int(v) for v in forced_zone)
        wx = rx1 + 10
        wy = ry1 + 10
        region_w = max(200, (rx2 - rx1) - 30)
        ws_zone_x0 = rx1
        print(f"  Workspace forced to solution zone: {forced_zone}")
    fallback_y = max(200, int(H * 0.55))  # bottom fallback when no slot fits
    ph_legend = {"x": None, "y": None}    # running cursor for the fill-in legend

    # Page-turn support: snapshot the primary-column cursor so that when a new
    # storyboard page begins the writing restarts at the TOP of the solution zone
    # (in the wide primary column) instead of continuing to stack every page into
    # one ever-growing column. That endless stacking is what overflowed into the
    # bottom-left overflow column and produced two derivations superimposed.
    _page_wx0, _page_wy0, _page_region_w0 = wx, wy, region_w

    # Per-statement ✓/✗ column for statement-evaluation / "how many are correct"
    # questions. Reconstructed ONCE from OCR geometry + Gemini box_2d order so that
    # every statement reliably gets its mark (a garbled enumerator must never drop
    # one) and the marks line up in a clean right-margin column.
    verdict_rows = _verdict_row_positions(
        annotations, enriched_ocr.get("elements") or [], stmt_ceiling, W, ws_zone_x0)

    # ── Wrong-option elimination geometry ────────────────────────────────────
    # Each rejected option gets ONE clean strikethrough across its whole row,
    # located by its (A)/(B)/(C)/(D) marker. This replaces ambiguous per-word
    # cross-outs that pile onto the topmost option (a word like "शुक्रवाहक"
    # appears in several options; resolving it grabs the wrong row).
    _opt_elems = [el.get("bounds") for el in (enriched_ocr.get("elements") or [])
                  if el.get("bounds")]
    opt_rows = {}
    for _L, _bbox in option_positions.items():
        _xs = [p[0] for p in _bbox]
        _ys = [p[1] for p in _bbox]
        opt_rows[_L] = (min(_xs), (min(_ys) + max(_ys)) / 2, min(_ys), max(_ys))

    def _option_row_right(cy, x_left):
        rights = [b[2] for b in _opt_elems
                  if abs((b[1] + b[3]) / 2 - cy) < 16 and b[2] > x_left]
        return max(rights) if rights else x_left + 320

    # A long MCQ option wraps onto 2+ lines; a single line-1 strike leaves the rest
    # uncrossed and reads as a half-done elimination. Compute ONE strike segment per
    # text line of an option so the whole option gets struck — each segment hugging
    # that line's own text (so the strike never overshoots into the blank margin).
    _marker_tops = sorted(opt_rows[L][2] for L in opt_rows)

    def _option_bottom(L):
        top = opt_rows[L][2]
        below = [mt for mt in _marker_tops if mt > top + 4]
        if below:
            return below[0]                      # next option's marker top bounds this one
        gaps = [b - a for a, b in zip(_marker_tops, _marker_tops[1:])]
        return top + (max(gaps) if gaps else 60)  # last option: one typical option height

    def _option_strike_segments(L):
        # Geometry-driven (NOT OCR line-clustering): EasyOCR routinely emits a tall
        # garbled box that spans both wrapped lines, which chains them into one cluster
        # and collapses the strike back to a single line. Instead, derive the line
        # COUNT from the option's vertical extent ÷ its marker height, place a strike at
        # each line, and hug that line's own text width via _option_row_right.
        top, bottom = opt_rows[L][2], _option_bottom(L)
        x_left, marker_cy = opt_rows[L][0], opt_rows[L][1]
        max_strike_x = min(ws_zone_x0 - 18, x_left + 430, 0.56 * W)
        mb_h = opt_rows[L][3] - top                       # marker box height ≈ line height
        line_h = mb_h if 18 <= mb_h <= 40 else 28
        n_lines = max(1, min(4, round((bottom - top) / line_h)))
        segs = []
        for i in range(n_lines):
            cy = marker_cy + i * line_h
            rights = [b[2] for b in _opt_elems
                      if abs((b[1] + b[3]) / 2 - cy) < 14 and b[2] > x_left
                      and b[0] < 0.62 * W]
            if not rights:               # no text on this line → don't strike empty space
                continue
            segs.append((x_left - 4, cy, min(max(rights) + 6, max_strike_x)))
        return segs or [(x_left - 4, marker_cy,
                         min(_option_row_right(marker_cy, x_left) + 6, max_strike_x))]

    answer_opt = None
    for _a in annotations:
        if _a.get("action") in ANSWER_ACTIONS:
            _t = re.sub(r"[^A-Da-d]", "",
                        str(_a.get("target") or _a.get("option") or "")).upper()
            if len(_t) == 1:
                answer_opt = _t
    struck_options = set()

    # ── Overflow column geometry (computed up-front; the font-fit below needs its
    # height). Bottom-left of frame, below the option block — used only when the
    # right-margin column can't hold the whole derivation legibly on its own.
    # `_ws_col` tracks which column is active. The top sits below the lowest
    # left-side OCR box (covers the option-text extent, not just markers).
    _BOTTOM_PAD = 12
    _left_occ_bottoms = [b[3] for b in occupied if b[0] < ws_zone_x0 + 20]
    if _left_occ_bottoms:
        _ovf_y0 = int(max(_left_occ_bottoms)) + 20
    elif option_positions:
        _opt_all_pts = [p for pts in option_positions.values() for p in pts]
        _ovf_y0 = int(max(p[1] for p in _opt_all_pts)) + 20
    else:
        _ovf_y0 = int(0.65 * H)
    _ovf_y0 = min(_ovf_y0, H - 80)
    _ovf_x0 = 24
    _ovf_x1 = max(ws_zone_x0 - 20, 200)
    _ovf_region_w = max(100, _ovf_x1 - _ovf_x0)
    _ws_col = 0   # 0 = primary right column; 1 = overflow bottom-left column

    # Worked-solution workspace: size the step font/spacing so ALL derivation
    # lines fit the space that ACTUALLY exists. Preferred outcome is a single
    # right-margin column (one tidy top-to-bottom reading flow); only when the
    # steps can't fit there at a legible size do we size to the right + bottom-left
    # columns COMBINED, which keeps the final steps from clipping off the bottom.
    right_avail_h = max(80, (H - _BOTTOM_PAD) - wy)
    ovf_avail_h = max(0, (H - _BOTTOM_PAD) - _ovf_y0)
    # Hard fit backstop: drop the least-essential intermediate steps if the
    # derivation can't physically stack inside the two columns at the floor font.
    # Runs BEFORE the font-fit so the size estimate and placement see the fitted set.
    # Non-step actions (annotate_word, non-formula write_note) also consume right-column
    # height — subtract their footprint so _fit_steps sees the true available space and
    # prunes substitution-expansion steps before the last-resort clamp triggers.
    # Items that consume workspace column space but are NOT write_steps and
    # therefore not prunable by _fit_steps:
    #   • annotate_word — placed in the right column via the note path
    #   • write_note (formula-like) — converted to write_step at render time
    #     but _fit_steps only counts action=="write_step", so it misses them
    #   • write_note (non-formula) — may land in the workspace via _find_slot
    # _fit_steps_overhead: subtract ALL of these so _fit_steps sees the true
    #   available space for the prunable write_step actions.
    # _non_step_items: only those NOT already counted by _is_workspace_write_action
    #   (formula-like write_notes are already in effective_lines; annotate_word
    #   and non-formula write_notes are not).
    _fit_steps_overhead = sum(
        1 for a in annotations
        if a.get("action") in ("annotate_word", "write_note")
    )
    _non_step_items = sum(
        1 for a in annotations
        if a.get("action") == "annotate_word"
        or (a.get("action") == "write_note"
            and not _is_formula_like_text(a.get("text", "")))
    )
    _fit_steps_avail = (right_avail_h + ovf_avail_h) * 0.90 - _fit_steps_overhead * 30
    annotations = _fit_steps(annotations, _fit_steps_avail, pad_per_line=16)
    n_steps = sum(1 for a in annotations if _is_workspace_write_action(a))
    if n_steps > 0:
        # Weight each step by its expected rendered height. A \frac renders
        # 3–6× taller than a plain text line depending on nesting depth and how
        # many fraction tokens appear. Counting effective weighted lines gives a
        # much better font-size estimate than treating everything as one line.
        _FRAC_WEIGHTS = {0: 1.0, 1: 3.0, 2: 5.0}
        effective_lines = 0.0
        for _a in annotations:
            if _is_workspace_write_action(_a):
                _bl = _a.get("board_lines") or []
                if _bl:
                    effective_lines += float(len(_bl))
                else:
                    _st = _sanitize_text(_a.get("text", ""))
                    if "\\frac" in _st:
                        _depth = min(2, _frac_nesting_depth(_st))
                        _nf = _st.count("\\frac")
                        effective_lines += _FRAC_WEIGHTS[_depth] * max(1.0, _nf / 2.0)
                    else:
                        effective_lines += 1.0
        effective_lines += _non_step_items
        # Right column alone first; only fall back to right + overflow combined
        # when the single column would force the font below a legible size.
        _MIN_COMFORT_FS = 14
        _fs_right = (right_avail_h / max(effective_lines, 1.0) - 12) / 1.4
        if _fs_right >= _MIN_COMFORT_FS:
            step_fs = int(max(_MIN_COMFORT_FS, min(int(font_body.size), _fs_right)))
        else:
            # Many steps for the space → size to the right + overflow columns
            # combined, with ~10% packing headroom (real \frac descenders + per-line
            # gaps run a little taller than the weighted estimate). A lower floor
            # (10) is accepted here because a complete-but-small derivation beats one
            # whose final steps clip off the bottom edge.
            _avail = (right_avail_h + ovf_avail_h) * 0.90
            _per_line = _avail / max(effective_lines, 1.0)
            step_fs = int(max(10, min(int(font_body.size), (_per_line - 12) / 1.4)))
    else:
        step_fs = int(font_body.size)
    step_font = _sized_variant(font_body, step_fs)
    # Horizontal fit: shrink so the widest Latin/math step stays inside the column.
    widest = 0.0
    for a in annotations:
        if _is_workspace_write_action(a):
            _lines = a.get("board_lines") or [a.get("text", "")]
            for _ln in _lines:
                st = _sanitize_text(str(_ln))
                if st and not _contains_devanagari(st):
                    try:
                        widest = max(widest, _measure_draw.textlength(st, font=step_font))
                    except Exception:
                        pass
    if widest > region_w > 0:
        step_fs = int(max(13, step_fs * region_w / widest))
        step_font = _sized_variant(font_body, step_fs)
    step_hindi_font = _sized_variant(hindi_font, step_fs + 2)
    step_gap = max(6, int(step_fs * 0.45))

    # (Overflow column geometry _ovf_x0/_ovf_x1/_ovf_y0/_ws_col is computed
    # above, before the font-fit, because the auto-fit needs the overflow height.)

    def _compute_diagram(spec, occ):
        """Lay a diagram spec into the larger clear region; return its layout dict."""
        text_bottom = max((b[3] for b in occ if b[3] < H - 20 and b[0] < W - 50),
                          default=int(0.5 * H))
        bottom_x2 = ws_zone_x0 - 24 if (n_steps > 0 and ws_zone_x0 - 24 > 340) else W - 24
        bottom_region = (24, int(text_bottom) + 18, bottom_x2, H - 24)
        right_top = 150
        col_notes = [b[3] for b in occ if b[2] > ws_zone_x0 and b[1] < 0.5 * H]
        if col_notes:
            right_top = max(right_top, int(max(col_notes)) + 18)
        right_region = (ws_zone_x0, right_top, W - 18, H - 24)
        area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])
        hint = (spec.get("layout") or "").lower()
        if hint in ("vertical", "axis"):
            region = right_region if area(right_region) >= 0.5 * area(bottom_region) else bottom_region
        else:
            bottom_h = bottom_region[3] - bottom_region[1]
            region = bottom_region if bottom_h >= 150 else right_region
        return _layout_diagram(spec, region, (font_body, hindi_font), _measure_draw)

    # Pre-reserve each diagram's footprint BEFORE the main loop places any notes, so a
    # note timed earlier than the diagram is never parked where the figure will be
    # drawn (which produced notes sitting on top of the diagram). The big figure wins
    # its space first; notes then flow into whatever is left.
    diagram_layouts = {}
    diagram_node_boxes = {}    # node-id -> box, so a fill_placeholder that targets a
    for i, ann in enumerate(annotations):  # diagram node lands AT the node, not a legend
        if ann.get("action") == "draw_diagram" and ann.get("diagram"):
            dg = _compute_diagram(ann["diagram"], occupied)
            if dg:
                diagram_layouts[i] = dg
                occupied.append(dg["bbox"])
                for nid, box in (dg.get("node_boxes") or {}).items():
                    diagram_node_boxes[str(nid)] = box

    # Pre-route all matching connectors together: lane assignment needs to see
    # every pair at once so they fan across the gutter instead of overlapping.
    match_routes = _route_match_pairs(annotations, ocr_index, option_positions, W, H)

    # Base obstacle map for a page wipe: the printed question OCR text, the
    # watermark band, any protected table and pre-reserved diagrams — everything
    # that is NOT prior-page worked-solution writing. On a page turn `occupied` is
    # reset to this so the fresh page avoids the question but reuses the whole
    # (now-clear) solution zone.
    _page_base_occupied = list(occupied)
    _prev_page = None

    temp_schedule = []
    for i, ann in enumerate(annotations):
        action = ann["action"]
        t = ann["time"]

        # ── Page turn → wipe the solution workspace ───────────────────────────
        # Storyboard pages are meant to CLEAR the board between them (the printed
        # question persists; the worked solution is erased). Pages only ever
        # increase; a lower value (e.g. a page-less mark_answer defaulting to 1
        # after a page-2 step) is not a real turn, so only reset on an increase.
        _pg = ann.get("page", 1)
        if _prev_page is not None and _pg > _prev_page:
            wx, wy, region_w = _page_wx0, _page_wy0, _page_region_w0
            _ws_col = 0
            occupied[:] = _page_base_occupied
        _prev_page = max(_prev_page or 0, _pg)
        entry = {**ann, "write_start": t, "write_end": t + 0.8}
        if action == "write_note" and _is_formula_like_text(ann.get("text", "")):
            action = "write_step"
            entry["action"] = action

        if action in ("circle_word", "circle_existing"):
            entry["write_end"] = t + 1.0
            box = _resolve_box(ann, ocr_index, W, H, option_positions=option_positions)
            entry["ellipse_params"] = None
            if box:
                x1, y1, x2, y2 = box
                # Tight, vertically-centred ring that HUGS the exact word/phrase
                # (small even padding) instead of a loose oval that cuts through it.
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                rx = max(14, (x2 - x1) / 2 + 8)
                ry = max(11, (y2 - y1) / 2 + 7)
                entry["ellipse_params"] = (cx, cy, rx, ry)
                occupied.append((cx - rx, cy - ry, cx + rx, cy + ry))

        elif action == "underline_existing":
            entry["write_end"] = t + 1.0
            box = _resolve_box(ann, ocr_index, W, H, option_positions=option_positions)
            # A thin underline is intentionally NOT added as an obstacle, so a
            # word's meaning can still be written just below it.
            ul = _underline_for_box(box, occupied, W, H) if box else None
            # Reject underlines that resolve to the options block or below.
            # "underline_existing" targets belong to the question stem; when OCR
            # resolves the target to a wrong box in the options zone the line
            # appears in blank space. Use opt_top (first option marker row) as
            # the hard ceiling. Without option data fall back to H*0.92.
            if ul:
                if option_positions:
                    reject_y = opt_top - 4
                else:
                    content_boxes = [b for b in occupied if b != (W - 175, 0, W, 150)]
                    reject_y = (min(max(b[3] for b in content_boxes) + 10, H * 0.97)
                                if content_boxes else H * 0.92)
                if ul[1] > reject_y:
                    ul = None
            entry["underline_params"] = ul

        elif action == "cross_out_word":
            entry["write_end"] = t + 0.8
            tgt = str(ann.get("target", "") or "")
            letter = re.sub(r"[^A-Da-d]", "", tgt).upper()
            is_marker = len(letter) == 1 and len(tgt.strip()) <= 4 and len(option_positions) >= 2
            opt_L = None
            box = None
            if is_marker:
                opt_L = letter                       # explicit marker target "(A)"
            else:
                box = _resolve_box(ann, ocr_index, W, H, option_positions=option_positions)
                if box:                              # word target → which option row?
                    bcy = (box[1] + box[3]) / 2
                    for L, (xl, cy, y1, y2) in opt_rows.items():
                        if y1 - 6 <= bcy <= y2 + 6:
                            opt_L = L
                            break
            if opt_L is not None:
                # One strike per wrong option; never strike the correct answer
                # (an ambiguous word may resolve into the wrong/right row).
                if opt_L != answer_opt and opt_L not in struck_options:
                    if opt_L in opt_rows:
                        xl, cy = opt_rows[opt_L][0], opt_rows[opt_L][1]
                    else:                            # marker OCR-missed → extrapolate its row
                        ib = _option_box_or_infer(opt_L, option_positions)
                        xl, cy = (ib[0], (ib[1] + ib[3]) / 2) if ib else (None, None)
                    if xl is not None:
                        struck_options.add(opt_L)   # track for answer-ordering; no visual strike
                # else: suppress (redundant, or would cross the answer)
            else:
                entry["cross_params"] = box          # genuine stem-word cross → an X

        elif action in NOTE_ACTIONS:  # annotate_word, write_note
            text = _sanitize_text(ann.get("text", ""))
            target_str = ann.get("target")
            custom_box = ann.get("box") or ann.get("box_2d") or ann.get("box_norm")
            anchor = None
            skip_hug = False  # True when the anchor is inside question text

            # Phase A: resolve anchor early so self-labeling can use it.
            if not custom_box and target_str:
                anchor = _resolve_box(ann, ocr_index, W, H, option_positions=option_positions)
            custom_cand = None
            if custom_box:
                custom_cand = _resolve_box(ann, ocr_index, W, H, option_positions=option_positions)
                if custom_cand and action == "annotate_word":
                    # annotate_word corrections must NEVER land on top of the question
                    # text (the custom_box / box_2d Gemini gives points INTO the text).
                    # Use it only as an arrow anchor; the note always goes to workspace.
                    if target_str and not anchor:
                        anchor = custom_cand
                    skip_hug = True  # skip "hug" → go straight to workspace

            # Self-label a lone annotate_word so it reads without an arrow:
            # prefix the target cue ("s -> b") so the correction is self-contained.
            # Anchor must be set first (done above) for this check to fire.
            if (action == "annotate_word" and anchor and target_str
                    and "->" not in text and "→" not in text and len(text) <= 44):
                cue = _target_cue(target_str)
                if cue and cue not in text:
                    # ASCII arrow — the Devanagari font has no "→" glyph (renders tofu).
                    text = f"{cue} -> {text}"

            is_hi = _contains_devanagari(text)
            base = 25 if action == "annotate_word" else 28
            fnt = _sized_variant(hindi_font if is_hi else font_body,
                                 _note_font_size(text, base))
            max_w = 260 if action == "annotate_word" else 340
            layout = _build_text_layers(text, fnt, pen, max_w, _measure_draw)
            bw, bh = layout["block_w"], layout["block_h"]

            # Phase B: slot placement from custom_box (needs bw/bh from layout).
            slot = None
            need_arrow = False
            if custom_cand and action != "annotate_word":
                # write_note: honour the custom_box if it doesn't overlap content.
                note_rect = (custom_cand[0], custom_cand[1],
                             custom_cand[0] + bw, custom_cand[1] + bh)
                if not _boxes_overlap(note_rect, occupied, pad=4):
                    slot = note_rect
                if slot is None and target_str and not anchor:
                    anchor = _resolve_box(ann, ocr_index, W, H,
                                          option_positions=option_positions)
                    skip_hug = True
            if slot is None:
                # 1) annotate_word: hug the target word directly when there's room.
                #    Skipped when the anchor is inside the question text body.
                if anchor and not skip_hug:
                    for (bx, by) in ((anchor[0], anchor[3] + 6), (anchor[0] - 12, anchor[3] + 8),
                                     (anchor[2] - bw, anchor[3] + 6)):
                        bx = max(8, min(int(bx), W - bw - 8)); by = int(by)
                        cand = (bx, by, bx + bw, by + bh)
                        if by + bh < H and not _boxes_overlap(cand, occupied + note_block, pad=3):
                            slot = cand; break
                # 2) tidy right-margin workspace column.
                if slot is None:
                    slot, _ = _find_slot(W, H, max(bw, 1), bh, occupied + note_block, rng,
                                         top_band=158, x_lo=ws_zone_x0, tidy=True)
                    if slot is not None and anchor:
                        need_arrow = True
                # 3) tidy LEFT columns (clear lower-left) — neat top-down stacks in empty
                #    space instead of a random scatter over the watermark/centre.
                if slot is None:
                    slot, _ = _find_slot(W, H, max(bw, 1), bh, occupied + note_block, rng,
                                         top_band=158, x_lo=16, tidy=True)
                    if slot is not None and anchor:
                        need_arrow = True
            if slot is None:
                slot = (40, fallback_y, 40 + bw, fallback_y + bh)
                fallback_y = min(H - bh - 10, fallback_y + bh + 14)
                need_arrow = (anchor is not None)
            occupied.append(slot)
            entry["write_pos"] = (slot[0], slot[1])
            entry["text_layout"] = layout
            # Subtle white backing card so the note stays legible over the slide's
            # faint watermark/background (matches the clean reference look).
            entry["note_card"] = (slot[0] - 8, slot[1] - 4, slot[0] + bw + 8, slot[1] + bh + 6)

            ap = ann.get("arrow_params")
            if ap and len(ap) == 4:
                entry["arrow_params"] = tuple(ap)
            elif need_arrow and anchor:
                # annotate_word already self-labels ("s -> b") so no arrow is
                # needed when the anchor is inside question text — a long diagonal
                # connector cutting across the slide looks worse than the absence
                # of the link.
                if action == "annotate_word" and skip_hug:
                    pass
                else:
                    # Connect the note to its word when the connector won't slice
                    # across other text (self-labeled, so an unlinked note still reads).
                    acx, acy = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
                    scx, scy = slot[0] + 8, slot[1] + 6
                    a_from, s_to = (acx, anchor[3] + 2), (scx, scy)
                    if (math.hypot(acx - scx, acy - scy) <= 300
                            and not _arrow_crosses_text(a_from, s_to, occupied)):
                        entry["arrow_params"] = (acx, anchor[3] + 2, scx, scy)

        elif action == "fill_placeholder":  # answer for a printed-figure blank
            # CASE 1 — the fill targets a NODE of a drawn diagram (Gemini sometimes
            # "fills" a sequence/flowchart it just drew, label = node id like "n3").
            # Place the text BESIDE that node, synced to when it is spoken, instead of
            # a scattered "N = ..." legend disconnected from the figure.
            raw_label = str(ann.get("label") or ann.get("target") or "").strip()
            nbox = diagram_node_boxes.get(raw_label)
            if nbox is not None:
                value = _sanitize_text(ann.get("text", ""))
                is_hi = _contains_devanagari(value)
                fnt = _sized_variant(hindi_font if is_hi else font_body,
                                     _note_font_size(value, 22))
                layout = _build_text_layers(value, fnt, pen, 230, _measure_draw)
                bw, bh = layout["block_w"], layout["block_h"]
                ncy = (nbox[1] + nbox[3]) / 2
                prefer_left = (nbox[0] + nbox[2]) / 2 > 0.5 * W
                cands = []
                left = (int(nbox[0] - bw - 14), int(ncy - bh / 2))
                right = (int(nbox[2] + 14), int(ncy - bh / 2))
                below = (int(nbox[0]), int(nbox[3] + 4))
                cands = ([left, right] if prefer_left else [right, left]) + [below]
                lx = ly = None
                for cx, cy in cands:
                    cslot = (cx, cy, cx + bw, cy + bh)
                    if cx >= 6 and cx + bw <= W - 6 and not _boxes_overlap(cslot, occupied, pad=2):
                        lx, ly = cx, cy; break
                if lx is None:
                    lx, ly = below
                slot = (lx, ly, lx + bw, ly + bh)
                occupied.append(slot)
                entry["write_pos"] = (lx, ly)
                entry["text_layout"] = layout
                entry["note_card"] = (lx - 6, ly - 3, lx + bw + 6, ly + bh + 5)
                # short connector from the node edge to the milestone label
                if lx + bw <= nbox[0]:                       # label sits to the LEFT
                    entry["arrow_params"] = (nbox[0], ncy, lx + bw + 2, ly + bh / 2)
                elif lx >= nbox[2]:                          # label sits to the RIGHT
                    entry["arrow_params"] = (nbox[2], ncy, lx - 2, ly + bh / 2)
                temp_schedule.append(entry)
                continue
            # CASE 2 — a real printed-figure blank (A)(B)(C)(D).
            # The blanks are usually TINY cells in a dense printed figure — long Hindi
            # terms can't fit inside them, and scattering the answers around the figure
            # with loose arrows reads as a mess. So write them as a tidy LABELLED LEGEND
            # ("A = FSH") stacked in clear space, exactly like the teacher does. Each
            # entry still appears at the moment it is spoken (its own `time`).
            value = _sanitize_text(ann.get("text", ""))
            label = re.sub(r"[^A-Za-z0-9]", "",
                           str(ann.get("label") or ann.get("target") or "")).upper()[:1]
            text = f"{label} = {value}" if label else value
            is_hi = _contains_devanagari(text)
            fnt = _sized_variant(hindi_font if is_hi else font_body,
                                 _note_font_size(text, 27))
            layout = _build_text_layers(text, fnt, pen, 320, _measure_draw)
            bw, bh = layout["block_w"], layout["block_h"]

            if ph_legend["x"] is None:        # anchor the legend column in clear space
                ph_legend["x"] = int(max(ws_zone_x0, 0.52 * W))
                ph_legend["y"] = 172
            lx, ly = ph_legend["x"], ph_legend["y"]
            slot = (lx, ly, lx + bw, ly + bh)
            if ly + bh > H - 20 or _boxes_overlap(slot, occupied, pad=3):
                fb, _ = _find_slot(W, H, max(bw, 1), bh, occupied, rng,
                                   top_band=158, x_lo=ph_legend["x"], tidy=True)
                if fb is None:
                    fb, _ = _find_slot(W, H, max(bw, 1), bh, occupied, rng, top_band=158)
                if fb is not None:
                    lx, ly, slot = fb[0], fb[1], fb
            ph_legend["y"] = ly + bh + 12
            occupied.append(slot)
            entry["write_pos"] = (lx, ly)
            entry["text_layout"] = layout
            entry["note_card"] = (lx - 8, ly - 4, lx + bw + 8, ly + bh + 6)
            # Thin connector to the blank ONLY when its position is OCR-verified
            # (not Gemini's shifted box_2d, which would point into empty space).
            if label and placeholders and label in placeholders:
                pb = placeholders[label]
                pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                if not _arrow_crosses_text((pcx, pcy), (lx + 6, ly + bh / 2), occupied):
                    entry["arrow_params"] = (pcx, pcy, lx + 6, ly + bh / 2)

        elif action in WRITE_ACTIONS:  # stacked workspace column (derivation / Hindi)
            raw_lines = ann.get("board_lines") or []
            if raw_lines:
                # Sanitize each board_line SEPARATELY so that the join's \n is never
                # misread as a LaTeX control char by _restore_backslash_latex.
                clean_lines = [_sanitize_text(str(l)) for l in raw_lines if str(l).strip()]
                text = "\n".join(clean_lines)
            else:
                text = _sanitize_text(ann.get("text", ""))
            entry["text"] = text
            entry["is_hindi"] = _contains_devanagari(text)
            if entry["is_hindi"]:
                layout = _build_text_layers(text, step_hindi_font, pen, region_w, _measure_draw)
                entry["text_layout"] = layout
                bw, bh = layout["block_w"], layout["block_h"]
                clear_y, step_box = _next_clear_y(wx, wy, bw, bh, occupied, H, step_gap)
                entry["write_pos"] = (wx, clear_y)
                occupied.append(step_box)
                wy = clear_y + bh + step_gap
            elif raw_lines:
                # Multi-line board content → always use _build_text_layers (handles \n)
                layout = _build_text_layers(text, step_font, pen, region_w, _measure_draw)
                entry["text_layout"] = layout
                bw, bh = layout["block_w"], layout["block_h"]
                clear_y, step_box = _next_clear_y(wx, wy, bw, bh, occupied, H, step_gap)
                if clear_y + bh > H - _BOTTOM_PAD and _ws_col == 0:
                    _ws_col = 1
                    wx, wy = _ovf_x0, _ovf_y0
                    region_w = _ovf_region_w
                    layout = _build_text_layers(text, step_font, pen, region_w, _measure_draw)
                    entry["text_layout"] = layout
                    bw, bh = layout["block_w"], layout["block_h"]
                    clear_y, step_box = _next_clear_y(wx, wy, bw, bh, occupied, H, step_gap)
                if clear_y + bh > H - _BOTTOM_PAD:
                    clear_y = max(0, H - _BOTTOM_PAD - bh)
                    step_box = (step_box[0], clear_y, step_box[2], clear_y + bh)
                entry["render_font"] = step_font
                entry["line_height"] = int(step_fs * 1.45)
                entry["write_pos"] = (wx, clear_y)
                occupied.append(step_box)
                wy = clear_y + bh + step_gap
            else:
                entry["render_font"] = step_font
                entry["line_height"] = int(step_fs * 1.45)
                if "\\frac" in text:
                    layout = _build_text_layers(text, step_font, pen, region_w, _measure_draw)
                    entry["text_layout"] = layout
                    bw, bh = layout["block_w"], layout["block_h"]
                else:
                    try:
                        _mw = int(_measure_draw.textlength(text, font=step_font))
                    except Exception:
                        _mw = region_w + 1
                    if _mw > region_w * 0.88:
                        layout = _build_text_layers(text, step_font, pen, region_w, _measure_draw)
                        entry["text_layout"] = layout
                        bw, bh = layout["block_w"], layout["block_h"]
                    else:
                        bw = min(region_w, _mw)
                        bh = entry["line_height"]
                clear_y, step_box = _next_clear_y(wx, wy, bw, bh, occupied, H, step_gap)
                if clear_y + bh > H - _BOTTOM_PAD and _ws_col == 0:
                    _ws_col = 1
                    wx, wy = _ovf_x0, _ovf_y0
                    region_w = _ovf_region_w
                    try:
                        _mw2 = int(_measure_draw.textlength(text, font=step_font))
                    except Exception:
                        _mw2 = region_w + 1
                    if "\\frac" in text or _mw2 > region_w * 0.88:
                        layout = _build_text_layers(text, step_font, pen, region_w, _measure_draw)
                        entry["text_layout"] = layout
                        bw, bh = layout["block_w"], layout["block_h"]
                    else:
                        bw = min(region_w, _mw2)
                    clear_y, step_box = _next_clear_y(wx, wy, bw, bh, occupied, H, step_gap)
                if clear_y + bh > H - _BOTTOM_PAD:
                    clear_y = max(0, H - _BOTTOM_PAD - bh)
                    step_box = (step_box[0], clear_y, step_box[2], clear_y + bh)
                entry["write_pos"] = (wx, clear_y)
                occupied.append(step_box)
                wy = clear_y + bh + step_gap

            # Boxed final answer (whiteboard-storyboard mode)
            if ann.get("box_answer") and entry.get("write_pos"):
                bx, by = entry["write_pos"]
                entry["answer_box"] = (bx - 10, by - 8, bx + bw + 12, by + bh + 8)
                wy += 10

        elif action == "draw_arrow":
            entry["write_end"] = t + 0.6
            fb = _resolve_box(ann, ocr_index, W, H, key_prefix="from_", option_positions=option_positions)
            tb = _resolve_box(ann, ocr_index, W, H, key_prefix="to_", option_positions=option_positions)
            if fb and tb:
                entry["arrow_params"] = ((fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2,
                                         (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2)

        elif action == "match_pair":  # connect a List-I item to its List-II match
            entry["write_end"] = t + 0.8
            arrow = match_routes.get(i)   # diagonal stroke across the gutter (rank-routed)
            if arrow:
                entry["match_arrow"] = arrow

        elif action in ANSWER_ACTIONS:
            target = (ann.get("target", "") or ann.get("option", ""))
            entry["write_end"] = t + 1.0
            # Extract the option letter robustly: "(A)", "Option C", "b." → A/C/B.
            opt = re.sub(r"[^A-D]", "", str(target).upper())[:1]
            # Fall back to extrapolating the marker's spot when OCR missed it (the
            # answer option, often A, is the one most likely to have been dropped).
            obox = _option_box_or_infer(opt, option_positions) if opt else None
            if obox:
                ox1, oy1, ox2, oy2 = obox
                # Ring the correct option label "(X)" (green) + a tick beside it —
                # a clean affirmative mark instead of a slash that reads like a cross.
                entry["answer_ring"] = (ox1 - 7, oy1 - 5, min(ox1 + 48, ox2 + 6), oy2 + 5)

        elif action == "verdict_mark":  # ✓/✗ beside a statement (assertion / count Qs)
            entry["write_end"] = t + 0.7
            v = str(ann.get("verdict", "")).strip().lower()
            is_true = v in ("true", "correct", "right", "yes", "सही", "1", "t")
            pos = verdict_rows.get(i)
            if pos:
                # Robust reconstructed column: guaranteed one mark per statement.
                mx, cy = int(pos[0]), pos[1]
                cand = (mx - 4, cy - 14, mx + 28, cy + 14)
                for _ in range(6):           # nudge only off an earlier ✓/✗, not text
                    if not _boxes_overlap(cand, occupied, pad=2):
                        break
                    cy += 26
                    cand = (mx - 4, cy - 14, mx + 28, cy + 14)
                entry["verdict_params"] = (mx, cy, is_true)
                occupied.append(cand)
                temp_schedule.append(entry)
                continue
            box = _resolve_verdict_box(ann.get("target"),
                                       enriched_ocr.get("elements") or [], stmt_ceiling)
            if not box:                          # non-statement target → generic resolve
                box = _resolve_box(ann, ocr_index, W, H, option_positions=option_positions,
                                   y_max=stmt_ceiling)
            if box:
                x1, y1, x2, y2 = box
                cy = (y1 + y2) / 2
                # Place the ✓/✗ just PAST the end of the statement's text on this row,
                # in the clear gap before the right-margin workspace/diagram column.
                # (The target is only the "अभिकथन A"/"कारण R" label, so we widen to the
                # full statement line via the OCR boxes on this row.) This keeps the
                # mark beside its statement and clear of both the figure and the logo.
                row_right = max([b[2] for b in occupied
                                 if b[1] - 4 < cy < b[3] + 4 and b[0] < ws_zone_x0] + [x2])
                mx = int(min(row_right + 18, ws_zone_x0 - 36))
                if mx < row_right:           # statement runs into the column: use margin
                    mx = int(min(row_right + 18, W - 46))
                cand = (mx - 4, cy - 14, mx + 28, cy + 14)
                if _boxes_overlap(cand, occupied, pad=2):
                    cy += 30  # nudge down to dodge another mark/note at this level
                    cand = (mx - 4, cy - 14, mx + 28, cy + 14)
                entry["verdict_params"] = (mx, cy, is_true)
                occupied.append(cand)

        elif action == "draw_diagram":  # build a schematic (flowchart/sequence/axis)
            dg = diagram_layouts.get(i)   # laid out + reserved in the pre-pass above
            if dg:
                entry["diagram_layout"] = dg
            entry["write_end"] = t + 5.0

        temp_schedule.append(entry)

    # Strikeout effect removed — wrong options are tracked internally (struck_options)
    # for answer-mark ordering but no strike lines are drawn on screen.

    # Pass 2: stretch text-writing durations to fill the spoken segment.
    stretch_actions = TEXT_ACTIONS + WRITE_ACTIONS
    for i, entry in enumerate(temp_schedule):
        act = entry["action"]
        if act in stretch_actions or act == "draw_diagram":
            t_curr = entry["time"]
            t_next = temp_schedule[i + 1]["time"] if i + 1 < len(temp_schedule) else total_duration - 1.0
            cap = 14.0 if act == "draw_diagram" else 8.0   # diagrams build over a longer span
            write_dur = max(1.0, min(cap, (t_next - t_curr) * 0.9))
            entry["write_duration"] = write_dur
            entry["write_end"] = t_curr + write_dur
        schedule.append(entry)

    # Phase gate: keep cross-outs and the answer mark OUT of the reading phase, so
    # the pen never "cuts" an option while the teacher is still reading the stem.
    # The reading phase ends at the last stem underline — but CAP that at ~30% of the
    # audio: some questions underline a term again LATE (e.g. re-reading कारण R while
    # explaining it), and those late underlines must not drag the gate to mid-lecture
    # and re-clamp every verdict/note back to the end.
    underline_times = [e["time"] for e in schedule if e["action"] == "underline_existing"]
    eval_acts = ("cross_out_word", "match_pair", "annotate_word") + VERDICT_ACTIONS + ANSWER_ACTIONS
    eval_times = [e["time"] for e in schedule if e["action"] in eval_acts]
    # Reading ends at the teacher's FIRST evaluative mark; underlines that happen
    # AFTER that are re-reading mid-lecture (e.g. pointing at a term while explaining
    # it) and must NOT inflate the gate. Anchoring to an ABSOLUTE reading time (not a
    # % of total) is essential: on a long statement-count question, % of total pushed
    # the gate to ~75s and clamped the early ✓/✗ verdicts (statements judged at ~29s)
    # to mid-video, badly desyncing them from the narration.
    first_eval = min(eval_times) if eval_times else total_duration
    reading_underlines = [tt for tt in underline_times if tt <= first_eval + 0.5]
    last_reading = max(reading_underlines) if reading_underlines else 0.0
    gate = max(last_reading, min(0.16 * total_duration, 12.0))
    for e in schedule:
        if not exact_mode and e["action"] in eval_acts and e["write_start"] < gate:
            dur = max(0.7, e["write_end"] - e["write_start"])
            e["write_start"] = gate
            e["write_end"] = gate + dur

    # Pace the teaching marks so the board is never blank for a long hold and never
    # dumps the whole solution in a rushed burst at the very end.
    #
    # The old guard kept the model's raw times whenever their SPAN was wide — but a
    # wide span is NOT an even spread. A lone early note plus a late cluster spans the
    # whole video yet leaves the middle blank, which is exactly how a back-loaded
    # audio renders: the teacher reads/explains for minutes, then every derivation
    # step is matched into the final seconds. The span test passed, so the board sat
    # static the whole time and then dumped the solution at the end.
    #
    # So we trigger on the DISTRIBUTION, not the span: if the board would sit static
    # longer than a comfortable hold — a big blank lead-in before the first mark, OR
    # a big gap between consecutive marks — re-space EVERY teaching mark evenly across
    # the teaching window. A timeline that is already well synced has only small gaps,
    # so its `max_hold` stays under the limit and it is left completely untouched.
    teach_actions = [e for e in schedule
                     if e["action"] not in ("underline_existing",) + ANSWER_ACTIONS]
    if not exact_mode and len(teach_actions) >= 3:
        ordered = sorted(teach_actions, key=lambda e: e["write_start"])
        first_ul = min(underline_times) if underline_times else 0.0
        # Teaching window: opens once the stem has been read (so we never write over
        # the teacher still reading the question) and closes before the conclusion.
        # Anchor to the FIRST underline, NOT `gate`: gate is the cross-out/answer
        # phase gate and inflates to mid-lecture whenever the only evaluative mark is
        # a late answer (then every re-reading underline counts as "reading"). Using
        # it here would collapse the window to the final seconds and cram everything.
        win_start = max(first_ul + 12.0, 0.10 * total_duration)
        times = [e["write_start"] for e in ordered]
        # win_end closes before the conclusion, but must NEVER precede the last
        # teaching anchor: a long derivation can legitimately run to ~97% of the clip,
        # and clamping those final steps back to 0.90*total collapses them onto a
        # single instant (overlapping ink, then a static tail). Extend the window to
        # cover the real last step so anchored late steps keep their own slots.
        win_end = min(total_duration - 3.0,
                      max(0.90 * total_duration, max(times) + 0.5))
        lead_in = times[0] - win_start                      # blank board before step 1
        gaps = [times[k + 1] - times[k] for k in range(len(times) - 1)]
        max_hold = max([lead_in] + gaps)
        # Tolerate a natural pause; re-space only when a hold is long enough to read
        # as "the video froze" — ~18% of the runtime, floored at 22 s for short clips.
        hold_limit = max(22.0, 0.18 * total_duration)
        if win_end > win_start and max_hold > hold_limit:
            # (1) ANCHOR-AWARE re-spacing. The old code laid every step on a blind even
            # grid, ignoring WHEN it is spoken — so a step the teacher says at 0:46
            # could be written at 0:14, ~30 s early (the desync complaint). Instead,
            # compute the even slot but CLAMP each step to a window around its audio
            # anchor (its pre-spaced write_start = the matched spoken time): a step may
            # appear at most EARLY_CAP before it is said, and at most LATE_CAP after.
            # This still de-clusters bursts and pulls the timeline forward, but every
            # mark stays in lip-sync with the narration instead of racing ahead of it.
            span = win_end - win_start
            anchors = list(times)                  # audio-aligned times, pre-mutation
            early_cap = max(4.0, 0.04 * total_duration)   # how far AHEAD of speech allowed
            late_cap = max(8.0, 0.08 * total_duration)    # writing a bit AFTER speech is fine
            new_times = []
            for k in range(len(ordered)):
                slot = win_start + (span * k / max(1, len(ordered) - 1))
                a = anchors[k]
                new_times.append(min(max(slot, a - early_cap), a + late_cap))
            # Preserve order with a minimum visible gap, never past the window end.
            min_gap = 0.8
            for k in range(1, len(new_times)):
                if new_times[k] < new_times[k - 1] + min_gap:
                    new_times[k] = new_times[k - 1] + min_gap
            for k, e in enumerate(ordered):
                dur = max(0.7, e["write_end"] - e["write_start"])
                nt = min(new_times[k], win_end)
                e["write_start"] = nt
                e["write_end"] = nt + dur

            # (2) FILL THE READING PHASE. With steps now anchored to their (later)
            # spoken times, the lead-in is no longer papered over by dragging steps
            # early — so it would sit blank again. Spread the stem underlines across
            # the lead-in instead: the teacher is shown marking each given AS it is
            # read, genuine in-sync activity that fills the gap WITHOUT desyncing the
            # solution. Only the lead-in case needs this (a mid-solution gap doesn't),
            # and the whole block is inside the blank-board trigger, so a healthy,
            # well-synced video never reaches here and is left byte-identical.
            if lead_in > hold_limit:
                ul_ordered = sorted(
                    (e for e in schedule if e["action"] == "underline_existing"),
                    key=lambda e: e["write_start"])
                if len(ul_ordered) >= 2:
                    read_start = max(first_ul, 2.0)
                    read_end = max(read_start + 4.0, new_times[0] - 1.5)
                    rspan = read_end - read_start
                    for k, e in enumerate(ul_ordered):
                        d = max(0.6, e["write_end"] - e["write_start"])
                        nt = read_start + (rspan * k / max(1, len(ul_ordered) - 1))
                        e["write_start"] = nt
                        e["write_end"] = nt + d

            # Keep the answer mark strictly AFTER the (re-spaced) final step, so the
            # circle never appears before the derivation that justifies it. Only push
            # answers that are now too early; a correctly-late answer keeps its time.
            last_step_end = max(e["write_end"] for e in ordered)
            for e in schedule:
                if e["action"] in ANSWER_ACTIONS and e["write_start"] < last_step_end + 1.0:
                    d = max(0.7, e["write_end"] - e["write_start"])
                    e["write_start"] = min(last_step_end + 1.0, total_duration - 1.0)
                    e["write_end"] = e["write_start"] + d

    # ── Deterministic order for option-elimination marks ──────────────────────
    # Gemini frequently FRONT-LOADS the answer + cross-outs to the very start (it
    # knows the answer immediately) and the cue-based audio sync can't always pull
    # them back, so the ✓/✗ on options pop up BEFORE the teacher evaluates them. A
    # teacher eliminates options DURING the explanation and marks the answer LAST.
    # Enforce that invariant — but only fix marks that actually violate it, so a
    # correctly-timed late answer/cross-out keeps its spoken sync.
    if option_positions and not exact_mode:
        def _retime(e, nt):
            dur = max(0.7, e["write_end"] - e["write_start"])
            e["write_start"] = nt
            e["write_end"] = nt + dur

        def _has_cue(e):
            # A real rejection/conclusion cue (the teacher's actual words) means the sync
            # could anchor this mark to the MOMENT it is spoken; a missing/tiny cue means
            # it could only be pinned to where the option is first read/mentioned.
            return len(str(e.get("spoken_cue") or "").split()) >= 3

        # Reading runs longer on long questions (the teacher reads all four options), so
        # the gate alone under-estimates it; treat the first ~12% as reading too.
        read_end = max(gate + 0.5, 0.12 * total_duration)
        crosses = [e for e in schedule if e["action"] == "cross_out_word"]
        answers = [e for e in schedule if e["action"] in ANSWER_ACTIONS]
        if crosses or answers:
            tail = total_duration - 1.5
            # HYBRID: a cross WITH a genuine rejection cue stays at its spoken moment
            # (natural, per-option timing). A cue-less cross can only be pinned to where
            # the option is MENTIONED — a beat before the teacher finishes explaining why
            # it is wrong — so instead of striking it early, CLUSTER it in the conclusion:
            # the teacher finishes the reasoning, THEN strikes the wrong options and
            # circles the answer. Nothing is ever struck before its reasoning.
            cued = [e for e in crosses if _has_cue(e) and e["write_start"] >= read_end]
            loose = sorted((e for e in crosses if e not in cued),
                           key=lambda e: e["write_start"])
            # Crosses must come AFTER the last derivation step — a teacher never
            # strikes an option before finishing the worked solution. Include the
            # last write_step time so numerical questions with a long derivation
            # (e.g. 181–280 s) don't fire crosses mid-solution at 55% of total.
            _step_times = [e["write_start"] for e in schedule if e["action"] in WRITE_ACTIONS]
            _last_step = max(_step_times, default=gate)
            win_start = max(read_end + 1.0, 0.55 * total_duration,
                            max((e["write_start"] for e in cued), default=0.0) + 1.0,
                            _last_step + 0.5)
            win_start = min(win_start, max(read_end + 1.0, tail - 4.0))
            span = max(0.0, (tail - win_start) - 1.5)   # leave the final slot for the answer
            m = len(loose)
            for k, e in enumerate(loose):
                _retime(e, win_start + (span * k / m if m > 1 else 0.0))
            # The answer comes LAST — after every cross-out. Only move it when it is early
            # or would otherwise precede an elimination.
            cross_ts = [e["write_start"] for e in schedule if e["action"] == "cross_out_word"]
            last_cross = max(cross_ts, default=(win_start if loose else gate + 1.0))
            for e in sorted(answers, key=lambda e: e["write_start"]):
                if e["write_start"] < gate + 1.0 or (cross_ts and e["write_start"] < last_cross):
                    _retime(e, min(last_cross + 1.2, tail))
                last_cross = max(last_cross, e["write_start"])

    # ── Page-turn wipe times ──────────────────────────────────────────────────
    # The board is not an infinite scroll. When a new page starts writing its
    # worked solution, the previous page's solution writing is erased so the two
    # never overlap (the heavy-overlap defect). Each workspace-write action from
    # page p renders only until the first workspace write of page p+1 begins; the
    # printed question stays (it is the background). Question annotations (circle /
    # underline / cross / answer ring / verdict / match) sit on the question and
    # are NOT page-scoped, so they persist for the rest of the video.
    WIPEABLE = set(WRITE_ACTIONS) | set(TEXT_ACTIONS) | {"draw_diagram"}
    pages_with_writes = sorted({e.get("page", 1) for e in schedule
                                if e["action"] in WIPEABLE})
    if len(pages_with_writes) > 1:
        first_write = {}
        for e in schedule:
            if e["action"] in WIPEABLE:
                p = e.get("page", 1)
                first_write[p] = min(first_write.get(p, e["write_start"]),
                                     e["write_start"])
        for e in schedule:
            if e["action"] not in WIPEABLE:
                continue
            later = [q for q in pages_with_writes if q > e.get("page", 1)]
            if later:
                e["_wipe_after"] = first_write[later[0]]

    return schedule
