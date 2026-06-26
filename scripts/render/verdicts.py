"""Verdict (tick/cross) placement for assertion-reason and statement-count
questions: resolve a statement heading by ordinal and lay out a tidy
right-margin column of marks. Extracted from render_video.py (Step 3)."""

import re


_ROMAN_ORD = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}


def _resolve_verdict_box(target, elements, ceiling):
    """Resolve a verdict's statement HEADING by ordinal position.

    Fuzzy text matching fails here because EasyOCR garbles the enumerator — it
    reads "कथन I" as "कथन lः" and "कथन II" as "कथन १ः", so "I" and "II" can't be
    told apart by score (and "कथन I" is even a substring of "कथन II"). But the
    Gemini `target` is clean, and the statement headings are simply the rows that
    START with the base label ("कथन"/"अभिकथन"/"कारण") above the options ceiling.
    Take the enumerator from the clean target and pick that heading by order.
    """
    parts = str(target or "").split()
    if not parts:
        return None
    base = parts[0]                                  # कथन / अभिकथन / कारण
    ordinal = None
    if len(parts) >= 2:
        enr = parts[-1].upper().strip(".:)( ")
        ordinal = _ROMAN_ORD.get(enr) or (int(enr) if enr.isdigit() else None)
    heads = [el["bounds"] for el in elements
             if el.get("bounds") and el["bounds"][1] < ceiling
             and (el.get("text") or "").strip().startswith(base)]
    heads.sort(key=lambda b: b[1])                   # top → bottom
    if not heads:
        return None
    # numbered statements map by order; a lone label (assertion A / reason R) → its
    # heading (the lowest match, skipping any higher stem mention).
    idx = (ordinal - 1) if (ordinal and ordinal - 1 < len(heads)) else len(heads) - 1
    return tuple(heads[idx])


def _verdict_row_positions(annotations, elements, ceiling, W, ws_x0):
    """Place every statement's ✓/✗ in a tidy right-margin column — one per
    statement — so NO mark is ever dropped or shifted onto the wrong row.

    Marks must sit on the LABEL line of each statement. Two reconstruction paths:

    1. PRIMARY (enumerator-anchored): find the OCR lines that BEGIN with a statement
       enumerator (``A.``/``(B)``/``1)``/``कथन``). If exactly N are found, map the
       i-th verdict (Gemini's vertical order) to the i-th label line. This ignores
       wrapped stem/continuation lines entirely, so a multi-line stem or a two-line
       statement can never push the whole column up by a row (the q40 failure).

    2. FALLBACK (proportional, stem-clipped): if enumerators are garbled — EasyOCR
       sometimes mangles them — clip away everything ABOVE the first enumerator (kills
       a multi-line stem), then map each statement's box_2d centre proportionally onto
       the remaining block and snap to the nearest unused line centre. Gemini's
       ABSOLUTE box_2d coords are unreliable but its RELATIVE order is correct.

    Returns ``{annotation_index: (mark_x, mark_cy)}``.
    """
    vlist = [(i, ann) for i, ann in enumerate(annotations)
             if ann.get("action") == "verdict_mark"]
    # 2-verdict assertion-reason (अभिकथन A / कारण R) stays on the validated legacy
    # path; the reconstructed column is for true multi-statement count questions.
    if len(vlist) < 3:
        return {}
    N = len(vlist)
    _SKIP = ("निम्नलिखित", "विचार", "नीचे", "विकल्प", "चुनिए", "कीजिए",
             "उत्तर", "कूट", "सही उत्तर", "कथनों पर")

    # Gather candidate text elements on the LEFT side, between the stem and options.
    cand = []
    for el in elements:
        b = el.get("bounds")
        if not b:
            continue
        cy = (b[1] + b[3]) / 2
        if not (30 < cy < ceiling - 2):
            continue
        if b[0] > 0.55 * W:             # right-margin notes/diagram, not statements
            continue
        cand.append((cy, b, (el.get("text") or "").strip()))
    if len(cand) < N:
        return {}
    cand.sort(key=lambda c: c[0])

    # Cluster candidate boxes into text LINES, keeping each line's joined text so a mark
    # snaps to a real line centre (never floats between a statement's two wrapped lines).
    def _merge(group):
        cym = sum(g[0] for g in group) / len(group)
        x0 = min(g[1][0] for g in group); y0 = min(g[1][1] for g in group)
        x1 = max(g[1][2] for g in group); y1 = max(g[1][3] for g in group)
        txt = " ".join(g[2] for g in sorted(group, key=lambda g: g[1][0]) if g[2])
        return {"cy": cym, "b": (x0, y0, x1, y1), "txt": txt}

    lines, group = [], [cand[0]]
    for c in cand[1:]:
        if c[0] - group[-1][0] <= 14:
            group.append(c)
        else:
            lines.append(_merge(group)); group = [c]
    lines.append(_merge(group))

    # Lines that BEGIN with a statement enumerator: "A."/"(B)"/"1)"/"कथन"/"अभिकथन"/...
    _ENUM = re.compile(r"^\(?\s*(?:[A-Ja-j]|[1-9]|I{1,3}|IV|VI{0,3}|V)\s*[\.\)\:\-]")
    _LABEL = ("कथन", "अभिकथन", "कारण", "सूची")
    starts = [k for k, ln in enumerate(lines)
              if _ENUM.match(ln["txt"]) or ln["txt"].startswith(_LABEL)]

    # Verdict order top→bottom: trust Gemini's box_2d ordering, else annotation order.
    boxes = [ann.get("box_2d") for _, ann in vlist]
    if all(b and len(b) == 4 for b in boxes):
        order = [i for (i, _), _ in sorted(zip(vlist, boxes), key=lambda z: z[1][0])]
    else:
        order = [i for i, _ in vlist]

    # ── PRIMARY: enumerators read cleanly → map i-th verdict to i-th label line.
    if len(starts) == N:
        text_right = max(ln["b"][2] for ln in lines[starts[0]:])
        mark_x = int(min(text_right + 22, ws_x0 - 30, W - 46))
        return {i: (mark_x, lines[starts[rank]]["cy"])
                for rank, i in enumerate(order)}

    # ── FALLBACK: clip everything ABOVE the first enumerator (kills a multi-line stem),
    # else drop stem/instruction lines by keyword, then place proportionally.
    if starts:
        block = lines[starts[0]:]
    else:
        block = [ln for ln in lines if not any(k in ln["txt"] for k in _SKIP)]
    if len(block) < N:
        return {}
    line_cys = [ln["cy"] for ln in block]
    top = block[0]["b"][1]; bot = block[-1]["b"][3]
    if bot - top < 8:
        return {}
    text_right = max(ln["b"][2] for ln in block)
    mark_x = int(min(text_right + 22, ws_x0 - 30, W - 46))

    if all(b and len(b) == 4 for b in boxes) and (max(b[2] for b in boxes) >
                                                  min(b[0] for b in boxes)):
        gtop = min(b[0] for b in boxes)         # statement A's top in Gemini frame
        gbot = max(b[2] for b in boxes)         # last statement's bottom
        prop = [(i, top + ((b[0] + b[2]) / 2 - gtop) / (gbot - gtop) * (bot - top))
                for (i, ann), b in zip(vlist, boxes)]
    else:                                       # no usable box_2d → even bands by order
        prop = [(i, top + (bot - top) * (r + 0.5) / N)
                for r, (i, ann) in enumerate(vlist)]

    # Snap each statement (top→bottom) to the nearest UNUSED line centre, monotonic so
    # marks never reorder or pile onto one line.
    out, used = {}, -1
    for i, pcy in sorted(prop, key=lambda p: p[1]):
        cands = [(abs(lc - pcy), idx) for idx, lc in enumerate(line_cys) if idx > used]
        if cands:
            _, idx = min(cands); used = idx; out[i] = (mark_x, line_cys[idx])
        else:
            out[i] = (mark_x, pcy)              # ran out of lines → keep proportional
    return out
