#!/usr/bin/env python3
"""Extract question text and option positions from question image using EasyOCR."""

import easyocr
import json
import re
import sys
import numpy as np
from PIL import Image
from ocr_utils import enrich_ocr_data

# Matches an ISOLATED single-letter placeholder token like "(A)", "A)", "[B)",
# "{D]", "A." — but NOT a long string that merely contains "(A)".
_PLACEHOLDER_RE = re.compile(r"^[\(\[\{]?\s*([A-Za-z])\s*[\)\]\}\.\-]?$")


# Lazily-created English-only reader, used to recover option markers on Latin
# slides where the bilingual (hi+en) reader mangles small Latin glyphs/digits
# (it tends to read "(A)" as Devanagari, e.g. "(8)"). Cached so it loads once.
_EN_READER = None


def _get_en_reader():
    global _EN_READER
    if _EN_READER is None:
        try:
            _EN_READER = easyocr.Reader(["en"], verbose=False)
        except Exception:
            _EN_READER = False  # mark as tried-and-failed
    return _EN_READER or None


def _recover_options_by_order(image_path, band_top, image_width, image_height):
    """Recover option-marker boxes STRUCTURALLY, by row order, not by glyph.

    On many printed slides EasyOCR systematically misreads the small
    parenthesised option letters — often with a consistent off-by-one shift
    ("(A)"→"(B)", "(B)"→"(C)", …) — so the LETTER cannot be trusted, but the
    vertical ROW ORDER can. This re-OCRs the left-margin column of the options
    band with an English allowlist, clusters the marker tokens into rows, and
    assigns A, B, C, D… purely by their top-to-bottom position.

    Returns {LETTER: (x1, y1, x2, y2)} in pixel coords (empty if not recoverable).
    """
    reader = _get_en_reader()
    if reader is None or image_height - band_top < 20:
        return {}
    try:
        img = Image.open(image_path).convert("RGB")
        x_lim = max(40, int(image_width * 0.16))      # markers hug the far left
        scale = 5
        crop = img.crop((0, int(band_top), x_lim, image_height))
        up = crop.resize((crop.width * scale, crop.height * scale))
        toks = reader.readtext(np.array(up), allowlist="ABCD()[].",
                               text_threshold=0.2, low_text=0.12)
    except Exception as e:
        print(f"  option row-recovery skipped ({e})")
        return {}

    cands = []
    for bbox, text, conf in toks:
        t = text.strip()
        # A real option marker has a bracket AND a letter ("(A)"), or is an empty
        # bracket pair where the letter was lost ("()"). This rejects stem
        # fragments — a lone "(", stray punctuation — even under the allowlist.
        has_bracket = any(c in t for c in "()[]")
        has_letter = any(c in t for c in "ABCDabcd")
        is_marker = (has_bracket and has_letter) or t in ("()", "[]", "( )")
        if not is_marker or conf < 0.5:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        box = (int(min(xs) / scale), int(min(ys) / scale + band_top),
               int(max(xs) / scale), int(max(ys) / scale + band_top))
        if box[0] > image_width * 0.13:                # not at the left margin
            continue
        h = box[3] - box[1]
        if h < 6 or h > 0.12 * image_height:           # implausible marker size
            continue
        cands.append(box)
    if len(cands) < 2:
        return {}

    # Cluster left-margin tokens into rows (one option per row); keep the
    # leftmost token of each row as the marker anchor.
    cands.sort(key=lambda b: (b[1] + b[3]) / 2)
    med_h = sorted(b[3] - b[1] for b in cands)[len(cands) // 2]
    tol = max(14, med_h * 0.7)
    rows = []  # [cy, marker_box]
    for b in cands:
        cy = (b[1] + b[3]) / 2
        for row in rows:
            if abs(cy - row[0]) <= tol:
                if b[0] < row[1][0]:
                    row[1] = b
                row[0] = (row[0] + cy) / 2
                break
        else:
            rows.append([cy, b])
    rows.sort(key=lambda r: r[0])

    # Drop a phantom leading row: options are evenly spaced, so if the first
    # inter-row gap is far larger than the rest, the top "row" is a stem line the
    # band reached, not option (A). Guards against the band starting too high.
    if len(rows) >= 3:
        gaps = [rows[i + 1][0] - rows[i][0] for i in range(len(rows) - 1)]
        rest = sorted(gaps[1:])
        med = rest[len(rest) // 2]
        if med > 0 and gaps[0] > 1.7 * med:
            rows = rows[1:]
    return {chr(ord("A") + i): tuple(row[1]) for i, row in enumerate(rows[:6])}


def detect_diagram_placeholders(reader, image_path, results, options_top,
                                image_width, image_height):
    """
    Find diagram/flowchart blanks like (A) (B) (C) (D) that sit INSIDE a figure.

    Generic and question-agnostic: scans OCR tokens for isolated single-letter
    brackets located between the question heading and the options list, then
    re-OCRs an upscaled crop of that band to recover small placeholders the
    full-image pass misses. Returns {LABEL: (x1, y1, x2, y2)} in pixel coords.
    Distinguishes figure placeholders from the multiple-choice option list
    (those tokens start long option rows, not isolated letters).
    """
    placeholders = {}

    def _valid_ph_box(x1, y1, x2, y2):
        """
        Reject candidates that are not plausible diagram-blank letters:
          - too large (a real "(A)" token is small; big boxes are misreads or a
            whole table cell/region matched as a letter),
          - inside the top-right PW watermark/logo (e.g. the logo's 'P').
        """
        w, h = x2 - x1, y2 - y1
        if w > 0.05 * image_width or h > 0.06 * image_height:
            return False
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if cx > 0.86 * image_width and cy < 0.16 * image_height:  # corner logo
            return False
        return True

    # Heading band = topmost long sentence; the diagram sits below it.
    heading_bottom = 0
    for bbox, text, _ in results:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        if (min(ys) < image_height * 0.22 and (max(xs) - min(xs)) > image_width * 0.30
                and len(text.strip()) > 15):
            heading_bottom = max(heading_bottom, max(ys))
    diagram_top = heading_bottom + 4
    diagram_bottom = options_top if options_top else image_height

    def _consider(bbox, text):
        m = _PLACEHOLDER_RE.match(text.strip())
        if not m:
            return
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        cy = (min(ys) + max(ys)) / 2
        if not (diagram_top <= cy <= diagram_bottom):
            return
        box = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        if not _valid_ph_box(*box):
            return
        placeholders.setdefault(m.group(1).upper(), box)

    # 1) isolated tokens already found by the full-image OCR.
    for bbox, text, _ in results:
        _consider(bbox, text)

    # 2) re-OCR an upscaled crop of the diagram band to recover tiny labels.
    band = None
    try:
        img = Image.open(image_path).convert("RGB")
        if diagram_bottom - diagram_top > 20:
            scale = 3
            crop = img.crop((0, int(diagram_top), image_width, int(diagram_bottom)))
            up = crop.resize((crop.width * scale, crop.height * scale))
            band = (up, scale)
            for bbox, text, _ in reader.readtext(np.array(up), text_threshold=0.35, low_text=0.25):
                m = _PLACEHOLDER_RE.match(text.strip())
                if not m:
                    continue
                label = m.group(1).upper()
                if label in placeholders:
                    continue
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                box = (
                    int(min(xs) / scale), int(min(ys) / scale + diagram_top),
                    int(max(xs) / scale), int(max(ys) / scale + diagram_top),
                )
                if _valid_ph_box(*box):
                    placeholders[label] = box
    except Exception as e:
        print(f"  placeholder re-OCR skipped ({e})")

    # 3) Targeted gap recovery. If the detected labels form a sequence with a hole
    # (e.g. A, B, D -> C is missing), a flowchart blank for that letter almost
    # certainly exists but was scored below threshold. Diagram blanks line up in a
    # few vertical columns, so re-OCR a TIGHT, heavily-upscaled strip around each
    # detected column at a very low threshold, accepting a token ONLY if it is
    # exactly the missing letter — knowing which letter to look for makes a
    # low-confidence hit safe, and the tight crop lets the detector actually
    # propose the tiny region (a full-width band scan misses it entirely).
    letters = sorted(ord(l) - ord("A") for l in placeholders if len(l) == 1 and l.isalpha())
    if len(letters) >= 2:
        gaps = [chr(ord("A") + i) for i in range(letters[0], letters[-1] + 1)
                if i not in letters]
        if gaps:
            try:
                img2 = Image.open(image_path).convert("RGB")
                # Cluster detected placeholder centres into columns.
                cxs = sorted(((b[0] + b[2]) / 2) for b in placeholders.values())
                cols, tol = [], max(40, image_width * 0.04)
                for cx in cxs:
                    if cols and cx - cols[-1][-1] <= tol:
                        cols[-1].append(cx)
                    else:
                        cols.append([cx])
                centres = [sum(c) / len(c) for c in cols]
                strip_h = max(20, diagram_bottom - diagram_top)
                for want in gaps:
                    best = None
                    for ctr in centres:
                        sx1 = int(max(0, ctr - 120))
                        sx2 = int(min(image_width, ctr + 120))
                        sub = img2.crop((sx1, int(diagram_top), sx2, int(diagram_bottom)))
                        up = sub.resize((sub.width * 4, sub.height * 4))
                        # Normal detection thresholds (very low ones fragment the
                        # region and find nothing); we instead accept the returned
                        # match at whatever confidence because we know the letter.
                        for bbox, text, conf in reader.readtext(
                                np.array(up), text_threshold=0.3, low_text=0.2):
                            m = _PLACEHOLDER_RE.match(text.strip())
                            if not m or m.group(1).upper() != want:
                                continue
                            if best is None or conf > best[-1]:
                                xs = [p[0] for p in bbox]
                                ys = [p[1] for p in bbox]
                                cand = (
                                    int(sx1 + min(xs) / 4), int(diagram_top + min(ys) / 4),
                                    int(sx1 + max(xs) / 4), int(diagram_top + max(ys) / 4),
                                )
                                if _valid_ph_box(*cand):
                                    best = cand + (conf,)
                    if best is not None:
                        placeholders[want] = best[:4]
                        print(f"  recovered placeholder '{want}' via targeted "
                              f"column re-OCR (conf {best[4]:.2f})")
            except Exception as e:
                print(f"  placeholder gap recovery skipped ({e})")

    return placeholders


def extract_question_info(image_path):
    """
    Extract text content, option bounding boxes, question region, and enriched OCR data.

    Returns:
        tuple: (full_text, option_positions, question_bbox, enriched_ocr)
            - full_text: All OCR text concatenated
            - option_positions: Dict mapping option letters to bounding boxes
              e.g. {"A": [[x1,y1], [x2,y1], [x2,y2], [x1,y2]], ...}
            - question_bbox: (x1, y1, x2, y2) bounding box covering the
              question text region (everything above the options), or None
            - enriched_ocr: Enriched structured OCR data dict
    """
    # Load image for dimensions
    img = Image.open(image_path)
    image_width, image_height = img.size

    # Run EasyOCR. We always enable Hindi (Devanagari) alongside English so the
    # same reader handles English, Hindi, and mixed questions (e.g. a Hindi
    # biology MCQ whose options still contain Latin tokens like "hCG"/"HPL").
    # EasyOCR allows Hindi only in combination with English, which is exactly
    # what we want here.
    reader = easyocr.Reader(["hi", "en"], verbose=False)
    results = reader.readtext(image_path)

    full_text = " ".join([text for _, text, _ in results])

    option_positions = {}
    option_y_positions = []

    # An option row STARTS with its marker, e.g. "(A) ...", "[B] ...", "C) ...".
    # Requiring the marker at the start (not merely contained) prevents the
    # question heading "...(A), (B), (C), (D)..." from being mistaken for options.
    option_start_re = re.compile(r"^\s*[\(\[]?\s*([a-dA-D48])\s*[\)\]\.]")

    for bbox, text, confidence in results:
        m = option_start_re.match(text.strip())
        if m and len(text.strip()) >= 3:  # supports compact rows like "(C)6"
            opt = m.group(1).upper()
            # EasyOCR commonly reads option markers "(A)" and "(B)" as "(4)"
            # and "(8)" in low-resolution slides.
            opt = {"4": "A", "8": "B"}.get(opt, opt)
            converted_bbox = [[int(coord) for coord in pt] for pt in bbox]
            option_positions[opt] = converted_bbox
            option_y_positions.append(min(p[1] for p in bbox))

    # Compute the question text bounding box: the region above the options
    question_bbox = None
    if results:
        all_x = []
        all_y = []
        for bbox, _, _ in results:
            for p in bbox:
                all_x.append(p[0])
                all_y.append(p[1])

        if option_y_positions:
            # Question region = everything above the first option
            options_top = min(option_y_positions)
            q_y_max = options_top - 5
        else:
            # No options detected — treat the entire text area as question
            q_y_max = max(all_y)

        question_bbox = (
            int(min(all_x)),
            int(min(all_y)),
            int(max(all_x)),
            int(q_y_max),
        )

    # Option-marker recovery: on Latin (English) slides the bilingual reader often
    # mangles the small "(A)…(D)" glyphs (Devanagari confusion, off-by-one letter
    # shifts), leaving options missing or mislabelled. When the slide is
    # Latin-dominant and we have fewer than four options, re-detect them
    # structurally by row order — which is reliable even when the glyph is not.
    latin = sum(1 for c in full_text if "a" <= c.lower() <= "z")
    devanagari = sum(1 for c in full_text if "ऀ" <= c <= "ॿ")
    if latin >= devanagari and len(option_positions) < 4:
        # Start the band a little ABOVE the detected question bottom: the primary
        # pass often mis-includes the first option row in the question region, so
        # question_bbox[3] can sit just below option (A). The marker filter and
        # row-spacing guard reject any stem text this margin happens to include.
        qb_bottom = question_bbox[3] if question_bbox else int(image_height * 0.5)
        band_top = max(0, int(qb_bottom - 0.06 * image_height))
        recovered = _recover_options_by_order(
            image_path, band_top, image_width, image_height)
        if len(recovered) > len(option_positions) and len(recovered) >= 2:
            option_positions = {
                letter: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                for letter, (x1, y1, x2, y2) in recovered.items()
            }
            option_y_positions = [min(p[1] for p in b)
                                  for b in option_positions.values()]
            print(f"  Recovered options by row-order (glyph OCR unreliable): "
                  f"{sorted(option_positions)}")

    # Enrich OCR data with types, index, and free space regions
    enriched_ocr = enrich_ocr_data(results, image_width, image_height, question_bbox)

    # Detect diagram/flowchart placeholders (A)/(B)/(C)/(D) inside any figure.
    options_top = min(option_y_positions) if option_y_positions else None
    placeholders = detect_diagram_placeholders(
        reader, image_path, results, options_top, image_width, image_height)
    enriched_ocr["placeholders"] = placeholders

    print(f"  OCR extracted {len(results)} text regions")
    if option_positions:
        print(f"  Found options: {list(option_positions.keys())}")
    if question_bbox:
        print(f"  Question region: {question_bbox}")
    if placeholders:
        print(f"  Detected diagram placeholders: {sorted(placeholders.keys())}")
    print(f"  Detected {len(enriched_ocr['free_spaces'])} free space regions")

    return full_text, option_positions, question_bbox, enriched_ocr


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "input/question.png"
    text, positions, q_bbox, enriched = extract_question_info(image)
    print(f"\nQuestion text:\n{text}")
    print(f"\nOption positions:\n{json.dumps(positions, indent=2)}")
    print(f"\nQuestion bounding box:\n{q_bbox}")
