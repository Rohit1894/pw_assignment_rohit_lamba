"""Pre-render validation gate.

Catches a BAD annotation set BEFORE it is turned into a video, so the system
never spends minutes rendering — and a student never watches — a board that is
structurally broken or about the wrong subject.

Two severities:
  * ERROR  — blocks the render (guaranteed-bad output: blank video, wrong-subject
             annotations, a question type that never marks its answer).
  * WARN   — advisory only; the render proceeds (quality issues: too many/too long
             notes, oversized diagram labels, type/action mismatches).

This gate validates the ANNOTATIONS (class 1: bad data reaching the renderer). It
does NOT and cannot catch render-side geometry bugs (class 2: good annotations
placed wrongly) — those are guarded inside render_video.py itself.

Run standalone on a saved file for CI/debugging:
    python scripts/validate_annotations.py output/annotations.json [--type mcq]
"""
import json
import os
import re
import sys

from action_schema import ACCEPTED_ACTIONS, normalize_actions

# ── Action vocabulary (canonical, from the shared schema) ────────────────────
KNOWN_ACTIONS = ACCEPTED_ACTIONS
ANSWER_ACTIONS = {"mark_answer"}        # canonical; tick_answer is normalised to this
NOTE_ACTIONS = {"annotate_word", "write_note"}
VERDICT_ACTIONS = {"verdict_mark"}

# Types whose teaching MUST end by marking one correct option.
TYPES_REQUIRING_ANSWER = {
    "mcq", "assertion_reason", "statement_count", "matching", "sequence",
    "flowchart_fill", "numerical",
}
KNOWN_TYPES = TYPES_REQUIRING_ANSWER | {"diagram_label", "other"}

# A question must never carry the distance-formula (physics/coordinate-geometry)
# annotations the old rule-based fallback emitted on the WRONG subject. This
# signature catches that leak precisely. To stay subject-neutral it only fires
# when the QUESTION ITSELF is not about distance/coordinates (see
# _DISTANCE_CONTEXT) — so a legitimate English coordinate-geometry problem is
# never blocked, while a biology/chemistry slide that leaked physics still is.
_DISTANCE_FORMULA = re.compile(r"x2\s*-\s*x1|y2\s*-\s*y1|d\s*=\s*[√v]\s*\(|√\(\(")
_DISTANCE_CONTEXT = re.compile(
    r"distance|coordinate|point|midpoint|section formula|दूरी|निर्देशांक|बिंदु",
    re.IGNORECASE,
)

# Quality thresholds (tuned so every currently-validated good output passes).
MAX_NOTES = 4               # prompt itself mandates "NEVER MORE THAN 4"
MAX_DIAGRAMS = 2            # prompt: "at most one or two diagrams"
NOTE_CHAR_LIMIT = 90        # a note longer than this is a sentence, not an anchor
NOTE_WORD_LIMIT = 14
NODE_LABEL_CHAR_LIMIT = 26  # board labels should be terse ("Hypothalamus", "GnRH")


def _norm_letter(target):
    """Normalise an option reference like "(B)" / "B." / "बी" / "b" -> "B"."""
    if not target:
        return None
    m = re.search(r"[A-Za-z]", str(target))
    return m.group(0).upper() if m else None


def _diagram_labels(action):
    """Yield every node/edge label inside a draw_diagram action."""
    dia = action.get("diagram")
    if not isinstance(dia, dict):
        return
    for node in dia.get("nodes", []) or []:
        if isinstance(node, dict) and node.get("label"):
            yield ("node", str(node["label"]))
    for edge in dia.get("edges", []) or []:
        if isinstance(edge, dict) and edge.get("label"):
            yield ("edge", str(edge["label"]))


def validate_annotation_set(annotations, *, question_type=None,
                            option_positions=None, question_text=None):
    """Return (errors, warnings) — two lists of human-readable strings.

    `annotations`     : the final action list (post sync/normalise).
    `question_type`   : from the .meta.json sidecar; None if unknown (reuse/whisper).
    `option_positions`: {"A": bbox, ...} from OCR, used to sanity-check the answer.
    """
    errors, warnings = [], []
    qtype = (question_type or "").strip().lower() or None
    opt_keys = {k.upper() for k in (option_positions or {})}

    # ── Structural sanity ────────────────────────────────────────────────────
    if not isinstance(annotations, list):
        return ([f"annotations is not a list (got {type(annotations).__name__})"], [])
    n = len(annotations)
    if n == 0:
        errors.append("0 annotations — the render would be a BLANK video (slide + "
                      "audio only). Refusing to render.")
        return errors, warnings
    if n < 3:
        warnings.append(f"only {n} annotation(s) — the board may look empty for a "
                        f"full-length explanation.")

    # Validate against the one canonical schema: work on a copy whose action names are
    # canonicalised (tick_answer -> mark_answer, write_equation -> write_step, ...), so
    # a legacy alias is never mistaken for an "unknown" or a missing answer mark. Copy
    # the dicts first — never mutate the caller's annotations.
    actions = normalize_actions([dict(a) for a in annotations if isinstance(a, dict)])
    kinds = [a.get("action") for a in actions]
    kind_set = set(kinds)

    # Unknown action types render as nothing — flag them so they're not silent.
    unknown = sorted(k for k in kind_set if k and k not in KNOWN_ACTIONS)
    for k in unknown:
        warnings.append(f"unknown action type {k!r} — the renderer will ignore it.")

    if qtype and qtype not in KNOWN_TYPES:
        warnings.append(f"unrecognised question_type {qtype!r}.")
    elif not qtype:
        warnings.append("question_type unknown — type/action checks skipped; only "
                        "structural checks run (set it in the .meta.json sidecar).")

    # ── Wrong-subject guard (ERROR) ──────────────────────────────────────────
    # Fire only when the distance-formula signature appears on a non-numerical
    # question whose own text is NOT about distance/coordinates — i.e. the
    # annotation is genuinely off-subject, not a real coordinate-geometry problem.
    qtext_is_distance = bool(_DISTANCE_CONTEXT.search(question_text or ""))
    for a in actions:
        text = " ".join(str(a.get(k, "")) for k in ("text", "target"))
        if (_DISTANCE_FORMULA.search(text) and qtype != "numerical"
                and not qtext_is_distance):
            errors.append("wrong-subject annotation detected — distance-formula / "
                          f"algebra text {text.strip()[:50]!r} on a non-numerical "
                          "question whose text is not about distance/coordinates. "
                          "This is the old physics fallback leaking in.")
            break

    # ── Answer presence & validity ───────────────────────────────────────────
    answer_acts = [a for a in actions if a.get("action") in ANSWER_ACTIONS]
    needs_answer = qtype in TYPES_REQUIRING_ANSWER or (
        qtype is None and opt_keys and not (kind_set & VERDICT_ACTIONS))
    if needs_answer and not answer_acts:
        msg = ("no mark_answer/tick_answer — the video never tells the student which "
               "option is correct.")
        (errors if qtype in TYPES_REQUIRING_ANSWER else warnings).append(msg)
    if len(answer_acts) > 1:
        warnings.append(f"{len(answer_acts)} answer marks — only one option should be "
                        "marked correct.")
    for a in answer_acts:
        letter = _norm_letter(a.get("target"))
        if opt_keys and letter and letter not in opt_keys:
            warnings.append(f"answer marks option {letter!r}, which OCR did not find "
                            f"among the options {sorted(opt_keys)}.")

    # ── Type ↔ action-set agreement (mostly WARN) ────────────────────────────
    if qtype:
        if qtype == "statement_count":
            nv = sum(1 for a in actions if a.get("action") in VERDICT_ACTIONS)
            if nv < 3:
                warnings.append(f"statement_count but only {nv} verdict_mark(s) — each "
                                "statement should get a ✓/✗.")
        if qtype == "assertion_reason":
            nv = sum(1 for a in actions if a.get("action") in VERDICT_ACTIONS)
            if nv < 2:
                warnings.append(f"assertion_reason but only {nv} verdict_mark(s) — "
                                "expected one for A and one for R.")
            if "cross_out_word" in kind_set:
                warnings.append("assertion_reason should NOT cross out options — the "
                                "answer follows from the two verdicts.")
        if qtype == "matching" and "match_pair" not in kind_set:
            warnings.append("matching question but no match_pair actions.")
        if qtype == "sequence" and "draw_diagram" not in kind_set:
            warnings.append("sequence question but no draw_diagram giving the order.")
        if qtype == "flowchart_fill":
            if "fill_placeholder" not in kind_set:
                warnings.append("flowchart_fill but no fill_placeholder actions.")
            if "draw_diagram" in kind_set:
                warnings.append("flowchart_fill should annotate the printed figure, "
                                "not draw_diagram a new one.")
        if qtype == "numerical" and "write_step" not in kind_set:
            warnings.append("numerical question but no write_step solution lines.")
        if qtype == "diagram_label" and "draw_diagram" in kind_set:
            warnings.append("diagram_label should annotate the existing figure, not "
                            "draw_diagram a new one.")
        if qtype == "mcq" and (kind_set & VERDICT_ACTIONS):
            warnings.append("mcq carries verdict_mark (✓/✗) actions — those belong to "
                            "statement/assertion questions and may mis-place.")

    # ── Board-clarity quality (WARN) ─────────────────────────────────────────
    notes = [a for a in actions if a.get("action") in NOTE_ACTIONS]
    if len(notes) > MAX_NOTES:
        warnings.append(f"{len(notes)} notes (write_note+annotate_word) — over the "
                        f"max of {MAX_NOTES}; the board will look cluttered.")
    for a in notes:
        t = str(a.get("text") or "").strip()
        if len(t) > NOTE_CHAR_LIMIT or len(t.split()) > NOTE_WORD_LIMIT:
            warnings.append(f"note is too long to read on the board ({len(t)} chars): "
                            f"{t[:60]!r}… — notes should be a few-word anchor.")

    diagrams = [a for a in actions if a.get("action") == "draw_diagram"]
    if len(diagrams) > MAX_DIAGRAMS:
        warnings.append(f"{len(diagrams)} diagrams — at most {MAX_DIAGRAMS} keep the "
                        "board legible.")
    for a in diagrams:
        for kind, label in _diagram_labels(a):
            if len(label.strip()) > NODE_LABEL_CHAR_LIMIT:
                warnings.append(f"diagram {kind} label is too long for a board "
                                f"({len(label)} chars): {label[:40]!r}… — prefer a "
                                "terse label.")

    return errors, warnings


def load_question_type(annotations_path):
    """Read the question_type persisted next to an annotations file (or None)."""
    meta = os.path.splitext(annotations_path)[0] + ".meta.json"
    try:
        with open(meta, encoding="utf-8") as f:
            return (json.load(f) or {}).get("question_type")
    except Exception:
        return None


def gate(annotations_path, *, option_positions=None, question_text=None,
         strict=False):
    """Run the gate on a saved annotations file.

    Prints errors/warnings. Returns True if the render may proceed, False if it
    must be blocked (any ERROR, or any WARNING when strict=True).
    """
    try:
        with open(annotations_path, encoding="utf-8") as f:
            anns = json.load(f)
    except Exception as e:
        print(f"  [validate] could not read {annotations_path}: {e}")
        return False
    qtype = load_question_type(annotations_path)
    errors, warnings = validate_annotation_set(
        anns, question_type=qtype, option_positions=option_positions,
        question_text=question_text)
    for w in warnings:
        print(f"  [validate][WARN] {w}")
    for e in errors:
        print(f"  [validate][ERROR] {e}")
    if errors:
        return False
    if strict and warnings:
        print("  [validate] strict mode: warnings treated as failure.")
        return False
    if not errors and not warnings:
        print(f"  [validate] OK ({len(anns)} annotations"
              f"{', type=' + qtype if qtype else ''}).")
    return True


def main(argv):
    if not argv:
        print("usage: validate_annotations.py <annotations.json> [--type T] [--strict]")
        return 2
    path = argv[0]
    strict = "--strict" in argv
    forced_type = None
    if "--type" in argv:
        i = argv.index("--type")
        if i + 1 < len(argv):
            forced_type = argv[i + 1]
    try:
        with open(path, encoding="utf-8") as f:
            anns = json.load(f)
    except Exception as e:
        print(f"could not read {path}: {e}")
        return 2
    qtype = forced_type or load_question_type(path)
    errors, warnings = validate_annotation_set(anns, question_type=qtype)
    for w in warnings:
        print(f"[WARN]  {w}")
    for e in errors:
        print(f"[ERROR] {e}")
    if not errors and not warnings:
        print(f"OK ({len(anns)} annotations{', type=' + qtype if qtype else ''}).")
    return 1 if errors or (strict and warnings) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
