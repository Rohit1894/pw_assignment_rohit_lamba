"""Single source of truth for the annotation action vocabulary.

The pipeline historically accepted several names for the same action (``tick_answer``
vs ``mark_answer``, ``write_equation``/``write_text`` vs ``write_step``,
``circle_existing`` vs ``circle_word``). Carrying duplicate names everywhere invites
edge bugs — a check that lists one name but not its twin silently misbehaves.

This module is the ONE place legacy names are declared and mapped to a canonical name.
Annotations are normalised to canonical names at ingestion (Gemini save + render load),
so downstream code — renderer and validator — reasons about a single clean schema.

Every alias below is RENDER-IDENTICAL to its canonical name: each pair/group shares the
exact same branch in ``render_video._build_schedule`` (circle group → one branch,
ANSWER_ACTIONS → one branch, WRITE_ACTIONS → one branch), so rewriting an alias to its
canonical name never changes how the video looks.
"""

# canonical name -> legacy aliases that mean exactly the same thing
ALIAS_GROUPS = {
    "mark_answer": ("tick_answer",),
    "circle_word": ("circle_existing",),
    "write_step":  ("write_equation", "write_text"),
}

# flat alias -> canonical lookup
ALIAS_TO_CANONICAL = {alias: canon
                      for canon, aliases in ALIAS_GROUPS.items()
                      for alias in aliases}

# The clean schema the rest of the system should think in terms of.
CANONICAL_ACTIONS = {
    "underline_existing", "circle_word", "cross_out_word", "verdict_mark",
    "match_pair", "annotate_word", "write_note", "draw_diagram", "mark_answer",
    "fill_placeholder", "write_step", "draw_arrow",
}

# Everything accepted on INPUT (canonical names + the legacy aliases).
ACCEPTED_ACTIONS = CANONICAL_ACTIONS | set(ALIAS_TO_CANONICAL)


def canonical_action(name):
    """Return the canonical name for an action (identity if already canonical)."""
    return ALIAS_TO_CANONICAL.get(name, name)


def normalize_actions(annotations):
    """Rewrite every item's ``action`` field to its canonical name, in place.

    Returns the same list for convenience. Safe on non-list / malformed input.
    """
    if isinstance(annotations, list):
        for a in annotations:
            if isinstance(a, dict):
                name = a.get("action")
                if name in ALIAS_TO_CANONICAL:
                    a["action"] = ALIAS_TO_CANONICAL[name]
    return annotations
