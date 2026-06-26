#!/usr/bin/env python3
"""
Sync annotation actions to the audio using Whisper word-level timestamps.

Gemini decides WHAT to annotate (and, ideally, the short phrase the teacher
SAYS at that moment — the action's `spoken_cue`). Whisper decides WHEN: this
module locates each action's cue in the spoken-word timeline and re-times the
action to the exact second those words are spoken. This removes the "writes
before the teacher speaks / screen shows something else" drift, because every
on-screen annotation is anchored to the matching audio.
"""

import json
import re
from difflib import SequenceMatcher


def _norm(s):
    """Lowercase, drop punctuation/brackets, collapse whitespace."""
    s = (s or "").lower()
    s = re.sub(r"[()\[\]{}.,:;!?\"'`/\\|_\-–—=+*]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_words(transcript_path):
    """
    Flatten a Whisper transcript JSON into [(norm_token, start, end)].

    Prefers word-level timestamps; falls back to segment-level if a segment has
    no per-word timing. Empty/punctuation-only tokens are dropped.
    """
    with open(transcript_path, encoding="utf-8") as f:
        data = json.load(f)
    words = []
    for seg in data.get("segments", []):
        ws = seg.get("words") or []
        if ws:
            for w in ws:
                tok = _norm(w.get("word"))
                if tok:
                    words.append((tok, float(w.get("start", 0.0)),
                                  float(w.get("end", 0.0))))
        else:
            tok = _norm(seg.get("text"))
            if tok:
                # Spread the segment text as one block at its start time.
                words.append((tok, float(seg.get("start", 0.0)),
                              float(seg.get("end", 0.0))))
    return words


def _sanitise_entry_times(entries, duration=None):
    """
    Repair a Gemini transcript's timestamps.

    Gemini's TEXT and ORDER are reliable, but its absolute timestamps often break
    (e.g. correct seconds for a while, then a reset to tiny values when it slips
    into minute notation). Trusting those raw numbers would cluster half the
    timeline at t≈0. So we KEEP the entry order and rebuild a strictly increasing
    time grid: values that don't increase (or exceed the audio) are dropped and
    re-interpolated by position between the surrounding trustworthy anchors, with
    the trailing run spread out to the audio's end.
    """
    n = len(entries)
    raw = [float(e.get("t", 0.0)) for e in entries]

    # De-clump phantom gaps: Gemini sometimes inserts a huge FAKE pause between two
    # consecutive sentences of continuous speech (e.g. a 45s jump in a 118s clip).
    # That smears one entry's words across most of the timeline, so every annotation
    # whose cue falls there is dumped at the very end while the middle sits empty.
    # Detect inter-entry gaps that are gross outliers vs the median and shrink them,
    # then re-accumulate, keeping the spoken timeline continuous.
    if len(raw) >= 4:
        gaps = [raw[i + 1] - raw[i] for i in range(len(raw) - 1)]
        pos = sorted(g for g in gaps if g > 0)
        if pos:
            median = pos[len(pos) // 2]
            cap_gap = max(median * 3.0, 10.0)
            rebuilt = [raw[0]]
            for g in gaps:
                rebuilt.append(rebuilt[-1] + min(max(g, 0.0), cap_gap))
            raw = rebuilt

    # Scale repair: Gemini frequently emits the transcript timestamps in the wrong
    # scale — COMPRESSED (top out at ~1.3 for a 95s clip) or OVER-LONG (running past
    # the audio's real end). The relative spacing is still informative, so when the
    # max timestamp is far below OR beyond the audio length, rescale the whole set
    # proportionally to span the real duration.
    if duration and raw:
        mx = max(raw)
        if mx > 0 and (mx < 0.5 * duration or mx > duration):
            scale = (duration * 0.97) / mx
            raw = [t * scale for t in raw]

    cap = duration if duration else (max(raw) if raw else 0.0)

    valid = [None] * n          # monotonic anchors we trust
    last = -1.0
    for i, t in enumerate(raw):
        if t > last + 0.05 and t <= cap + 0.5:
            valid[i] = t
            last = t

    # Interpolate the holes. Leading hole -> from 0; interior -> between anchors;
    # trailing hole -> spread from the last anchor to the audio end.
    anchors = [i for i, v in enumerate(valid) if v is not None]
    if not anchors:
        end = cap if cap else float(n)
        return [end * (i + 1) / (n + 1) for i in range(n)]
    out = list(valid)
    first_i = anchors[0]
    for i in range(first_i):
        out[i] = valid[first_i] * (i + 1) / (first_i + 1)
    for a, b in zip(anchors, anchors[1:]):
        if b - a > 1:
            for k in range(1, b - a):
                out[a + k] = valid[a] + (valid[b] - valid[a]) * k / (b - a)
    last_i = anchors[-1]
    end = cap if cap and cap > valid[last_i] else valid[last_i] + 1.5 * (n - last_i)
    for k, i in enumerate(range(last_i + 1, n), start=1):
        out[i] = valid[last_i] + (end - valid[last_i]) * k / (n - last_i)
    return out


def load_gemini_words(gtrans_path, duration=None):
    """
    Flatten a Gemini timestamped transcript ([{"t": sec, "text": "..."}]) into
    [(norm_token, start, end)], distributing each entry's tokens across the time
    until the next entry. Timestamps are sanitised first (see above), because
    Gemini's raw timestamp scale is unreliable even though its text/order is good.
    Gemini transcribes Hindi far better than Whisper on noisy lecture audio, so
    this is the preferred sync source when available.
    """
    with open(gtrans_path, encoding="utf-8") as f:
        entries = json.load(f)
    entries = [e for e in entries if isinstance(e, dict) and "text" in e]
    # Keep the model's emitted ORDER (do not sort by the unreliable raw t).
    times = _sanitise_entry_times(entries, duration)
    words = []
    for k, e in enumerate(entries):
        t0 = times[k]
        t1 = times[k + 1] if k + 1 < len(entries) else t0 + 4.0
        toks = _norm(e.get("text")).split()
        if not toks:
            continue
        step = (t1 - t0) / len(toks) if t1 > t0 else 0.2
        for j, tok in enumerate(toks):
            s = t0 + j * step
            words.append((tok, s, s + step))
    return words


def _cues_for(ann):
    """
    Candidate phrases to search for, best-first.

    `spoken_cue` (what the teacher actually says) is ideal when the transcript is
    accurate, but Hindi ASR is rough on conversational speech while it still gets
    the clean printed TERMS right — so we also try `target` and friends and keep
    whichever matches the transcript best. The written meaning (`text`) is last
    because the teacher speaks the SLIDE word, not the meaning we write for it.
    """
    cues = []
    for key in ("spoken_cue", "target", "start_target", "end_target",
                "from_target", "to_target", "label", "text"):
        v = _norm(ann.get(key))
        if v and v not in cues:
            cues.append(v)
    return cues


def _best_match(tokens, words, start_idx, min_ratio):
    """
    Best fuzzy window match for `tokens` in `words` at or after start_idx.

    Slides windows of a few sizes around len(tokens) and scores each by string
    similarity (Whisper is not letter-perfect, so exact match is too strict).
    Returns (ratio, start_time, word_index) or (0, None, None).
    """
    n = max(1, len(tokens))
    target = " ".join(tokens)
    sizes = sorted({max(1, n - 1), n, n + 1, n + 2})
    best = (0.0, None, None)
    for i in range(start_idx, len(words)):
        for wsize in sizes:
            j = i + wsize
            if j > len(words):
                continue
            cand = " ".join(w[0] for w in words[i:j])
            r = SequenceMatcher(None, target, cand).ratio()
            if r > best[0]:
                best = (r, words[i][1], i)
        # Early stop scan window: don't search the entire rest for every action
        # once we are far past a good hit (keeps it fast and order-preserving).
        if best[0] >= 0.95:
            break
    if best[0] >= min_ratio:
        return best
    return (0.0, None, None)


def align_annotations(annotations, words, min_ratio=0.5, duration=None):
    """
    Re-time `annotations` to the spoken-word timeline `words`.

    The transcript order is trustworthy but the model's ACTION order/times often
    are not (it may emit the answer before the fills, etc.). So we ignore the
    input order: each action is matched to where its cue is spoken, and the whole
    list is re-sorted by that matched time. The transcript becomes the single
    source of truth for both timing and ordering.

      1. For each action, take its GLOBAL best fuzzy match (any position) over all
         of its candidate cues — preferring the highest-similarity entry, which
         naturally favours the right occurrence of a repeated phrase (the full
         cue scores best where the most of it is actually said).
      2. SANITY GATE: if too few actions match, or they cover too little of the
         audio, the transcript is unreliable — keep the model timing instead.
      3. Anchored actions take their matched time; unmatched ones are slotted
         next to their original neighbours; everything is then re-sorted.
    Returns the re-timed list.
    """
    if not words or not annotations:
        return annotations

    anns = sorted(annotations, key=lambda a: float(a.get("time", 0.0)))
    n = len(anns)

    matched = [None] * n
    last_idx = 0
    for i, ann in enumerate(anns):
        # Explanation actions (notes, verdicts, the diagram, the answer) carry a
        # spoken_cue — the teacher's exact words at that moment. Match it GLOBALLY:
        # the cue is specific enough to find the right spot, so a short printed
        # `target` that ALSO appears in the reading phase can't override it, and a
        # forward cursor that has already run past the cue's position can't orphan it
        # (which is what dumped every explanation action at the clustered end-times).
        sc = _norm(ann.get("spoken_cue"))
        if len(sc.split()) >= 3:
            ratio, st, idx = _best_match(sc.split(), words, 0, max(min_ratio, 0.55))
            if st is not None:
                matched[i] = st
                continue
        # Reading-phase / cue-less actions (underlines): match the printed target with
        # a forward cursor so repeated short terms resolve in reading order.
        best = (0.0, None, None)
        search_start = max(0, last_idx - 8)
        for cue in _cues_for(ann):
            ratio, st, idx = _best_match(cue.split(), words, search_start, min_ratio)
            if st is not None and ratio > best[0]:
                best = (ratio, st, idx)
        if best[1] is not None:
            matched[i] = best[1]
            last_idx = best[2]

    anchors = [t for t in matched if t is not None]
    span = (max(anchors) - min(anchors)) if len(anchors) >= 2 else 0.0
    enough = len(anchors) >= max(3, int(0.25 * n))
    spread_ok = (duration is None) or (span >= 0.30 * duration)
    if not (enough and spread_ok):
        print(f"  Audio sync: transcript too sparse to align reliably "
              f"({len(anchors)} anchors, span {span:.0f}s); keeping model timing")
        return anns

    # Unmatched actions: place each beside its nearest matched neighbour in the
    # ORIGINAL list (adjacent actions are usually related), nudged by a small gap.
    for i in range(n):
        if matched[i] is not None:
            continue
        prev = next((matched[j] for j in range(i - 1, -1, -1) if matched[j] is not None), None)
        nxt = next((matched[j] for j in range(i + 1, n) if matched[j] is not None), None)
        if prev is not None and nxt is not None:
            matched[i] = (prev + nxt) / 2
        elif prev is not None:
            matched[i] = prev + 1.0
        elif nxt is not None:
            matched[i] = max(0.0, nxt - 1.0)
        else:
            matched[i] = 0.0

    for i, ann in enumerate(anns):
        ann["time"] = round(float(matched[i]), 2)
    anns.sort(key=lambda a: a["time"])
    print(f"  Aligned timeline to spoken words: {len(anchors)}/{n} actions matched, "
          f"span {span:.0f}s")
    return anns


if __name__ == "__main__":
    import sys
    transcript = sys.argv[1] if len(sys.argv) > 1 else "output/transcript.json"
    ann_path = sys.argv[2] if len(sys.argv) > 2 else "output/annotations.json"
    words = load_words(transcript)
    with open(ann_path, encoding="utf-8") as f:
        anns = json.load(f)
    anns = align_annotations(anns, words)
    print(json.dumps(anns, indent=2, ensure_ascii=False))
