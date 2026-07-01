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

import copy
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


def _seq_clock(raw):
    """
    Walk `raw` and re-time minute-rollover clock notation back to true seconds.

    Gemini narrates the first minute in plain seconds (0..59), then at the minute
    boundary switches to a clock reading written WITHOUT a real colon — either as a
    decimal `M.SS` (e.g. 1.01 = 1:01 = 61 s, so the raw value DROPS below the
    previous one) or as a concatenated `MMSS` integer (e.g. 107 = 1:07 = 67 s, so
    the raw value JUMPS by ~+47). Either way the running clock only moves forward,
    so at each entry we take the smallest interpretation that does not go backwards.
    """
    out = []
    prev = -1e9
    for v in raw:
        cands = [v]                                  # plain seconds
        intp = int(v)
        sec = round((v - intp) * 100)
        if sec < 60:                                 # M.SS decimal (e.g. 1.01 -> 61)
            cands.append(intp * 60 + (v - intp) * 100)
        if v >= 100:                                 # MMSS concat (e.g. 107 -> 67)
            m = int(round(v)) // 100
            rem = v - m * 100
            if 0 <= rem < 60:
                cands.append(m * 60 + rem)
        fwd = [c for c in cands if c >= prev - 0.5]
        prev = min(fwd) if fwd else min(cands, key=lambda c: abs(c - prev))
        out.append(prev)
    return out


def _decode_clock(raw, duration):
    """
    If a transcript's timestamps are minute-clock notation rather than true
    seconds, decode them; otherwise return None so the caller keeps the raw values.

    Gemini emits one of three encodings (see _seq_clock): plain seconds, uniform
    `M.SS` decimal, or plain-seconds-then-clock. A proportional rescale (the old
    repair) turns every minute rollover into a phantom ~30 s gap, which is exactly
    what left mid-solution stretches blank. We instead pick, by global fit:
      - leave PLAIN seconds untouched when they are already monotonic and span the
        audio (healthy transcripts are byte-identical — the decoder never engages);
      - uniform `M.SS` decode for clips narrated entirely in minute notation;
      - sequential rollover repair for the plain-then-clock mix.
    The winning candidate is the non-decreasing one whose end is closest to the
    audio length, so the duration arbitrates between ambiguous interpretations.
    """
    if not raw:
        return None
    nondec = all(raw[i + 1] >= raw[i] - 0.5 for i in range(len(raw) - 1))
    fits = (duration is None) or (0.5 * duration <= max(raw) <= 1.15 * duration)
    if nondec and fits:
        return None                                  # plain seconds, healthy

    cands = [list(raw)]                              # plain (fallback)
    if all((v - int(v)) * 100 < 60.0001 for v in raw):
        cands.append([int(v) * 60 + (v - int(v)) * 100 for v in raw])   # uniform M.SS
    cands.append(_seq_clock(raw))

    def score(d):
        mono = all(d[i + 1] >= d[i] - 0.5 for i in range(len(d) - 1))
        if duration:
            return (mono, -abs(max(d) - 0.97 * duration))
        return (mono, max(d))
    best = max(cands, key=score)
    # Only override the raw values if decoding actually improved the duration fit.
    if duration and not fits:
        if abs(max(best) - 0.97 * duration) >= abs(max(raw) - 0.97 * duration):
            return None
    return best if best != list(raw) else None


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

    # Layer-3: minute-clock decode. Gemini writes the first minute in seconds then
    # rolls into M.SS / MMSS notation; decode those to true seconds BEFORE the gap
    # and scale repairs below, so a rollover no longer becomes a phantom ~30s gap.
    decoded = _decode_clock(raw, duration)
    if decoded is not None:
        raw = decoded

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


def _step_times(anns):
    """Sorted times of the derivation (write_step) actions only — the part whose
    pacing the viewer reads as 'the solution being worked out'. Reading-phase
    underline clustering is benign, so quality checks ignore it."""
    return sorted(float(a.get("time", 0.0)) for a in anns
                  if a.get("action") == "write_step")


def _collapse_run(times, thresh=1.5):
    """Longest run of consecutive steps separated by less than `thresh` seconds —
    i.e. the length of the worst 'burst' where lines pile out near-simultaneously."""
    longest = cur = 0
    for i in range(1, len(times)):
        if times[i] - times[i - 1] < thresh:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return longest


def _alignment_score(anns, stats, duration):
    """
    Heuristic quality of an aligned timeline (higher = better): coverage (how many
    actions actually matched the transcript) + how well the derivation spans the
    audio, minus a burst penalty. Used by `align_best` to rank candidate transcripts.
    """
    n = stats.get("n") or len(anns) or 1
    coverage = stats.get("matched", 0) / n
    st = _step_times(anns)
    span = (st[-1] - st[0]) if len(st) >= 2 else 0.0
    spread = (span / duration) if duration else 0.0
    return coverage * 1.0 + min(spread, 1.0) * 0.6 - _collapse_run(st) * 0.08


def _is_bad_alignment(anns, stats, duration):
    """
    True only when the default alignment is GENUINELY broken — too few actions
    matched the transcript, or the derivation collapses into a real burst. A fully
    matched, well-spread timeline is NOT bad, so `align_best` returns it untouched
    (byte-identical) and never reaches for the alternative. This is the gate that
    keeps already-good videos exactly as they are while rescuing the broken ones.
    """
    n = stats.get("n") or len(anns) or 1
    coverage = stats.get("matched", 0) / n
    return coverage < 0.65 or _collapse_run(_step_times(anns)) >= 3


def align_best(annotations, sources, min_ratio=0.5, duration=None):
    """
    Align against candidate word-sources, keeping the DEFAULT unless it is broken.

    `sources` is an ordered list of (name, words); the FIRST is the default. If the
    default alignment is healthy (`_is_bad_alignment` False) it is returned as-is —
    so well-synced videos are byte-identical and the alternative is never even tried.
    Only when the default is broken (e.g. an English SLIDE with HINDI narration whose
    Hindi spoken_cues can't match an English Whisper transcript) do we align the
    alternatives and switch to one that is both healthy AND clearly better-scoring.
    No language guess is needed; the failure is detected from the alignment itself.
    """
    if not sources:
        return annotations
    name0, words0 = sources[0]
    base, stats0 = align_annotations(copy.deepcopy(annotations), words0,
                                     min_ratio=min_ratio, duration=duration,
                                     return_stats=True)
    if not _is_bad_alignment(base, stats0, duration):
        return base
    base_score = _alignment_score(base, stats0, duration)
    best_name, best_anns, best_score = name0, base, base_score
    for name, words in sources[1:]:
        if not words:
            continue
        anns, stats = align_annotations(copy.deepcopy(annotations), words,
                                        min_ratio=min_ratio, duration=duration,
                                        return_stats=True)
        if _is_bad_alignment(anns, stats, duration):
            continue
        sc = _alignment_score(anns, stats, duration)
        if sc > best_score + 0.06:
            best_name, best_anns, best_score = name, anns, sc
    if best_name != name0:
        print(f"  Audio sync: default '{name0}' transcript aligned poorly "
              f"(score {base_score:.2f}); using '{best_name}' instead "
              f"(score {best_score:.2f})")
    else:
        print(f"  Audio sync: default '{name0}' alignment is weak "
              f"(score {base_score:.2f}) but no better transcript available; keeping it")
    return best_anns


def align_annotations(annotations, words, min_ratio=0.5, duration=None, return_stats=False):
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
    def _result(anns_out, matched_n, span_val):
        if return_stats:
            return anns_out, {"matched": matched_n, "n": len(anns_out), "span": span_val}
        return anns_out

    if not words or not annotations:
        return _result(annotations, 0, 0.0)

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
        return _result(anns, len(anchors), span)

    # Unmatched actions: spread each RUN of consecutive unmatched actions EVENLY
    # across the gap between its bounding matched anchors. The old code filled each
    # hole toward its nearest neighbour, so a whole run of holes converged onto the
    # next anchor — collapsing an entire derivation into a 1-2 s burst that the
    # minimum-gap pass then fanned out flatly (the "writes everything at once, then
    # the board sits dead" artifact). Even spreading paces the writing between the
    # two spoken moments that bracket it. Fully-matched timelines have no holes, so
    # they are byte-identical.
    i = 0
    while i < n:
        if matched[i] is not None:
            i += 1
            continue
        j = i
        while j < n and matched[j] is None:
            j += 1
        prev = matched[i - 1] if i - 1 >= 0 else None
        nxt = matched[j] if j < n else None
        run = j - i
        if prev is not None and nxt is not None:
            for k in range(run):
                matched[i + k] = prev + (nxt - prev) * (k + 1) / (run + 1)
        elif prev is not None:
            for k in range(run):
                matched[i + k] = prev + 1.0 * (k + 1)
        elif nxt is not None:
            base = max(0.0, nxt - 1.0 * run)
            for k in range(run):
                matched[i + k] = base + 1.0 * k
        else:
            for k in range(run):
                matched[i + k] = 0.0
        i = j

    for i, ann in enumerate(anns):
        ann["time"] = round(float(matched[i]), 2)
    anns.sort(key=lambda a: a["time"])
    print(f"  Aligned timeline to spoken words: {len(anchors)}/{n} actions matched, "
          f"span {span:.0f}s")
    return _result(anns, len(anchors), span)


if __name__ == "__main__":
    import sys
    transcript = sys.argv[1] if len(sys.argv) > 1 else "output/transcript.json"
    ann_path = sys.argv[2] if len(sys.argv) > 2 else "output/annotations.json"
    words = load_words(transcript)
    with open(ann_path, encoding="utf-8") as f:
        anns = json.load(f)
    anns = align_annotations(anns, words)
    print(json.dumps(anns, indent=2, ensure_ascii=False))
