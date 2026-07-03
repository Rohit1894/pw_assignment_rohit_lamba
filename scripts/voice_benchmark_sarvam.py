#!/usr/bin/env python3
"""Sarvam voice benchmark: generate audio samples for each candidate speaker
and produce a CSV template + markdown report for manual scoring.

Usage:
    python scripts/voice_benchmark_sarvam.py --language hinglish --pace 0.92
    python scripts/voice_benchmark_sarvam.py --speakers shubh,ritu,priya

Outputs:
    output/voice_tests/<speaker>/sample_01.wav  ... sample_N.wav
    output/voice_tests/voice_score_template.csv
    output/voice_tests/report.md
"""

import argparse
import csv
import json
import os
import sys

# Add scripts/ to path so we can import from generate_audio_sarvam
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_TEST_SENTENCES = [
    ("kinematics", "In kinematics, हम motion को study करते हैं without considering forces."),
    ("velocity",   "Velocity बताती है कि object किस direction में और kitni speed से move कर रहा है।"),
    ("acceleration", "Acceleration means velocity में change होना per second."),
    ("displacement", "Displacement वह shortest path है जो object ने start से end तक cover किया।"),
    ("formula",    "Ab hum formula लिखते हैं: v square equals u square plus two a s."),
    ("equation",   "यह equation हमें final velocity calculate करने में help करती है।"),
    ("zero",       "Maximum height पर final velocity zero हो जाती है।"),
    ("max_height", "Maximum height calculate करने के लिए हम यह equation use करते हैं।"),
    ("units",      "Height का answer twenty meter per second square में आता है।"),
    ("v_square",   "v square equals u square plus two a s — यह kinematics की basic equation है।"),
    ("zero_calc",  "zero square equals twenty square minus two into ten into H."),
    ("photosynthesis", "Photosynthesis में plants sunlight को use करके food बनाते हैं।"),
    ("respiration", "Respiration एक process है जिसमें glucose break down होती है energy के लिए।"),
    ("mitochondria", "Mitochondria को cell का powerhouse कहते हैं।"),
    ("chromosome", "Chromosome में genetic information stored होती है।"),
    ("molarity",   "Molarity means number of moles of solute per litre of solution."),
    ("valency",    "Valency किसी element की combining capacity होती है।"),
    ("quadratic",  "यह एक quadratic equation है जिसे हम factorisation से solve करेंगे।"),
    ("differentiation", "Differentiation में हम function का rate of change find करते हैं।"),
    ("integration", "Integration differentiation का reverse process है।"),
    ("trigonometry", "Trigonometry में sine, cosine, और tangent ratios use होते हैं।"),
    ("probability", "Probability बताती है कि कोई event होने के कितने chances हैं।"),
]

_VOICES_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "sarvam_voices.json")


def _load_candidates() -> list:
    try:
        with open(_VOICES_JSON, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("candidates", [])
    except Exception:
        return ["shubh", "ritu", "priya", "neha", "rahul"]


def _generate_samples(speaker, out_dir, pace, model, language_code):
    from generate_audio_sarvam import generate_sarvam_audio_segment
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, (label, text) in enumerate(_TEST_SENTENCES, 1):
        out_path = os.path.join(out_dir, f"sample_{i:02d}_{label}.wav")
        print(f"    [{speaker}] sample {i:02d} ({label})")
        try:
            generate_sarvam_audio_segment(
                text=text,
                output_path=out_path,
                target_language_code=language_code,
                speaker=speaker,
                model=model,
                pace=pace,
            )
            paths.append((label, out_path, "ok"))
        except Exception as e:
            print(f"      WARN: failed for {speaker} sample {i}: {e}")
            paths.append((label, out_path, f"error: {str(e)[:60]}"))
    return paths


def _write_csv(speakers, csv_path):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["speaker", "pronunciation_score", "teacher_feel_score",
                    "clarity_score", "math_score", "overall_score", "notes"])
        for sp in speakers:
            w.writerow([sp, "", "", "", "", "", ""])
    print(f"  CSV template -> {csv_path}")


def _write_report(speakers, results, out_dir):
    report_path = os.path.join(out_dir, "report.md")
    lines = ["# Sarvam Voice Benchmark Report\n",
             f"Generated for {len(speakers)} speakers × {len(_TEST_SENTENCES)} sentences.\n",
             "## Speakers Tested\n"]
    for sp in speakers:
        res = results.get(sp, [])
        ok = sum(1 for _, _, s in res if s == "ok")
        total = len(res)
        lines.append(f"- **{sp}**: {ok}/{total} samples generated")
        err = [(l, s) for l, _, s in res if s != "ok"]
        if err:
            for label, msg in err[:3]:
                lines.append(f"  - WARN [{label}]: {msg}")
        lines.append("")
    lines += [
        "\n## Scoring Guide\n",
        "Listen to each speaker's samples and fill in `voice_score_template.csv`:\n",
        "| Column | What to score (1–5) |",
        "|--------|---------------------|",
        "| pronunciation_score | Are technical words (kinematics, velocity, etc.) pronounced correctly? |",
        "| teacher_feel_score  | Does the voice sound like a warm, helpful teacher? |",
        "| clarity_score       | Is every word clearly audible? |",
        "| math_score          | Are math formulas read naturally? |",
        "| overall_score       | Overall impression for educational videos |",
        "\n## Sample Files\n",
    ]
    for sp in speakers:
        sp_dir = os.path.join(out_dir, sp)
        lines.append(f"### {sp}\n")
        res = results.get(sp, [])
        for label, path, status in res:
            rel = os.path.relpath(path, out_dir) if os.path.exists(path) else path
            icon = "✓" if status == "ok" else "✗"
            lines.append(f"- {icon} [{label}]({rel.replace(chr(92), '/')})")
        lines.append("")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report -> {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Sarvam voice benchmark")
    parser.add_argument("--language", default="hinglish")
    parser.add_argument("--pace", type=float, default=0.92)
    parser.add_argument("--model", default="bulbul:v3")
    parser.add_argument("--language-code", default="hi-IN")
    parser.add_argument("--output-dir", default="output/voice_tests")
    parser.add_argument("--speakers",
                        help="Comma-separated list; default: all candidates")
    args = parser.parse_args()

    speakers = (args.speakers.split(",") if args.speakers
                else _load_candidates())
    speakers = [s.strip().lower() for s in speakers if s.strip()]
    if not speakers:
        print("No speakers specified.")
        sys.exit(1)

    print(f"Benchmarking {len(speakers)} speakers: {', '.join(speakers)}")
    results = {}
    for sp in speakers:
        print(f"\n  Speaker: {sp}")
        sp_dir = os.path.join(args.output_dir, sp)
        results[sp] = _generate_samples(sp, sp_dir, args.pace,
                                         args.model, args.language_code)

    csv_path = os.path.join(args.output_dir, "voice_score_template.csv")
    _write_csv(speakers, csv_path)
    _write_report(speakers, results, args.output_dir)

    ok_counts = {sp: sum(1 for _, _, s in results[sp] if s == "ok")
                 for sp in speakers}
    best = max(ok_counts, key=ok_counts.get) if ok_counts else None
    print(f"\nDone. Listen and fill in {csv_path}")
    if best:
        print(f"Most successful speaker (fewest errors): {best}")


if __name__ == "__main__":
    main()
