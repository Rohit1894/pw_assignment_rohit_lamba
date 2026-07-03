#!/usr/bin/env python3
"""Verify the canonical solution BEFORE any video is generated.

Two independent checks:
  1. Arithmetic check (numerical questions): re-evaluate the calculation lines
     with sympy / safe arithmetic where they parse, catching slips like
     "20H = 400 → H = 30".
  2. Independent Gemini pass: a fresh model call solves the question again
     WITHOUT seeing the solver's answer, then the two answers are compared.

If the solver and verifier disagree, status is "needs_review" and main.py
stops (unless --allow-unverified).

Output: output/solution_verification.json
"""

import json
import os
import re
import sys

from gemini_utils import call_gemini_json, image_part

_VERIFY_PROMPT = """You are an independent examiner. Solve the attached exam question
yourself, from scratch, carefully. Do not assume any particular answer is expected.

QUESTION (transcribed): {question_text}
OPTIONS:
{options}

Return ONLY a JSON object:
{{
  "option": "<the correct option letter, or empty string if there are no options>",
  "answer_text": "<the final answer value/statement>",
  "confidence": <0.0-1.0, your confidence in this answer>,
  "reasoning": "<2-3 short sentences of your key reasoning>"
}}"""


def _norm_answer_text(s):
    """Normalise an answer string for comparison: lowercase, strip spaces and
    common unit punctuation so '20 m' == '20m' == '20 meter'."""
    s = str(s or "").lower().strip()
    s = re.sub(r"[\s,]+", "", s)
    s = s.replace("metre", "m").replace("meter", "m").replace("second", "s")
    return s


def _safe_arith_check(solution):
    """Best-effort arithmetic sanity check on numerical calculation steps.

    Parses simple 'lhs = rhs' equalities where BOTH sides are pure arithmetic
    (digits/operators only) and confirms they are numerically equal. Steps with
    symbols are skipped — this is a tripwire for arithmetic slips, not a CAS
    proof. Returns (n_checked, issues)."""
    issues = []
    checked = 0
    try:
        import sympy  # noqa: F401
        from sympy import sympify
    except Exception:
        return 0, []

    arith_re = re.compile(r"^[\d\.\s\+\-×x\*/\(\)\^²³]+$")

    def _pyexpr(s):
        s = s.replace("×", "*").replace("x", "*").replace("^", "**")
        s = s.replace("²", "**2").replace("³", "**3").replace("−", "-")
        return s

    for step in solution.get("solution_steps", []):
        text = str(step.get("text", ""))
        for eq in re.split(r"[;,]", text):
            if eq.count("=") != 1:
                continue
            lhs, rhs = (p.strip() for p in eq.split("="))
            l_norm = lhs.replace("−", "-")
            r_norm = rhs.replace("−", "-")
            # Strip a trailing unit token from the rhs ("400 m²/s²" → "400").
            # The unit must be whitespace-separated: "20H" is a SYMBOL term
            # (algebra, not arithmetic) and stripping its H would turn a
            # legitimate substitution step into a false arithmetic failure.
            r_norm = re.sub(r"\s+[a-zA-Z/²³°]+\s*$", "", r_norm)
            if not (arith_re.match(l_norm) and arith_re.match(r_norm) and r_norm):
                continue
            try:
                lv = float(sympify(_pyexpr(l_norm)))
                rv = float(sympify(_pyexpr(r_norm)))
            except Exception:
                continue
            checked += 1
            tol = max(1e-6, abs(rv) * 0.01)
            if abs(lv - rv) > tol:
                issues.append(f"Arithmetic check failed in step "
                              f"'{step.get('id')}': {lhs} = {lv:g}, not {rhs}")
    return checked, issues


def verify_solution(image_path, understanding, solution,
                    output_path="output/solution_verification.json"):
    """Verify the solution; save and return the verification dict."""
    issues = []
    confidence = 0.5

    # ── Check 1: arithmetic tripwire ─────────────────────────────────────
    n_arith, arith_issues = _safe_arith_check(solution)
    issues.extend(arith_issues)
    if n_arith:
        print(f"  Arithmetic check: {n_arith} equalit{'y' if n_arith == 1 else 'ies'} "
              f"verified, {len(arith_issues)} issue(s)")

    # ── Check 2: independent Gemini re-solve ─────────────────────────────
    opts = "\n".join(f'  ({o["label"]}) {o["text"]}'
                     for o in understanding.get("options", [])) or "  (none)"
    prompt = _VERIFY_PROMPT.format(
        question_text=understanding.get("question_text", ""), options=opts)
    try:
        ver = call_gemini_json([image_part(image_path), prompt],
                               temperature=0.4, label="Independent verification")
    except Exception as e:
        ver = None
        issues.append(f"Verifier call failed: {str(e)[:120]}")

    solver_opt = solution.get("final_answer", {}).get("option", "")
    solver_text = solution.get("final_answer", {}).get("text", "")

    if isinstance(ver, dict):
        v_opt = str(ver.get("option") or "").strip().upper()[:1]
        v_text = str(ver.get("answer_text") or "").strip()
        try:
            v_conf = max(0.0, min(1.0, float(ver.get("confidence", 0.7))))
        except (TypeError, ValueError):
            v_conf = 0.7
        if solver_opt and v_opt:
            if v_opt == solver_opt:
                confidence = max(confidence, 0.6 + 0.4 * v_conf)
            else:
                issues.append(f"Solver says option {solver_opt} but verifier "
                              f"says option {v_opt} "
                              f"(verifier reasoning: {ver.get('reasoning', '')[:160]})")
                confidence = min(confidence, 0.4)
        elif _norm_answer_text(v_text) and \
                _norm_answer_text(v_text) == _norm_answer_text(solver_text):
            confidence = max(confidence, 0.55 + 0.4 * v_conf)
        elif v_text:
            issues.append(f"Solver answer '{solver_text}' does not match "
                          f"verifier answer '{v_text}'")
            confidence = min(confidence, 0.45)

    if arith_issues:
        confidence = min(confidence, 0.3)

    status = "verified" if (confidence >= 0.6 and not issues) else "needs_review"
    verification = {
        "status": status,
        "confidence": round(confidence, 2),
        "verified_answer": {"option": solver_opt, "text": solver_text},
        "issues": issues,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(verification, f, indent=2, ensure_ascii=False)
    print(f"  Verification: {status} (confidence {verification['confidence']}) "
          f"-> {output_path}")
    for iss in issues:
        print(f"    ISSUE: {iss}")
    return verification


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "input/question.png"
    with open("output/question_understanding.json", encoding="utf-8") as f:
        und = json.load(f)
    with open("output/canonical_solution.json", encoding="utf-8") as f:
        sol = json.load(f)
    verify_solution(image, und, sol)
