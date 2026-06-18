"""
safety/leak_filter.py — Answer-leak filter for the Verified Math Tutor.

Purpose
-------
While a student is still working on a problem ("gated" mode), prevent the
LLM's tutoring prose from revealing the correct answer literally.

How it works
------------
1.  The CAS solves the problem and supplies ``answers`` as a list of strings
    (e.g. ``["3", "-2"]``).
2.  This module replaces every literal occurrence of those answer strings —
    including trivial re-orderings (e.g. ``"3, 2"`` when answers are
    ``["2", "3"]``) — with the token ``[hidden while you work]``.

⚠️  KNOWN LIMITATION — LITERAL MATCHING ONLY
---------------------------------------------
This filter catches *literal* occurrences of the answer strings.  It does
**NOT** catch:

* Answers disguised in prose ("the solution is one more than two"),
* Answers embedded in equations the student hasn't simplified yet
  ("x = 6/2"),
* Answers revealed through strong hints ("try substituting the value that
  makes x − 3 equal to zero"),
* Answers in a different but equivalent form ("x = 6/2" when answer is 3).

These require semantic / algebraic understanding and are the responsibility
of the LLM prompt discipline and ``claim_cert.py``, not this filter.
Any production deployment should treat literal filtering as one layer of a
defence-in-depth stack, not a complete solution.
"""

from __future__ import annotations

import itertools
import re
from typing import Sequence


_HIDDEN = "[hidden while you work]"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _escape(s: str) -> str:
    """Regex-escape a literal string."""
    return re.escape(s.strip())


def _make_patterns(answers: Sequence[str]) -> list[re.Pattern[str]]:
    """
    Build a list of compiled regex patterns to match:

    1. Each individual answer string.
    2. Every permutation of the full answer set written as a
       comma-separated list (``"3, 2"`` and ``"2, 3"`` for answers
       ``["2", "3"]``).

    We use word boundaries (``\\b``) around numeric literals so that
    ``"3"`` does not match inside ``"13"`` or ``"30"``.
    """
    patterns: list[re.Pattern[str]] = []
    stripped = [a.strip() for a in answers if a.strip()]

    # 1. Individual answers
    for ans in stripped:
        esc = _escape(ans)
        # \b only works where the boundary is between a \w and a \W char.
        # Negative numbers like "-2" start with "-" which is \W, so the
        # leading \b would sit between two \W chars and never match.
        # Instead we use a lookbehind/lookahead that rejects digits on
        # either side, which correctly handles "-2" without eating "13".
        pat = rf"(?<!\d){esc}(?!\d)"
        patterns.append(re.compile(pat))

    # 2. Comma-separated permutations (trivial reorderings)
    if len(stripped) > 1:
        for perm in itertools.permutations(stripped):
            # e.g. "2, 3"  or  "2,3"  (with or without space after comma)
            joined = r"\s*,\s*".join(_escape(v) for v in perm)
            patterns.append(re.compile(rf"\b{joined}\b"))

    return patterns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def redact_answers(text: str, answers: list[str], gated: bool) -> str:
    """
    Replace literal answer occurrences in *text* with ``[hidden while you work]``.

    Parameters
    ----------
    text:
        The LLM-generated tutoring prose to filter.
    answers:
        The CAS-certified correct answers as plain strings
        (e.g. ``["3", "-2"]``).  These come from the CAS solver, never
        from the LLM.
    gated:
        When ``True`` the student is still working — apply redaction.
        When ``False`` (problem solved / answer revealed) pass through
        unchanged.

    Returns
    -------
    str
        Filtered text.  When ``gated`` is ``False``, the original *text*
        is returned unmodified.

    Notes
    -----
    See module docstring for the *literal-only* limitation.  This function
    intentionally does the minimum safe thing: redact literals.  It does
    not attempt semantic analysis.
    """
    if not gated or not answers:
        return text

    result = text
    for pattern in _make_patterns(answers):
        result = pattern.sub(_HIDDEN, result)
    return result
