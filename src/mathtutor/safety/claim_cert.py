"""
safety/claim_cert.py — Claim certification for the Verified Math Tutor.

Purpose
-------
The LLM is allowed to make concrete symbolic claims in its tutoring prose
(e.g. "the derivative of x² is <<claim>>2x<</claim>>").  Before that prose
reaches the student, every ``<<claim>>…<</claim>>`` span is:

1. **Parsed** by the CAS.
2. **Verified** against the CAS ground truth for the current problem.
3. **Kept and unwrapped** if correct, or **silently dropped** if wrong or
   unparseable.

This ensures the LLM can contribute pedagogically useful language while the
CAS retains exclusive authority over correctness.

Fail-safe
---------
Any claim that cannot be parsed, or whose equivalence cannot be determined,
is *dropped* (not kept).  We do less rather than risk asserting a
false claim.

Logging
-------
Dropped claims are logged (at WARNING level) with their raw text and the
reason for dropping, for post-hoc auditing.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from mathtutor.contracts import ParseError, Target, Verifier

log = logging.getLogger(__name__)

# Delimiter pattern: <<claim>>…<</claim>>
# We use a non-greedy match so adjacent claims don't merge.
_CLAIM_RE = re.compile(r"<<claim>>(.*?)<</claim>>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_claims(text: str) -> list[tuple[str, str]]:
    """
    Return a list of ``(full_match, inner_text)`` pairs for every
    ``<<claim>>…<</claim>>`` span found in *text*.
    """
    return [(m.group(0), m.group(1).strip()) for m in _CLAIM_RE.finditer(text)]


def _verify_claim(
    inner: str,
    verifier: Verifier,
    target: Target,
) -> bool:
    """
    Return ``True`` iff the claim expressed by *inner* is verified by the CAS.

    A claim passes iff:
    * It parses successfully, AND
    * ``verifier.accepts(student, target).value_equivalent`` is ``True``.

    We use ``value_equivalent`` (not ``correct``) so form constraints on
    the *student* answer don't falsely reject a correct interim claim.

    Returns ``False`` on any exception — fail-safe.
    """
    try:
        artifact = verifier.parse(inner)
    except ParseError as exc:
        log.warning("Claim DROPPED — ParseError: %r  reason=%s", inner, exc)
        return False
    except Exception as exc:  # pragma: no cover — unexpected
        log.warning("Claim DROPPED — unexpected parse error: %r  reason=%s", inner, exc)
        return False

    try:
        judgment = verifier.accepts(artifact, target)
    except Exception as exc:
        log.warning("Claim DROPPED — verifier error: %r  reason=%s", inner, exc)
        return False

    if not judgment.value_equivalent:
        log.warning(
            "Claim DROPPED — value_equivalent=False: %r  "
            "decidable=%s confidence=%.4f",
            inner,
            judgment.decidable,
            judgment.confidence,
        )
        return False

    # Undecidable claims: keep only if confidence is high enough to be useful.
    # We set a conservative threshold of 0.95 so near-certain probabilistic
    # checks still pass, but genuinely uncertain ones are dropped.
    if not judgment.decidable and judgment.confidence < 0.95:
        log.warning(
            "Claim DROPPED — low-confidence probabilistic check: %r  "
            "confidence=%.4f",
            inner,
            judgment.confidence,
        )
        return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def certify(text: str, verifier: Verifier, target: Target) -> str:
    """
    Remove unverifiable or false ``<<claim>>…<</claim>>`` spans from *text*.

    Parameters
    ----------
    text:
        LLM-generated tutoring prose containing zero or more
        ``<<claim>>…<</claim>>`` spans.
    verifier:
        A domain :class:`~mathtutor.contracts.Verifier` instance.
        The CAS — not the LLM — decides correctness.
    target:
        The CAS-certified correct answer for the current problem.

    Returns
    -------
    str
        The prose with verified claims *unwrapped* (delimiters removed,
        inner text kept) and unverifiable/false claims *deleted entirely*.

    Behaviour
    ---------
    * **Correct claim** ``<<claim>>3<</claim>>`` → ``3``
    * **False claim**   ``<<claim>>5<</claim>>`` → *(deleted)*
    * **Unparseable**   ``<<claim>>??<</claim>>`` → *(deleted)*

    Fail-safe
    ---------
    On any unexpected error the offending span is dropped and a WARNING is
    logged.  The rest of the text is returned intact.
    """
    claims = _extract_claims(text)
    if not claims:
        return text

    result = text
    for full_match, inner in claims:
        try:
            if _verify_claim(inner, verifier, target):
                # Unwrap: replace delimiter+content with bare content
                result = result.replace(full_match, inner, 1)
            else:
                # Drop entirely (logging already done inside _verify_claim)
                result = result.replace(full_match, "", 1)
        except Exception as exc:  # pragma: no cover — belt-and-suspenders
            log.warning(
                "Claim DROPPED — unexpected error in certify loop: %r  reason=%s",
                full_match,
                exc,
            )
            result = result.replace(full_match, "", 1)

    return result
