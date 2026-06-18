# mathtutor/llm/offline.py

"""Deterministic, template-based offline tutor.

This module exposes a single callable interface:

    coach(context: CoachingContext) -> str

It is the fallback used by the Orchestrator when the real LLM is unavailable
or not configured.  It uses ONLY:
  * CAS-verified facts carried in the CoachingContext
  * Canned phrase templates keyed on verdict / support level
  * No network calls, no random state (deterministic given the same context)

The interface it satisfies is intentionally minimal so that any LLM wrapper
can drop in as a replacement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── shared contracts ──────────────────────────────────────────────────────────
from mathtutor.contracts import (
    Judgment,
    SupportLevel,
    Verdict,
    verdict_from_judgment,
)


# ── coaching context ──────────────────────────────────────────────────────────

@dataclass
class CoachingContext:
    """All CAS-verified facts the tutor needs to phrase a response.

    The orchestrator builds one of these after every verification step and
    passes it to coach().  All math content inside must already be CAS-
    certified before reaching the tutor — the tutor NEVER adjudicates math.

    Attributes
    ----------
    verdict:
        Coarse verdict from contracts.Verdict.
    judgment:
        Full Judgment dataclass from the CAS verifier.
    support_level:
        Scaffolding level chosen by the orchestrator.
    hint_level:
        0-based index of how many hints have already been given.
    kc_name:
        Human-readable name of the knowledge component being practised.
    problem_statement:
        The original problem text shown to the student.
    student_raw:
        Exactly what the student typed (pre-parse).
    correct_answer_str:
        CAS-formatted correct answer, already filtered by leak_filter.
        May be None when the orchestrator withholds it (independent mode).
    misconception_id:
        BuggyRule.id if the CAS diagnosed a specific misconception, else None.
    misconception_description:
        Human-readable description of the misconception.  May be None.
    worked_steps:
        Ordered list of CAS-certified step strings for WORKED support.
    extra:
        Arbitrary extra context (e.g. partial credit detail).
    """

    verdict: Verdict
    judgment: Judgment
    support_level: SupportLevel = SupportLevel.INDEPENDENT
    hint_level: int = 0
    kc_name: str = "this topic"
    problem_statement: str = ""
    student_raw: str = ""
    correct_answer_str: str | None = None
    misconception_id: str | None = None
    misconception_description: str | None = None
    worked_steps: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


# ── phrase banks ──────────────────────────────────────────────────────────────

# Keys: (verdict, support_level, hint_level_clamped)
# hint_level is clamped to 0..2 so we don't need an unbounded table.

_CORRECT_PHRASES = [
    "Great work — that's correct!",
    "Exactly right. Well done!",
    "That's the correct answer. Keep it up!",
]

_PARTIAL_PHRASES = [
    "You've got part of it right, but the answer isn't complete yet.",
    "That's partially correct — you're on the right track, but something is missing.",
    "Good start! You have some of the solution, but there's more to find.",
]

_ABSTAIN_PHRASES = [
    "I couldn't parse that input — could you rephrase it using standard notation?",
    "That didn't look like a math expression I can read. Try writing it differently.",
    "I had trouble understanding that. Please check your notation and try again.",
]

# Wrong, independent (no hint at all)
_WRONG_INDEPENDENT = [
    "That's not quite right. Review {kc_name} and try again.",
    "Not correct. Think about {kc_name} and give it another shot.",
    "That answer isn't right. Revisit {kc_name} before your next attempt.",
]

# Wrong, completion — give a structural hint, NO values
_WRONG_COMPLETION_H0 = [
    "Think about the structure of the solution to \"{problem}\". What's the first step?",
    "What form should the answer to \"{problem}\" take?",
    "Consider the method for {kc_name}. What operation should you apply first?",
]
_WRONG_COMPLETION_H1 = [
    "You need to apply {kc_name} here. Try setting up the equation carefully.",
    "Remember the key rule for {kc_name} and apply it step by step.",
    "Double-check your setup for {kc_name}. The process should guide you to the answer.",
]
_WRONG_COMPLETION_H2 = [
    "Work through the standard procedure for {kc_name} one step at a time.",
    "Go back to the definition of {kc_name} and apply each rule in order.",
    "Break the problem into smaller pieces using what you know about {kc_name}.",
]

# Misconception addendum (appended after the main phrase)
_MISCONCEPTION_ADDENDUM = (
    "It looks like there may be a common error here: {description}. "
    "Review that rule before retrying."
)

# Worked solution header/footer
_WORKED_HEADER = "Let's walk through this step by step:"
_WORKED_FOOTER = "Try a similar problem on your own to reinforce this."
_WORKED_REVEAL = "The correct answer is: {answer}"


# ── helpers ───────────────────────────────────────────────────────────────────

def _pick(phrases: list[str], index: int = 0) -> str:
    """Return phrase at *index* mod len(phrases), deterministically."""
    return phrases[index % len(phrases)]


def _fmt(template: str, ctx: CoachingContext) -> str:
    return template.format(
        kc_name=ctx.kc_name,
        problem=ctx.problem_statement or "this problem",
        answer=ctx.correct_answer_str or "(see solution)",
        description=ctx.misconception_description or "an unknown misconception",
    )


# ── public interface ──────────────────────────────────────────────────────────

def coach(context: CoachingContext) -> str:
    """Return a templated coaching message based purely on CAS facts.

    This function is deterministic: same context → same output.
    It never adjudicates correctness; it only phrases what the CAS decided.

    Parameters
    ----------
    context:
        CoachingContext populated by the orchestrator.

    Returns
    -------
    str
        A plain-text coaching message ready for display.
    """
    parts: list[str] = []

    verdict = context.verdict
    sl = context.support_level
    hl = min(context.hint_level, 2)  # clamp to 0-2

    # ── 1. main verdict phrase ──────────────────────────────────────────────
    if verdict == Verdict.CORRECT:
        parts.append(_pick(_CORRECT_PHRASES, hl))

    elif verdict == Verdict.PARTIAL:
        parts.append(_pick(_PARTIAL_PHRASES, hl))

    elif verdict == Verdict.ABSTAIN:
        parts.append(_pick(_ABSTAIN_PHRASES, hl))

    else:  # WRONG — branch on support level
        if sl == SupportLevel.WORKED:
            # Full worked solution: header + CAS steps + reveal (if available)
            parts.append(_WORKED_HEADER)
            if context.worked_steps:
                for i, step in enumerate(context.worked_steps, 1):
                    parts.append(f"  Step {i}: {step}")
            else:
                parts.append(_fmt(
                    "  Apply the standard procedure for {kc_name}.", context
                ))
            if context.correct_answer_str:
                parts.append(_fmt(_WORKED_REVEAL, context))
            parts.append(_WORKED_FOOTER)

        elif sl == SupportLevel.COMPLETION:
            bank = [_WRONG_COMPLETION_H0, _WRONG_COMPLETION_H1, _WRONG_COMPLETION_H2][hl]
            parts.append(_fmt(_pick(bank, hl), context))

        else:  # INDEPENDENT — no hints, no values
            parts.append(_fmt(_pick(_WRONG_INDEPENDENT, hl), context))

    # ── 2. misconception addendum (all non-correct verdicts) ───────────────
    if verdict != Verdict.CORRECT and context.misconception_id:
        if context.misconception_description:
            parts.append(_fmt(_MISCONCEPTION_ADDENDUM, context))

    # ── 3. partial credit detail ────────────────────────────────────────────
    if verdict == Verdict.PARTIAL and context.judgment.detail.get("found_roots"):
        found = context.judgment.detail["found_roots"]
        total = context.judgment.detail.get("total_roots", "?")
        parts.append(
            f"You found {len(found)} of {total} solution(s). "
            "Keep going — what other values satisfy the equation?"
        )

    return "\n".join(parts)


# ── LLM-compatible wrapper ────────────────────────────────────────────────────

class OfflineTutor:
    """Drop-in replacement for an LLM coaching object.

    The Orchestrator calls ``llm.coach(context)``; this class satisfies
    that interface using only deterministic templates.

    Example
    -------
    >>> tutor = OfflineTutor()
    >>> msg = tutor.coach(ctx)   # ctx: CoachingContext
    """

    name: str = "offline-template-tutor"

    def coach(self, context: CoachingContext) -> str:  # noqa: D102
        """Delegate to the module-level ``coach`` function."""
        return coach(context)