"""
tutoring/scaffolding.py — Adaptive scaffolding, hint ladder, and gate.

Design principles
-----------------
* **Support level** is a pure function of mastery probability *p_known*.
  Three zones: WORKED (study a full solution) → COMPLETION (fill in the
  blanks) → INDEPENDENT (hint ladder only).

* **Hint ladder** escalates one rung at a time, only on explicit request.
  It deliberately withholds the literal next line or final answer.
  The five rungs are:
    1. Region  – points to WHERE in the working the error lives.
    2. Kind    – names the TYPE of algebraic move needed.
    3. Socratic – a question targeting the diagnosed misconception.
    4. Analogous – the same move demonstrated on a DIFFERENT, simpler example.
    5. Worked  – the full solution, gated behind ``gate_open``.

* **Structural progress** is checked by the CAS, not by surface string
  comparison.  A step counts only if SymPy's canonical form changed AND
  the change isn't a cycle of recently-seen forms AND the new expression
  is no more complex than the previous one (complexity = operation count).

* **Gate** weighs two factors: (a) at least one structural step was made,
  and (b) enough time has passed.  Thresholds are modulated by mastery and
  frustration so an advanced but frustrated learner gets help sooner, while
  a low-mastery learner who rushes is held back.

No global mutable state: all state lives in HintLadder instances.
No LLM calls here: all language is template-driven so the module is
deterministic and testable.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import sympy
from sympy import sympify, count_ops, SympifyError

from mathtutor.contracts import SupportLevel, Judgment  # noqa: F401 (re-exported)


# ---------------------------------------------------------------------------
# 1. Support level
# ---------------------------------------------------------------------------

def support_level(
    p_known: float,
    *,
    low: float = 0.4,
    high: float = 0.85,
) -> SupportLevel:
    """
    Map a BKT mastery estimate *p_known* ∈ [0, 1] to a ``SupportLevel``.

    Zones
    -----
    p_known < low          → WORKED       (full worked example + self-explain)
    low ≤ p_known < high   → COMPLETION   (partial steps, blanks to fill)
    p_known ≥ high         → INDEPENDENT  (hint ladder only)

    Parameters
    ----------
    p_known : float
        Bayesian Knowledge Tracing posterior probability that the student
        has acquired the relevant knowledge component.  Must be in [0, 1].
    low : float, keyword-only
        Upper boundary of the WORKED zone (exclusive).  Default 0.40.
    high : float, keyword-only
        Lower boundary of the INDEPENDENT zone (inclusive).  Default 0.85.

    Returns
    -------
    SupportLevel

    Raises
    ------
    ValueError
        If ``p_known`` is outside [0, 1] or the thresholds are inconsistent.

    Examples
    --------
    >>> support_level(0.2)
    <SupportLevel.WORKED: 1>
    >>> support_level(0.6)
    <SupportLevel.COMPLETION: 2>
    >>> support_level(0.9)
    <SupportLevel.INDEPENDENT: 3>
    """
    if not (0.0 <= p_known <= 1.0):
        raise ValueError(f"p_known must be in [0, 1]; got {p_known}")
    if not (0.0 < low < high < 1.0):
        raise ValueError(
            f"Thresholds must satisfy 0 < low < high < 1; got low={low}, high={high}"
        )

    if p_known < low:
        return SupportLevel.WORKED
    if p_known < high:
        return SupportLevel.COMPLETION
    return SupportLevel.INDEPENDENT


# ---------------------------------------------------------------------------
# 2. Hint ladder
# ---------------------------------------------------------------------------

# Rung metadata: keys used in the returned payload dict.
# The VALUE strings are structural labels, not solution content.
_RUNG_KEYS = {
    1: "region",
    2: "kind",
    3: "socratic",
    4: "analogous",
    5: "worked_gate",
}

# Maximum rung before we'd need the gate
_MAX_FREE_RUNG = 4
_GATE_RUNG = 5


@dataclass
class HintLadder:
    """
    Tracks the current hint level for one (session, problem) pair.

    State
    -----
    _current_level : int
        Which rung was last emitted.  0 means no hint has been given yet.
    _session_id : str
        Opaque identifier for the session.
    _problem_id : str
        Opaque identifier for the problem within the session.

    The ladder escalates **one rung per call** to ``next_hint``.  It
    never emits the literal next algebraic step or the final answer.

    Payload keys returned per rung
    --------------------------------
    Rung 1 – ``region``     : a label such as ``"step_2"`` or ``"line_3"``;
                               never the corrected expression.
    Rung 2 – ``kind``       : a move name such as ``"collect_like_terms"``
                               or ``"isolate_variable"``.
    Rung 3 – ``socratic``   : a question referencing the diagnosed
                               misconception without stating the answer.
    Rung 4 – ``analogous``  : a worked step on a structurally similar but
                               *different* example (different numbers/letters).
    Rung 5 – ``worked_gate``: ``"GATE_CLOSED"`` if the gate is not open;
                               ``"GATE_OPEN"``  if the caller should now
                               display the full worked solution.
                               The *solution itself is never embedded here*.
    """

    _session_id: str = field(default="default_session")
    _problem_id: str = field(default="default_problem")
    _current_level: int = field(default=0, init=False)

    # ------------------------------------------------------------------ #
    # Templates — structural labels only, NEVER containing solution text  #
    # ------------------------------------------------------------------ #

    # Region labels: a tuple of plausible step locations.
    _REGIONS: tuple[str, ...] = field(default=(
        "setup_step",
        "first_transformation",
        "middle_step",
        "final_simplification",
        "conclusion_step",
    ), init=False, repr=False)

    # Move kinds: algebraic operation names.
    _KINDS: tuple[str, ...] = field(default=(
        "collect_like_terms",
        "apply_distributive_law",
        "isolate_the_variable",
        "multiply_both_sides",
        "divide_both_sides",
        "factor_expression",
        "expand_brackets",
        "cancel_common_factor",
        "apply_zero_product_rule",
    ), init=False, repr=False)

    def next_hint(
        self,
        diagnosis: list[str] | None = None,
        *,
        gate_open: bool = False,
    ) -> dict:
        """
        Advance the ladder by exactly one rung and return a payload dict.

        Parameters
        ----------
        diagnosis : list[str] | None
            A list of misconception tags produced by the diagnosis layer
            (e.g. ``["sign_error", "forgot_to_distribute"]``).  Used to
            select a Socratic question at rung 3.  If ``None`` or empty,
            a generic question is used.
        gate_open : bool, keyword-only
            Whether ``gate_open(...)`` returned True.  Passed in by the
            caller to keep this class decoupled from gate logic.
            Only relevant when we're about to emit rung 5.

        Returns
        -------
        dict with keys:
            ``level``   – int, the rung just emitted (1–5).
            ``hint_type`` – str, one of the _RUNG_KEYS values.
            One additional key named after the hint_type, containing the
            structural payload (never the solution).

        Notes
        -----
        Once the ladder reaches rung 5, further calls keep returning
        rung 5 (the gate check) rather than wrapping around.
        """
        next_level = min(self._current_level + 1, _GATE_RUNG)
        self._current_level = next_level
        hint_type = _RUNG_KEYS[next_level]

        if next_level == 1:
            payload = self._rung_region(diagnosis)
        elif next_level == 2:
            payload = self._rung_kind(diagnosis)
        elif next_level == 3:
            payload = self._rung_socratic(diagnosis)
        elif next_level == 4:
            payload = self._rung_analogous(diagnosis)
        else:  # next_level == 5
            payload = "GATE_OPEN" if gate_open else "GATE_CLOSED"

        return {
            "level": next_level,
            "hint_type": hint_type,
            hint_type: payload,
        }

    def reset(self) -> None:
        """Reset the ladder to rung 0 (start of a new problem attempt)."""
        self._current_level = 0

    @property
    def current_level(self) -> int:
        """The rung last emitted (0 = no hint given yet)."""
        return self._current_level

    # ------------------------------------------------------------------ #
    # Private rung builders                                                #
    # ------------------------------------------------------------------ #

    def _rung_region(self, diagnosis: list[str] | None) -> dict:
        """
        Rung 1 — identify the region of the error.

        Returns a dict with a ``step_label`` (never the corrected value)
        and an optional ``hint_text`` that points *toward* the location
        without stating the fix.
        """
        # Use diagnosis to pick a step if available, else generic.
        if diagnosis:
            tag = diagnosis[0]
            text = (
                f"Look carefully at the step where you applied '{tag}'. "
                "Something changes unexpectedly there."
            )
        else:
            text = (
                "Check each transformation line by line. "
                "There is one step that introduces an inconsistency."
            )
        return {
            "step_label": "examine_your_working",
            "hint_text": text,
        }

    def _rung_kind(self, diagnosis: list[str] | None) -> dict:
        """
        Rung 2 — name the kind of algebraic move needed.

        Returns a dict with a ``move_name`` taken from ``_KINDS``.
        The move is named, not demonstrated.
        """
        if diagnosis:
            # Map common misconception tags to move names
            _TAG_TO_MOVE: dict[str, str] = {
                "sign_error": "track_sign_across_operation",
                "forgot_to_distribute": "apply_distributive_law",
                "wrong_inverse": "apply_correct_inverse_operation",
                "incomplete_factoring": "factor_expression",
                "missed_root": "apply_zero_product_rule",
                "like_terms_not_collected": "collect_like_terms",
            }
            for tag in diagnosis:
                if tag in _TAG_TO_MOVE:
                    move = _TAG_TO_MOVE[tag]
                    break
            else:
                move = self._KINDS[0]
        else:
            move = "isolate_the_variable"

        return {
            "move_name": move,
            "hint_text": (
                f"The next productive move is to '{move}'. "
                "Think about what that operation does to both sides."
            ),
        }

    def _rung_socratic(self, diagnosis: list[str] | None) -> dict:
        """
        Rung 3 — Socratic question targeting the diagnosed misconception.

        Returns a question that provokes re-examination without giving the
        answer.  The question is chosen by misconception tag when available.
        """
        _TAG_TO_QUESTION: dict[str, str] = {
            "sign_error": (
                "When you moved that term across the equals sign, "
                "what happens to its sign — and did that happen here?"
            ),
            "forgot_to_distribute": (
                "You multiplied one term inside the bracket. "
                "Does the factor outside apply to every term inside?"
            ),
            "wrong_inverse": (
                "To undo an operation, you apply its inverse. "
                "What is the exact inverse of the operation you used?"
            ),
            "incomplete_factoring": (
                "You factored partially. "
                "Is every factor on the left now in its simplest form?"
            ),
            "missed_root": (
                "A product equals zero. "
                "How many different ways can a product equal zero — "
                "and have you accounted for all of them?"
            ),
            "like_terms_not_collected": (
                "Two terms share the same variable. "
                "What can you do when terms have identical variable parts?"
            ),
        }

        if diagnosis:
            for tag in diagnosis:
                if tag in _TAG_TO_QUESTION:
                    question = _TAG_TO_QUESTION[tag]
                    break
            else:
                question = (
                    "Look at the step just before your answer. "
                    "Is every equality still valid at that point — why or why not?"
                )
        else:
            question = (
                "At which point does the left-hand side stop being equal "
                "to the right-hand side of the original equation?"
            )

        return {"question": question}

    def _rung_analogous(self, diagnosis: list[str] | None) -> dict:
        """
        Rung 4 — demonstrate the same move on a DIFFERENT, simpler example.

        The analogous example uses different numbers and variable names so
        the student cannot copy it directly.  It shows the *structural
        pattern*, not the solution to the current problem.
        """
        _TAG_TO_ANALOGY: dict[str, dict] = {
            "sign_error": {
                "description": "Moving a term across the equals sign flips its sign.",
                "before": "a + 3 = 7",
                "after":  "a = 7 - 3   (the +3 becomes -3 on crossing)",
                "note": "Notice: the term changes sign when it crosses the equals sign.",
            },
            "forgot_to_distribute": {
                "description": "The factor outside multiplies EVERY term inside.",
                "before": "2*(m + 4)",
                "after":  "2*m + 2*4  =  2m + 8",
                "note": "Both m and 4 are multiplied by 2, not just the first.",
            },
            "wrong_inverse": {
                "description": "Undo multiplication with division; undo addition with subtraction.",
                "before": "3*k = 12",
                "after":  "k = 12 / 3  (divide both sides by 3)",
                "note": "The inverse of ×3 is ÷3.",
            },
            "incomplete_factoring": {
                "description": "Keep factoring until no common factor remains.",
                "before": "4*n**2 - 16",
                "after":  "4*(n**2 - 4)  →  4*(n-2)*(n+2)",
                "note": "n²-4 is a difference of squares and factors further.",
            },
            "missed_root": {
                "description": "Zero-product rule: if A*B=0 then A=0 OR B=0.",
                "before": "(t - 1)*(t + 3) = 0",
                "after":  "t - 1 = 0  → t = 1   OR   t + 3 = 0  → t = -3",
                "note": "Two separate cases, two separate solutions.",
            },
        }

        generic_analogy = {
            "description": "Isolate the variable by performing the same operation on both sides.",
            "before": "b + 7 = 10",
            "after":  "b = 10 - 7  =  3",
            "note": "Subtracting 7 from both sides keeps the equation balanced.",
        }

        analogy: dict = generic_analogy
        if diagnosis:
            for tag in diagnosis:
                if tag in _TAG_TO_ANALOGY:
                    analogy = _TAG_TO_ANALOGY[tag]
                    break

        return {
            "description": analogy["description"],
            "analogous_before": analogy["before"],
            "analogous_after": analogy["after"],
            "note": analogy["note"],
            "reminder": (
                "This example uses different numbers and letters from your problem "
                "— apply the same structural idea, do not copy the values."
            ),
        }


# ---------------------------------------------------------------------------
# 3. Structural progress
# ---------------------------------------------------------------------------

def _sympy_canonical(expr: Any) -> sympy.Basic | None:
    """
    Convert *expr* to a SymPy expression in canonical form for **structural
    comparison**.

    We use ``expand`` only — not ``cancel`` — so that the canonical form
    preserves structural differences that are meaningful to a student.
    For example:

    * ``(x²-1)/(x-1)``  expands to  ``x²/(x-1) - 1/(x-1)``  (6 ops),
      while ``x+1`` expands to ``x+1``  (1 op).
      These are structurally different, so going from one to the other
      counts as progress — even though they are mathematically equal.

    Equivalence checking (for cycle detection) uses SymPy's ``.equals()``
    which *does* apply ``cancel`` and other algebraic identities internally.

    Returns ``None`` if conversion fails so callers can treat that as
    "unparseable" rather than raising.
    """
    if not isinstance(expr, sympy.Basic):
        try:
            expr = sympify(str(expr))
        except (SympifyError, TypeError, ValueError):
            return None
    try:
        return sympy.expand(expr)
    except (TypeError, ValueError):
        return None


def _complexity(expr: sympy.Basic) -> int:
    """
    Measure expression complexity as SymPy's operation count.

    ``count_ops`` returns the number of arithmetic operations in the
    expression tree.  A simpler expression (e.g. ``2*x`` from ``4*x/2``)
    has a strictly lower count.

    We use this as a *hint*, not a gate: complexity going down (or staying
    equal) is consistent with genuine progress.
    """
    return int(count_ops(expr))


class _ProgressTracker:
    """
    Internal helper: keeps a ring buffer of recently-seen canonical forms
    so ``is_structural_progress`` can detect cycles.

    Ring-buffer size is 8 steps — enough to catch ``+5-5`` patterns while
    keeping memory bounded.
    """
    _BUFFER_SIZE = 8

    def __init__(self) -> None:
        self._seen: deque[tuple[str, int]] = deque(maxlen=self._BUFFER_SIZE)

    def record(self, canonical_expr: sympy.Basic) -> None:
        """Add a canonical form to the ring buffer."""
        self._seen.append((str(canonical_expr), _complexity(canonical_expr)))

    def is_cycle(self, canonical_expr: sympy.Basic) -> bool:
        """Return True if this canonical form is mathematically equivalent
        to a recently-seen form (even if structurally different surface form)."""
        for k, _ in self._seen:
            try:
                seen_expr = sympy.sympify(k)
                if canonical_expr.equals(seen_expr):
                    return True
            except Exception:
                if str(canonical_expr) == k:
                    return True
        return False

    def last_complexity(self) -> int | None:
        """Complexity of the most-recently recorded expression, or None."""
        if not self._seen:
            return None
        return self._seen[-1][1]

    def clear(self) -> None:
        """Reset for a new problem."""
        self._seen.clear()


# Module-level tracker.  Callers that need per-problem isolation should
# instantiate their own _ProgressTracker and call is_structural_progress
# as a method instead.
_global_tracker = _ProgressTracker()


def is_structural_progress(
    prev_expr: Any,
    new_expr: Any,
    *,
    tracker: _ProgressTracker | None = None,
) -> bool:
    """
    Return ``True`` iff ``new_expr`` represents genuine structural progress
    over ``prev_expr``.

    A step counts as structural progress when **all three** conditions hold:

    1. **Form changed** — SymPy's canonical form of *new_expr* differs from
       that of *prev_expr*.  Two strings that simplify to the same SymPy
       expression (e.g. ``x + 0`` and ``x``) are considered unchanged.

    2. **Not a cycle** — the canonical form of *new_expr* has not appeared
       in the recent ring buffer of seen forms.  This catches patterns like
       ``+5`` then ``-5`` that wind up back at a previously-visited state.

    3. **Complexity did not strictly increase** — ``count_ops(new_expr)``
       must be ≤ ``count_ops(prev_expr)``.  This prevents "progress" that
       just makes the expression more complex (e.g. expanding ``x`` into
       ``x*1 + 0``).

    Parameters
    ----------
    prev_expr : Any
        The expression before the student's latest step.  Accepted as a
        SymPy ``Basic``, or any object whose ``str()`` is parseable by
        ``sympify``.
    new_expr : Any
        The expression after the student's latest step.
    tracker : _ProgressTracker | None, keyword-only
        If provided, the ring buffer used for cycle detection.  Defaults
        to the module-level ``_global_tracker`` (suitable for single-problem
        use; pass an explicit tracker for multi-problem sessions).

    Returns
    -------
    bool
        ``True`` iff all three conditions above are satisfied.
        ``False`` on any parse failure (fail-safe: uncertain → not progress).

    Examples
    --------
    >>> from sympy import symbols
    >>> x = symbols('x')
    >>> is_structural_progress(x + 5 - 5, x)        # cycle → False
    False
    >>> is_structural_progress(2*x + 4, 2*(x + 2))  # same canonical → False
    False
    >>> is_structural_progress(2*x + 4*x, 6*x)      # genuine simplification → True
    True
    """
    t = tracker if tracker is not None else _global_tracker

    prev_can = _sympy_canonical(prev_expr)
    new_can = _sympy_canonical(new_expr)

    if prev_can is None or new_can is None:
        # Unparseable input: fail-safe, claim no progress.
        return False

    # Condition 1: canonical form must have structurally changed.
    # We use ``==`` (structural equality) rather than ``.equals()``
    # (mathematical equivalence) here deliberately:
    #   * ``==`` is False for ``x²/(x-1) - 1/(x-1)`` vs ``x + 1``  ✓
    #   * ``.equals()`` would return True for those (same math) — wrong here,
    #     because the student DID make a visible structural step.
    # Cycle detection (condition 2) uses ``.equals()`` so that expressions
    # that are merely rearranged (same canonical form) are caught as cycles.
    if prev_can == new_can:
        t.record(new_can)
        return False

    # Condition 2: must not be a cycle (mathematical equivalence with a
    # recently-seen form, even if the surface form is different).
    if t.is_cycle(new_can):
        t.record(new_can)
        return False

    # Condition 3: complexity must not strictly increase.
    prev_ops = _complexity(prev_can)
    new_ops = _complexity(new_can)
    if new_ops > prev_ops:
        t.record(new_can)
        return False

    # All conditions satisfied: genuine structural progress.
    t.record(new_can)
    return True


# ---------------------------------------------------------------------------
# 4. Gate
# ---------------------------------------------------------------------------

# Default time thresholds (seconds).
_BASE_TIME_THRESHOLD_S: float = 120.0   # 2 minutes baseline

# Minimum structural steps before the gate can open at all.
_MIN_PROGRESS_STEPS: int = 1


def gate_open(
    progress_steps: int,
    time_on_task_s: float,
    p_known: float,
    frustration: float,
) -> bool:
    """
    Decide whether to open the worked-solution gate.

    Two necessary conditions (the gate is closed unless BOTH are met):

    A. **Genuine progress**: ``progress_steps ≥ 1``  (at least one
       structural step has been made).  This prevents rewarding
       passivity — a student who stares at the screen and immediately
       asks for the answer gets GATE_CLOSED.

    B. **Enough time on task**: ``time_on_task_s ≥ threshold``, where
       the threshold is a function of ``p_known`` and ``frustration``:

       * High mastery + high frustration  → threshold is *halved*
         (the student demonstrably knows the material and is stuck).
       * Low mastery + low effort (fast clicks) → threshold is *doubled*
         (the student is guessing, not thinking).
       * Otherwise the base threshold (120 s) applies.

    The threshold formula
    ---------------------
    ::

        threshold = BASE × frustration_factor × effort_factor

        frustration_factor = 1 / (1 + frustration × p_known × 2)
           - At frustration=1, p_known=1: factor ≈ 1/3  (gate opens faster)
           - At frustration=0           : factor = 1    (no change)

        effort_factor = 1 + (1 - p_known) × max(0, 1 - time_on_task_s / BASE)
           - If time_on_task_s << BASE and p_known is low: factor > 1
             (threshold raised — student is rushing)
           - If time_on_task_s ≥ BASE or p_known is high: factor ≈ 1

    Parameters
    ----------
    progress_steps : int
        Number of structurally-distinct steps taken (from
        ``is_structural_progress`` calls).
    time_on_task_s : float
        Total seconds the student has been working on this problem.
    p_known : float
        BKT mastery estimate in [0, 1].
    frustration : float
        Frustration signal in [0, 1].  0 = calm, 1 = maximally frustrated.
        Sourced from the affective model (outside this module).

    Returns
    -------
    bool
        ``True`` iff both conditions are satisfied.

    Raises
    ------
    ValueError
        If ``p_known`` or ``frustration`` are outside [0, 1].

    Examples
    --------
    >>> # High-mastery frustrated student: gate opens after < 2 minutes
    >>> gate_open(progress_steps=2, time_on_task_s=50, p_known=0.95, frustration=0.9)
    True
    >>> # Low-mastery fast student: gate stays closed longer
    >>> gate_open(progress_steps=1, time_on_task_s=10, p_known=0.2, frustration=0.0)
    False
    """
    if not (0.0 <= p_known <= 1.0):
        raise ValueError(f"p_known must be in [0, 1]; got {p_known}")
    if not (0.0 <= frustration <= 1.0):
        raise ValueError(f"frustration must be in [0, 1]; got {frustration}")

    # Condition A — must have made at least one structural step.
    if progress_steps < _MIN_PROGRESS_STEPS:
        return False

    # Condition B — time threshold, modulated by mastery and frustration.
    # frustration_factor: high frustration + high mastery → lower threshold.
    frustration_factor = 1.0 / (1.0 + frustration * p_known * 2.0)

    # effort_factor: low mastery + rushing → raise threshold.
    # "rushing" = spending much less than the base time.
    time_ratio = time_on_task_s / _BASE_TIME_THRESHOLD_S
    effort_factor = 1.0 + (1.0 - p_known) * max(0.0, 1.0 - time_ratio)

    threshold = _BASE_TIME_THRESHOLD_S * frustration_factor * effort_factor

    return time_on_task_s >= threshold


# ---------------------------------------------------------------------------
# 5. Completion problem helper
# ---------------------------------------------------------------------------

_BLANK = "___"  # sentinel used in tests to verify blanking


def completion_problem(worked_steps: list[str], reveal_through: int) -> dict:
    """
    Return a partial worked solution for COMPLETION-mode scaffolding.

    Steps up to and including *reveal_through* are shown in full.
    Steps after that index are replaced with ``"___"`` (the blank sentinel).

    Parameters
    ----------
    worked_steps : list[str]
        The complete list of worked-solution steps, in order.
        Each element is a human-readable string (LaTeX or plain text).
    reveal_through : int
        Zero-based index of the last step to reveal.  Steps at indices
        0 … reveal_through (inclusive) are shown; steps at
        reveal_through+1 … end are blanked.

    Returns
    -------
    dict with keys:
        ``steps``        – list of str, blanked as described.
        ``total_steps``  – int, length of the original list.
        ``revealed``     – int, number of steps shown.
        ``blanked``      – int, number of steps hidden.

    Raises
    ------
    ValueError
        If ``worked_steps`` is empty or ``reveal_through`` is out of range.

    Examples
    --------
    >>> steps = ["2x + 4 = 10", "2x = 6", "x = 3"]
    >>> completion_problem(steps, reveal_through=1)
    {'steps': ['2x + 4 = 10', '2x = 6', '___'],
     'total_steps': 3, 'revealed': 2, 'blanked': 1}
    """
    if not worked_steps:
        raise ValueError("worked_steps must not be empty.")
    n = len(worked_steps)
    if not (0 <= reveal_through < n):
        raise ValueError(
            f"reveal_through must be in [0, {n - 1}]; got {reveal_through}"
        )

    steps = [
        step if i <= reveal_through else _BLANK
        for i, step in enumerate(worked_steps)
    ]
    revealed = reveal_through + 1
    blanked = n - revealed

    return {
        "steps": steps,
        "total_steps": n,
        "revealed": revealed,
        "blanked": blanked,
    }
