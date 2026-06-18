"""
tests/test_scaffolding.py — pytest suite for tutoring/scaffolding.py

Test strategy
-------------
Each public function / class has its own section.  Tests are written
as plain functions (no class-based grouping) to keep them concise and
independently runnable.

The sentinel "ANSWER" must never appear in any hint payload — this is
the key non-negotiable invariant checked exhaustively below.
"""

from __future__ import annotations

import pytest
from sympy import symbols, expand, Integer

from mathtutor.contracts import SupportLevel
from mathtutor.tutoring.scaffolding import (
    support_level,
    HintLadder,
    is_structural_progress,
    gate_open,
    completion_problem,
    _ProgressTracker,
    _BLANK,
)

x, y, a, b, t = symbols("x y a b t")


# ===========================================================================
# Helpers
# ===========================================================================

def _all_strings_in(obj) -> list[str]:
    """
    Recursively collect every string value nested anywhere in *obj*
    (dict, list, or scalar).  Used to hunt for the "ANSWER" sentinel.
    """
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        out: list[str] = []
        for v in obj.values():
            out.extend(_all_strings_in(v))
        return out
    if isinstance(obj, (list, tuple)):
        out = []
        for item in obj:
            out.extend(_all_strings_in(item))
        return out
    return []


def _ladder_full_escalation(diagnosis=None) -> list[dict]:
    """Return all five rungs for a fresh ladder."""
    ladder = HintLadder()
    results = []
    for _ in range(5):
        results.append(ladder.next_hint(diagnosis=diagnosis))
    return results


# ===========================================================================
# 1. support_level
# ===========================================================================

class TestSupportLevel:
    def test_low_p_gives_worked(self):
        assert support_level(0.0) == SupportLevel.WORKED
        assert support_level(0.2) == SupportLevel.WORKED
        assert support_level(0.39) == SupportLevel.WORKED

    def test_boundary_low_gives_completion(self):
        # At exactly the low threshold, we're in COMPLETION, not WORKED.
        assert support_level(0.40) == SupportLevel.COMPLETION

    def test_mid_p_gives_completion(self):
        assert support_level(0.5) == SupportLevel.COMPLETION
        assert support_level(0.7) == SupportLevel.COMPLETION
        assert support_level(0.84) == SupportLevel.COMPLETION

    def test_boundary_high_gives_independent(self):
        # At exactly the high threshold we graduate to INDEPENDENT.
        assert support_level(0.85) == SupportLevel.INDEPENDENT

    def test_high_p_gives_independent(self):
        assert support_level(0.9) == SupportLevel.INDEPENDENT
        assert support_level(1.0) == SupportLevel.INDEPENDENT

    def test_custom_thresholds(self):
        assert support_level(0.3, low=0.35, high=0.7) == SupportLevel.WORKED
        assert support_level(0.5, low=0.35, high=0.7) == SupportLevel.COMPLETION
        assert support_level(0.8, low=0.35, high=0.7) == SupportLevel.INDEPENDENT

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            support_level(-0.01)
        with pytest.raises(ValueError):
            support_level(1.01)

    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError):
            support_level(0.5, low=0.9, high=0.5)   # inverted
        with pytest.raises(ValueError):
            support_level(0.5, low=0.0, high=0.8)   # low == 0 not allowed


# ===========================================================================
# 2. HintLadder
# ===========================================================================

class TestHintLadder:
    # ---- basic escalation ----

    def test_first_hint_is_rung_1(self):
        ladder = HintLadder()
        result = ladder.next_hint()
        assert result["level"] == 1
        assert result["hint_type"] == "region"

    def test_ladder_escalates_one_rung_per_call(self):
        ladder = HintLadder()
        levels = [ladder.next_hint()["level"] for _ in range(5)]
        assert levels == [1, 2, 3, 4, 5]

    def test_rung_5_does_not_wrap_beyond_5(self):
        ladder = HintLadder()
        for _ in range(6):
            r = ladder.next_hint()
        assert r["level"] == 5

    def test_reset_restarts_from_rung_1(self):
        ladder = HintLadder()
        ladder.next_hint()
        ladder.next_hint()
        ladder.reset()
        assert ladder.current_level == 0
        result = ladder.next_hint()
        assert result["level"] == 1

    # ---- ANSWER sentinel must never appear ----

    def test_no_answer_sentinel_no_diagnosis(self):
        """
        Exhaustively check that the word 'ANSWER' never appears in any
        hint payload across a full ladder escalation with no diagnosis.
        """
        rungs = _ladder_full_escalation(diagnosis=None)
        for rung in rungs:
            strings = _all_strings_in(rung)
            for s in strings:
                assert "ANSWER" not in s.upper(), (
                    f"Rung {rung['level']} leaked the ANSWER sentinel: {s!r}"
                )

    def test_no_answer_sentinel_with_sign_error_diagnosis(self):
        rungs = _ladder_full_escalation(diagnosis=["sign_error"])
        for rung in rungs:
            for s in _all_strings_in(rung):
                assert "ANSWER" not in s.upper()

    def test_no_answer_sentinel_with_distribution_diagnosis(self):
        rungs = _ladder_full_escalation(diagnosis=["forgot_to_distribute"])
        for rung in rungs:
            for s in _all_strings_in(rung):
                assert "ANSWER" not in s.upper()

    def test_no_answer_sentinel_with_missed_root_diagnosis(self):
        rungs = _ladder_full_escalation(diagnosis=["missed_root"])
        for rung in rungs:
            for s in _all_strings_in(rung):
                assert "ANSWER" not in s.upper()

    def test_no_answer_sentinel_with_all_known_diagnoses(self):
        all_tags = [
            "sign_error", "forgot_to_distribute", "wrong_inverse",
            "incomplete_factoring", "missed_root", "like_terms_not_collected",
        ]
        for tag in all_tags:
            rungs = _ladder_full_escalation(diagnosis=[tag])
            for rung in rungs:
                for s in _all_strings_in(rung):
                    assert "ANSWER" not in s.upper(), (
                        f"Tag '{tag}', rung {rung['level']}: {s!r}"
                    )

    # ---- payload structure ----

    def test_rung_3_returns_a_question(self):
        ladder = HintLadder()
        ladder.next_hint(); ladder.next_hint()  # skip to rung 3
        r = ladder.next_hint(diagnosis=["sign_error"])
        assert "question" in r["socratic"]
        assert "?" in r["socratic"]["question"]

    def test_rung_4_has_reminder_about_different_numbers(self):
        ladder = HintLadder()
        for _ in range(3):
            ladder.next_hint()
        r = ladder.next_hint()
        reminder = r["analogous"].get("reminder", "")
        assert "different" in reminder.lower()

    def test_rung_5_gate_closed_by_default(self):
        ladder = HintLadder()
        for _ in range(4):
            ladder.next_hint()
        r = ladder.next_hint(gate_open=False)
        assert r["worked_gate"] == "GATE_CLOSED"

    def test_rung_5_gate_open_when_flag_set(self):
        ladder = HintLadder()
        for _ in range(4):
            ladder.next_hint()
        r = ladder.next_hint(gate_open=True)
        assert r["worked_gate"] == "GATE_OPEN"


# ===========================================================================
# 3. is_structural_progress
# ===========================================================================

class TestIsStructuralProgress:
    """
    Each test uses an explicit _ProgressTracker so they are completely
    independent from one another.
    """

    def _t(self):
        return _ProgressTracker()

    # ---- should return False ----

    def test_no_change_is_not_progress(self):
        tr = self._t()
        assert not is_structural_progress(x + 1, x + 1, tracker=tr)

    def test_algebraically_same_is_not_progress(self):
        # x + 0 and x expand to the same canonical form.
        tr = self._t()
        assert not is_structural_progress(x + 0, x, tracker=tr)

    def test_additive_cycle_plus_5_minus_5(self):
        """
        Classic cycle: x  →  x + 5  →  x + 5 - 5 (= x again).
        The third step is a cycle and must be rejected.
        """
        tr = self._t()
        # Step 1: x → x + 5 (progress? yes — form changed and got simpler... 
        #         actually count_ops(x+5) > count_ops(x), so this is 
        #         complexity-increasing and should return False)
        assert not is_structural_progress(x, x + 5, tracker=tr)
        # Step 2: x + 5 → x  (detected as cycle since x was starting form)
        # Even if x+5 wasn't recorded as progress, we still test the cycle 
        # detection logic independently:
        tr2 = self._t()
        tr2.record(expand(x))  # seed x as seen
        assert not is_structural_progress(x + 5, x, tracker=tr2)  # x is a cycle

    def test_plus5_minus5_is_cycle(self):
        """Explicit cycle: start at x+3, go to x+8, then back to x+3."""
        tr = self._t()
        tr.record(expand(x + 3))   # seed x+3 as seen
        # x+8 → different form, and ops(x+8) == ops(x+3), so progress=True
        assert is_structural_progress(x + 3, x + 8, tracker=tr)
        # x+3 again → cycle (it's in the buffer)
        assert not is_structural_progress(x + 8, x + 3, tracker=tr)

    def test_complexity_increase_rejected(self):
        """
        Expanding x into x*1 + 0*x is complexity-increasing.
        Sympy usually simplifies this, but we can test with an explicit
        expansion that genuinely adds operations.
        """
        tr = self._t()
        # 6*x has count_ops=1; 2*x + 4*x has count_ops=3
        # So rewriting 6*x as 2*x + 4*x is complexity-increasing → not progress.
        assert not is_structural_progress(6*x, 2*x + 4*x, tracker=tr)

    def test_unparseable_returns_false(self):
        tr = self._t()
        assert not is_structural_progress("{{invalid{{", x, tracker=tr)
        assert not is_structural_progress(x, "}}not sympy}}", tracker=tr)

    # ---- should return True ----

    def test_genuine_simplification_accepted(self):
        """
        ``(x**2 - x - 6) / (x - 3)  →  x + 2`` is a genuine simplification.

        SymPy's ``expand`` does NOT cancel polynomial factors in fractions;
        it distributes the denominator, giving a form with many more
        operations (9 ops) vs the simplified ``x + 2`` (1 op).
        Different ``expand``-canonical forms, complexity drops → True.
        """
        from sympy import sympify as S
        # expand((x^2-x-6)/(x-3)) = x^2/(x-3) - x/(x-3) - 6/(x-3) [9 ops]
        # expand(x+2) = x + 2  [1 op]
        before = S("(x**2 - x - 6) / (x - 3)")
        after  = S("x + 2")
        tr = self._t()
        assert is_structural_progress(before, after, tracker=tr)

    def test_collecting_like_terms_is_progress(self):
        """
        ``(x**2 + 2*x - 8) / (x - 2)  →  x + 4``

        SymPy ``expand`` does not cancel polynomial factors in rational
        expressions, so the before-form expands to a multi-term fraction
        (high ops) while the after-form is ``x + 4`` (low ops).
        Different expand-canonical forms, lower complexity → True.
        """
        from sympy import sympify as S
        # expand((x^2+2x-8)/(x-2))  =  multi-term fraction  [high ops]
        # expand(x+4)  =  x + 4  [1 op]
        before = S("(x**2 + 2*x - 8) / (x - 2)")
        after  = S("x + 4")
        tr = self._t()
        assert is_structural_progress(before, after, tracker=tr)

    def test_cancellation_is_progress(self):
        """
        ``(x**2 - 1) / (x - 1)  →  x + 1``

        SymPy does NOT auto-cancel common polynomial factors in rational
        expressions built from sympify.  Under ``expand`` only:
        * before canonicalises to ``x**2/(x-1) - 1/(x-1)``  (6 ops)
        * after canonicalises to  ``x + 1``                  (1 op)
        Different forms, complexity decreases → True.
        """
        from sympy import sympify as S
        expr_before = S("(x**2 - 1) / (x - 1)")
        expr_after  = S("x + 1")
        tr = self._t()
        # Sanity: SymPy keeps rational form until cancel is applied
        assert str(expr_before) == "(x**2 - 1)/(x - 1)"
        assert is_structural_progress(expr_before, expr_after, tracker=tr)

    def test_solving_linear_eq_is_progress(self):
        """2*x + 6  →  2*x  is NOT (complexity same but we subtract 6 so form
        changes and ops go from 2 to 1 after x = -3 — let's just do 2*x → x)."""
        tr = self._t()
        # 2*x has count_ops 1; x has count_ops 0. Simplification.
        assert is_structural_progress(2*x, x, tracker=tr)


# ===========================================================================
# 4. gate_open
# ===========================================================================

class TestGateOpen:
    """
    Key invariant: high-mastery + frustrated → gate opens FASTER than
    low-mastery + fast (low effort).
    """

    def test_no_progress_keeps_gate_closed(self):
        # Even a seasoned, frustrated student can't skip the work.
        assert not gate_open(
            progress_steps=0,
            time_on_task_s=9999.0,
            p_known=0.99,
            frustration=1.0,
        )

    def test_high_mastery_frustrated_opens_faster(self):
        """
        A student with p_known=0.95 and frustration=0.9 should hit the
        threshold at a much shorter time than the 120 s baseline.
        We test with 50 s — well below 120 s — and expect OPEN.
        """
        assert gate_open(
            progress_steps=2,
            time_on_task_s=50.0,
            p_known=0.95,
            frustration=0.9,
        )

    def test_low_mastery_fast_stays_closed(self):
        """
        A student with p_known=0.2 and frustration=0.0 who only spent 10 s
        should still be blocked.
        """
        assert not gate_open(
            progress_steps=1,
            time_on_task_s=10.0,
            p_known=0.2,
            frustration=0.0,
        )

    def test_high_mastery_frustrated_threshold_lt_low_mastery_fast_threshold(self):
        """
        Verify the asymmetry quantitatively: the effective threshold for
        (high mastery, high frustration) must be less than the effective
        threshold for (low mastery, low frustration + fast clicks).

        We do this by finding the minimum time at which each opens.
        """
        def min_time_to_open(p_known, frustration, steps=2):
            for t in range(1, 1000):
                if gate_open(progress_steps=steps, time_on_task_s=float(t),
                             p_known=p_known, frustration=frustration):
                    return t
            return 1000  # never opened in range

        t_expert_frustrated = min_time_to_open(p_known=0.95, frustration=0.9)
        t_novice_fast = min_time_to_open(p_known=0.2, frustration=0.0)

        assert t_expert_frustrated < t_novice_fast, (
            f"Expected expert+frustrated ({t_expert_frustrated}s) to open "
            f"before novice+fast ({t_novice_fast}s)"
        )

    def test_baseline_opens_after_sufficient_time(self):
        """
        Mid-mastery student who spends ≥ 120 s and made progress should
        eventually open.
        """
        assert gate_open(
            progress_steps=3,
            time_on_task_s=130.0,
            p_known=0.6,
            frustration=0.3,
        )

    def test_invalid_p_known_raises(self):
        with pytest.raises(ValueError):
            gate_open(1, 60.0, p_known=-0.1, frustration=0.5)

    def test_invalid_frustration_raises(self):
        with pytest.raises(ValueError):
            gate_open(1, 60.0, p_known=0.5, frustration=1.5)


# ===========================================================================
# 5. completion_problem
# ===========================================================================

class TestCompletionProblem:
    STEPS = ["2x + 4 = 10", "2x = 6", "x = 3"]

    def test_reveal_through_0_blanks_rest(self):
        result = completion_problem(self.STEPS, reveal_through=0)
        assert result["steps"] == ["2x + 4 = 10", _BLANK, _BLANK]
        assert result["revealed"] == 1
        assert result["blanked"] == 2

    def test_reveal_through_1_blanks_last(self):
        result = completion_problem(self.STEPS, reveal_through=1)
        assert result["steps"] == ["2x + 4 = 10", "2x = 6", _BLANK]
        assert result["revealed"] == 2
        assert result["blanked"] == 1

    def test_reveal_through_last_shows_all(self):
        result = completion_problem(self.STEPS, reveal_through=2)
        assert result["steps"] == self.STEPS
        assert result["blanked"] == 0

    def test_total_steps_is_correct(self):
        result = completion_problem(self.STEPS, reveal_through=1)
        assert result["total_steps"] == 3

    def test_empty_steps_raises(self):
        with pytest.raises(ValueError):
            completion_problem([], reveal_through=0)

    def test_out_of_range_reveal_raises(self):
        with pytest.raises(ValueError):
            completion_problem(self.STEPS, reveal_through=5)
        with pytest.raises(ValueError):
            completion_problem(self.STEPS, reveal_through=-1)
