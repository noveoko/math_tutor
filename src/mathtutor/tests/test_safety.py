"""
tests/test_safety.py — Tests for safety/leak_filter.py and safety/claim_cert.py.

All correctness decisions go through PolynomialVerifier (CAS), never through
string similarity.  The test matrix covers the four cases mandated by the spec:

  1. A leaked root (x = 3) is redacted when gated.
  2. A leaked root passes through unchanged when NOT gated.
  3. A false certified claim is removed.
  4. A true certified claim is kept and unwrapped.
  5. An unparseable claim is removed (fail-safe).

Additional edge-case tests cover:
  - Trivial reorderings of a two-root answer set.
  - An empty / no-claim text is returned unchanged.
  - Multiple claims in one response (some good, some bad).
"""

from __future__ import annotations

import pytest
import sympy
from sympy import FiniteSet, symbols

from mathtutor.contracts import Target
from mathtutor.domain.verifiers.polynomial import PolynomialVerifier
from mathtutor.safety.leak_filter import redact_answers
from mathtutor.safety.claim_cert import certify


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

x = symbols("x")


@pytest.fixture()
def verifier() -> PolynomialVerifier:
    """A fresh PolynomialVerifier for each test."""
    return PolynomialVerifier()


def make_target(value: sympy.Basic) -> Target:
    """Helper: wrap a SymPy value in a Target."""
    return Target(domain="expression", payload={"answer": value})


# ===========================================================================
# TASK A — leak_filter.redact_answers
# ===========================================================================


class TestRedactAnswers:
    """Spec-mandated leak-filter tests (§ TASK A)."""

    # --- Core spec test 1 ---------------------------------------------------

    def test_root_is_redacted_when_gated(self):
        """
        A leaked root "x = 3" is redacted from LLM prose when gated=True.
        """
        text = "Great work!  Now try to find x.  Hint: x = 3 is the solution."
        result = redact_answers(text, answers=["3"], gated=True)
        assert "3" not in result
        assert "[hidden while you work]" in result

    # --- Core spec test 2 ---------------------------------------------------

    def test_root_passes_through_when_not_gated(self):
        """
        The same text is returned unchanged when gated=False
        (student has finished / answer revealed).
        """
        text = "Great work!  Now try to find x.  Hint: x = 3 is the solution."
        result = redact_answers(text, answers=["3"], gated=False)
        assert result == text

    # --- Trivial reordering -------------------------------------------------

    def test_trivial_reordering_is_redacted(self):
        """
        "3, 2" and "2, 3" are both redacted for a two-root answer {2, 3}.
        """
        answers = ["2", "3"]

        text_forward = "The solutions are 2, 3 — check both!"
        result_f = redact_answers(text_forward, answers=answers, gated=True)
        assert "2, 3" not in result_f
        assert "[hidden while you work]" in result_f

        text_reverse = "The solutions are 3, 2 — check both!"
        result_r = redact_answers(text_reverse, answers=answers, gated=True)
        assert "3, 2" not in result_r
        assert "[hidden while you work]" in result_r

    # --- Individual roots in multi-root case --------------------------------

    def test_individual_roots_redacted_in_multi_root_case(self):
        """
        Each root is individually redacted even without the comma-list form.
        """
        answers = ["2", "3"]
        text = "Try substituting 2 into the equation.  Or maybe 3?"
        result = redact_answers(text, answers=answers, gated=True)
        assert "2" not in result
        assert "3" not in result

    # --- No answers → text unchanged ----------------------------------------

    def test_empty_answers_list_is_noop(self):
        text = "Try x = 7."
        assert redact_answers(text, answers=[], gated=True) == text

    # --- Word-boundary safety -----------------------------------------------

    def test_word_boundary_does_not_clobber_other_digits(self):
        """
        Answer "3" must not redact the "3" inside "13" or "30".
        """
        text = "Look at step 13 and 30 in your notes."
        result = redact_answers(text, answers=["3"], gated=True)
        # "13" and "30" should be untouched; standalone "3" would be caught
        assert "13" in result
        assert "30" in result

    # --- Negative answer ----------------------------------------------------

    def test_negative_root_is_redacted(self):
        text = "One solution is -2, the other is 5."
        result = redact_answers(text, answers=["-2", "5"], gated=True)
        assert "-2" not in result
        assert "5" not in result


# ===========================================================================
# TASK B — claim_cert.certify
# ===========================================================================


class TestCertify:
    """Spec-mandated claim-certification tests (§ TASK B)."""

    # --- Core spec test 3: false claim is removed ---------------------------

    def test_false_claim_is_removed(self, verifier):
        """
        A claim that is wrong (5 ≠ 3) is removed from the prose.
        """
        # Target: x = 3  (FiniteSet so the verifier uses solution-set path)
        target = make_target(FiniteSet(3))

        text = "The answer is <<claim>>5<</claim>> — did you get that?"
        result = certify(text, verifier, target)

        assert "<<claim>>" not in result
        assert "5" not in result
        # The surrounding prose survives
        assert "The answer is" in result
        assert "did you get that?" in result

    # --- Core spec test 4: true claim is kept and unwrapped -----------------

    def test_true_claim_is_kept_and_unwrapped(self, verifier):
        """
        A correct claim (3 == 3) is kept and its delimiters are removed.
        """
        target = make_target(FiniteSet(3))

        text = "The answer is <<claim>>3<</claim>> — well done!"
        result = certify(text, verifier, target)

        # Delimiters gone
        assert "<<claim>>" not in result
        assert "<</claim>>" not in result
        # Content preserved
        assert "3" in result
        assert "well done!" in result

    # --- Core spec test 5: unparseable claim is removed (fail-safe) ---------

    def test_unparseable_claim_is_removed(self, verifier):
        """
        A claim that cannot be parsed (e.g. random punctuation) is dropped,
        not propagated.  Fail-safe path.
        """
        target = make_target(FiniteSet(3))

        text = "Try <<claim>>???#!<</claim>> and see what happens."
        result = certify(text, verifier, target)

        assert "<<claim>>" not in result
        assert "???#!" not in result
        # Surrounding prose survives
        assert "Try" in result
        assert "and see what happens." in result

    # --- No claims → text returned unchanged --------------------------------

    def test_no_claims_text_unchanged(self, verifier):
        target = make_target(FiniteSet(3))
        text = "Here is a hint: think about what value makes x − 3 zero."
        assert certify(text, verifier, target) == text

    # --- Mixed: one good, one bad -------------------------------------------

    def test_mixed_claims_good_kept_bad_dropped(self, verifier):
        """
        In a response with two claims, the correct one is unwrapped and the
        false one is removed.
        """
        target = make_target(FiniteSet(3))
        text = (
            "First, note that x = <<claim>>3<</claim>>.  "
            "Also, some say x = <<claim>>7<</claim>> but that's wrong."
        )
        result = certify(text, verifier, target)

        assert "3" in result          # good claim kept
        assert "7" not in result      # false claim removed
        assert "<<claim>>" not in result

    # --- Expression claim (not just numbers) --------------------------------

    def test_expression_claim_kept_when_equivalent(self, verifier):
        """
        An expression claim that is algebraically equivalent to the target
        is kept even if written differently.
        e.g. target = 6/2, claim = 3  (both equal 3)
        """
        target = make_target(sympy.Rational(6, 2))   # evaluates to 3

        text = "So the answer simplifies to <<claim>>3<</claim>>."
        result = certify(text, verifier, target)

        assert "3" in result
        assert "<<claim>>" not in result

    # --- Equation claim matched against FiniteSet target --------------------

    def test_equation_claim_matched_to_solution_set(self, verifier):
        """
        A claim written as ``x = 3`` is equivalent to a FiniteSet({3}) target.
        """
        target = make_target(FiniteSet(3))

        text = "We find <<claim>>x = 3<</claim>>."
        result = certify(text, verifier, target)

        assert "<<claim>>" not in result
        assert "x = 3" in result
