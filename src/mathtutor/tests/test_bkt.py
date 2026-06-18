"""
tests/test_bkt.py
=================
Pytest suite for mathtutor/learner/bkt.py.

All expected numeric values are derived from the closed-form BKT equations
and shown step-by-step in comments so the test doubles as living documentation
of the model's arithmetic.

Default BKTParams used throughout unless stated:
    L0 = 0.20,  T = 0.30,  S = 0.10,  G = 0.20
"""

import math
import pytest

from mathtutor.learner.bkt import BKTParams, BKTLearnerState, propagate


# ===========================================================================
# Helpers / fixtures
# ===========================================================================

def fresh_state(kc_id: str = "KC1") -> tuple[BKTLearnerState, str]:
    """Return a BKTLearnerState freshly initialised with default params."""
    state = BKTLearnerState()
    state.register(kc_id)
    return state, kc_id


def one_step_correct(p: float, s: float = 0.10, g: float = 0.20,
                     t: float = 0.30) -> float:
    """
    Reference implementation of the two-step BKT update for a correct answer.
    Used to cross-check the main implementation independently.
    """
    # Step 1
    num = p * (1 - s)
    den = num + (1 - p) * g
    p_post = num / den
    # Step 2
    return p_post + (1 - p_post) * t


def one_step_incorrect(p: float, s: float = 0.10, g: float = 0.20,
                       t: float = 0.30) -> float:
    """Reference implementation for an incorrect answer."""
    num = p * s
    den = num + (1 - p) * (1 - g)
    p_post = num / den
    return p_post + (1 - p_post) * t


# ===========================================================================
# 1.  BKTParams validation
# ===========================================================================

class TestBKTParamsValidation:

    def test_default_params_are_valid(self):
        """Literature defaults should pass without error."""
        p = BKTParams()
        assert p.l0 == 0.20
        assert p.t  == 0.30
        assert p.s  == 0.10
        assert p.g  == 0.20

    def test_custom_valid_params(self):
        p = BKTParams(l0=0.3, t=0.4, s=0.05, g=0.15)
        assert p.l0 == 0.3

    def test_s_plus_g_equal_to_one_raises(self):
        """s + g == 1.0 is degenerate (boundary) — must raise."""
        with pytest.raises(ValueError, match="s \\+ g"):
            BKTParams(s=0.5, g=0.5)

    def test_s_plus_g_above_one_raises(self):
        """The spec example: s=0.6, g=0.6 must raise."""
        with pytest.raises(ValueError, match="s \\+ g"):
            BKTParams(s=0.6, g=0.6)

    @pytest.mark.parametrize("param,val", [
        ("l0", 0.0), ("l0", 1.0), ("l0", -0.1), ("l0", 1.1),
        ("t",  0.0), ("t",  1.0),
        ("s",  0.0), ("s",  1.0),
        ("g",  0.0), ("g",  1.0),
    ])
    def test_boundary_values_raise(self, param, val):
        """Strict open-interval: 0 and 1 themselves are not allowed."""
        kwargs = dict(l0=0.2, t=0.3, s=0.1, g=0.2)
        kwargs[param] = val
        with pytest.raises(ValueError):
            BKTParams(**kwargs)


# ===========================================================================
# 2.  Single correct answer from L0 = 0.20
# ===========================================================================

class TestSingleCorrectAnswer:
    """
    Walk through the arithmetic explicitly so the test is self-documenting.

    Starting state:  p = L0 = 0.20
    Params: S=0.10, G=0.20, T=0.30

    --- Step 1: condition on correct answer ---
        numerator   = p*(1-S)      = 0.20 * 0.90 = 0.180
        denominator = num + (1-p)*G = 0.180 + 0.80*0.20 = 0.180 + 0.160 = 0.340
        p_post      = 0.180 / 0.340 ≈ 0.52941

    --- Step 2: learning transition ---
        p_next = p_post + (1 - p_post)*T
               = 0.52941 + (1 - 0.52941)*0.30
               = 0.52941 + 0.47059*0.30
               = 0.52941 + 0.14118
               ≈ 0.67059

    So after one correct answer, p ≈ 0.671.
    We assert p ∈ [0.65, 0.69] to tolerate floating-point rounding.
    We also assert p < 0.95 (single answer CANNOT declare mastery).
    """

    def test_one_correct_raises_p_to_roughly_0_67(self):
        state, kc = fresh_state()

        # Sanity-check: initial p equals L0
        assert state.p_mastered(kc) == pytest.approx(0.20)

        p_new = state.observe(kc, correct=True)

        # --- Arithmetic shown above ---
        # Step 1: p_post = (0.20*0.90) / (0.20*0.90 + 0.80*0.20)
        #                = 0.180 / 0.340 ≈ 0.52941
        # Step 2: p_next = 0.52941 + (1 - 0.52941)*0.30 ≈ 0.67059
        assert 0.65 <= p_new <= 0.69, (
            f"Expected p ≈ 0.671 after one correct; got {p_new:.5f}"
        )

    def test_one_correct_cannot_declare_mastery(self):
        """
        Even with guessing modelled, a single correct answer from p=0.20
        should never cross the 0.95 mastery threshold.
        """
        state, kc = fresh_state()
        state.observe(kc, correct=True)
        assert not state.mastered(kc), (
            "A single correct answer must not declare mastery (guessing guard)."
        )

    def test_reference_formula_matches_implementation(self):
        """Cross-check implementation against the standalone reference formula."""
        state, kc = fresh_state()
        p_impl = state.observe(kc, correct=True)
        p_ref  = one_step_correct(0.20)
        assert p_impl == pytest.approx(p_ref, rel=1e-9)


# ===========================================================================
# 3.  Multiple consecutive correct answers cross 0.95
# ===========================================================================

class TestConsecutiveCorrectAnswers:
    """
    Trace through multiple correct answers.

    After answer 1: p ≈ 0.671  (see TestSingleCorrectAnswer arithmetic)
    After answer 2:
        Step 1: num = 0.671*(0.90) = 0.604
                den = 0.604 + 0.329*0.20 = 0.604 + 0.066 = 0.670
                p_post = 0.604/0.670 ≈ 0.901
        Step 2: p_next = 0.901 + 0.099*0.30 ≈ 0.931
    After answer 3:
        Step 1: num = 0.931*0.90 = 0.838
                den = 0.838 + 0.069*0.20 = 0.838 + 0.014 = 0.852
                p_post = 0.838/0.852 ≈ 0.984
        Step 2: p_next = 0.984 + 0.016*0.30 ≈ 0.989

    So 3 correct answers should cross 0.95.  We also verify 4 answers to be safe.
    """

    def test_three_correct_answers_cross_mastery(self):
        state, kc = fresh_state()
        for _ in range(3):
            state.observe(kc, correct=True)

        # After 3 correct answers p ≈ 0.989 (see arithmetic above)
        assert state.mastered(kc), (
            f"Expected mastery after 3 correct answers; p={state.p_mastered(kc):.4f}"
        )

    def test_four_correct_answers_cross_mastery(self):
        """Belt-and-braces: 4 answers also cross the threshold."""
        state, kc = fresh_state()
        for _ in range(4):
            state.observe(kc, correct=True)
        assert state.mastered(kc)

    def test_two_correct_answers_do_not_cross_mastery(self):
        """
        After 2 correct answers p ≈ 0.931, which is below 0.95.
        This confirms the threshold is non-trivial to reach.
        """
        state, kc = fresh_state()
        for _ in range(2):
            state.observe(kc, correct=True)

        p = state.p_mastered(kc)
        assert p < 0.95, f"Expected p < 0.95 after 2 correct; got {p:.4f}"
        # Also assert it's in a reasonable range
        assert 0.90 <= p <= 0.95, f"Unexpected value {p:.4f} after 2 correct"


# ===========================================================================
# 4.  Incorrect answer lowers p
# ===========================================================================

class TestIncorrectAnswerLowersP:
    """
    From p = 0.20, one incorrect answer:

    Step 1: p_post = p*S / (p*S + (1-p)*(1-G))
                   = 0.20*0.10 / (0.20*0.10 + 0.80*0.80)
                   = 0.020 / (0.020 + 0.640)
                   = 0.020 / 0.660
                   ≈ 0.030

    Step 2: p_next = 0.030 + (1 - 0.030)*0.30
                   = 0.030 + 0.970*0.30
                   = 0.030 + 0.291
                   ≈ 0.321

    So p *rises* (the learning transition still happens) but the posterior
    from Step 1 drops hard — correctly interpreting an error as evidence of
    non-mastery.  Net result: p_next ≈ 0.321 < 0.671 (one-correct baseline).
    The key invariant: an incorrect answer produces a lower p than the
    corresponding correct answer would have.
    """

    def test_incorrect_answer_produces_lower_p_than_correct(self):
        """
        Incorrect answer from p=0.20 → p ≈ 0.321.
        Correct answer from p=0.20 → p ≈ 0.671.
        Incorrect must give a lower result.
        """
        state_correct, kc_c = fresh_state("KC_correct")
        state_wrong,   kc_w = fresh_state("KC_wrong")

        p_after_correct   = state_correct.observe(kc_c, correct=True)
        p_after_incorrect = state_wrong.observe(kc_w,   correct=False)

        assert p_after_incorrect < p_after_correct, (
            f"Incorrect answer should yield lower p than correct; "
            f"got {p_after_incorrect:.4f} vs {p_after_correct:.4f}"
        )

    def test_incorrect_answer_from_low_prior_approximately_0_32(self):
        """
        Arithmetic:
            Step 1: p_post = (0.20*0.10) / (0.20*0.10 + 0.80*0.80)
                           = 0.020 / 0.660 ≈ 0.0303
            Step 2: p_next = 0.0303 + (1-0.0303)*0.30 ≈ 0.321
        Assert p ∈ [0.30, 0.34].
        """
        state, kc = fresh_state()
        p_new = state.observe(kc, correct=False)

        # Step 1: p_post = 0.020/0.660 ≈ 0.0303
        # Step 2: p_next ≈ 0.321
        assert 0.30 <= p_new <= 0.34, (
            f"Expected p ≈ 0.321 after incorrect from L0=0.20; got {p_new:.5f}"
        )

    def test_reference_formula_matches_for_incorrect(self):
        state, kc = fresh_state()
        p_impl = state.observe(kc, correct=False)
        p_ref  = one_step_incorrect(0.20)
        assert p_impl == pytest.approx(p_ref, rel=1e-9)

    def test_mastery_lost_after_errors(self):
        """
        Build up to high-p then give an incorrect answer; mastery should drop.
        """
        state, kc = fresh_state()
        for _ in range(3):
            state.observe(kc, correct=True)
        assert state.mastered(kc), "Pre-condition: should be mastered after 3 correct"

        state.observe(kc, correct=False)
        # After one wrong from ~0.989 the model should drop back below 0.95
        assert not state.mastered(kc), (
            "Mastery should be lost after an incorrect answer"
        )


# ===========================================================================
# 5.  Effective prior under prerequisite conditioning
# ===========================================================================

class TestEffectivePrior:

    def test_no_prereqs_returns_l0(self):
        state, kc = fresh_state()
        ep = state.effective_prior(kc, prereqs=[], mastered_set=set())
        assert ep == pytest.approx(0.20)

    def test_all_prereqs_mastered_returns_close_to_l0(self):
        """When all prereqs are mastered, effective_prior = L0 * (0.3 + 0.7*1) = L0."""
        state, kc = fresh_state()
        ep = state.effective_prior(kc, prereqs=["PRE1", "PRE2"],
                                   mastered_set={"PRE1", "PRE2"})
        # ALPHA=0.3 → effective = L0 * (0.3 + 0.7*1.0) = L0 * 1.0 = 0.20
        assert ep == pytest.approx(0.20, rel=1e-6)

    def test_no_prereqs_mastered_returns_reduced_prior(self):
        """
        With 0/2 prereqs mastered:
            r = 0/2 = 0.0
            effective = L0 * (0.3 + 0.7*0.0) = 0.20 * 0.3 = 0.06
        """
        state, kc = fresh_state()
        ep = state.effective_prior(kc, prereqs=["PRE1", "PRE2"],
                                   mastered_set=set())
        # 0.20 * 0.3 = 0.06
        assert ep == pytest.approx(0.06, rel=1e-6)

    def test_partial_prereqs_mastered(self):
        """
        With 1/2 prereqs mastered:
            r = 0.5
            effective = 0.20 * (0.3 + 0.7*0.5) = 0.20 * 0.65 = 0.13
        """
        state, kc = fresh_state()
        ep = state.effective_prior(kc, prereqs=["PRE1", "PRE2"],
                                   mastered_set={"PRE1"})
        assert ep == pytest.approx(0.13, rel=1e-6)

    def test_effective_prior_always_in_unit_interval(self):
        """Sanity: the result must always be in (0, 1)."""
        state, kc = fresh_state()
        for n_mastered in range(4):
            mastered_set = {f"P{i}" for i in range(n_mastered)}
            prereqs = [f"P{i}" for i in range(3)]
            ep = state.effective_prior(kc, prereqs=prereqs,
                                       mastered_set=mastered_set)
            assert 0.0 < ep < 1.0


# ===========================================================================
# 6.  Multi-KC independence
# ===========================================================================

class TestMultiKCIndependence:
    """
    Updating one KC must not affect another KC's state.
    """

    def test_kcs_are_independent(self):
        state = BKTLearnerState()
        state.register("ALGEBRA")
        state.register("GEOMETRY")

        p_geo_before = state.p_mastered("GEOMETRY")
        state.observe("ALGEBRA", correct=True)
        p_geo_after = state.p_mastered("GEOMETRY")

        assert p_geo_before == p_geo_after, (
            "Observing ALGEBRA should not change GEOMETRY's p"
        )

    def test_custom_params_per_kc(self):
        """Different KCs can have different BKTParams."""
        state = BKTLearnerState()
        easy_params = BKTParams(l0=0.5, t=0.5, s=0.05, g=0.15)
        hard_params = BKTParams(l0=0.1, t=0.1, s=0.15, g=0.10)

        state.register("EASY", easy_params)
        state.register("HARD", hard_params)

        p_easy = state.observe("EASY", correct=True)
        p_hard = state.observe("HARD", correct=True)

        # Easy KC should have a higher p after one correct answer
        assert p_easy > p_hard


# ===========================================================================
# 7.  Monotonicity and boundary sanity
# ===========================================================================

class TestMonotonicity:

    def test_p_never_exceeds_1(self):
        state, kc = fresh_state()
        for _ in range(20):
            p = state.observe(kc, correct=True)
            assert p <= 1.0, f"p exceeded 1.0: {p}"

    def test_p_never_below_0(self):
        state, kc = fresh_state()
        for _ in range(20):
            p = state.observe(kc, correct=False)
            assert p >= 0.0, f"p dropped below 0.0: {p}"

    def test_more_correct_answers_monotonically_increase_p(self):
        """
        Consecutive correct answers should monotonically increase p.
        """
        state, kc = fresh_state()
        prev_p = state.p_mastered(kc)
        for _ in range(10):
            new_p = state.observe(kc, correct=True)
            assert new_p >= prev_p, (
                f"p decreased from {prev_p:.4f} to {new_p:.4f} on a correct answer"
            )
            prev_p = new_p
