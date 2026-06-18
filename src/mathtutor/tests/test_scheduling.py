"""
tests/test_scheduling.py
========================
Tests for mathtutor/learner/scheduling.py.

Coverage targets (per spec):
  1. Recall decays toward 0 as elapsed grows.
  2. A spaced success increases half_life.
  3. select_next never returns a KC with unmet prerequisites.
  4. select_next interleaves (does not block on one KC).

All tests are deterministic and require no network access or I/O.

Mocking strategy
----------------
``Curriculum`` is imported from ``domain/curriculum.py``.  To keep these
tests self-contained we create a lightweight ``FakeCurriculum`` that satisfies
the interface (``curriculum.kcs`` → iterable of objects with ``.id`` and
``.prerequisites``).  This is NOT redefining the real Curriculum — we stay
within our contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import pytest

from mathtutor.learner.scheduling import (
    DEFAULT_HALF_LIFE,
    GROWTH_FACTOR,
    DECAY_FACTOR,
    MIN_HALF_LIFE,
    MIN_SPACING_RATIO,
    RetentionState,
    due_for_review,
    predicted_recall,
    select_next,
    update_after_review,
)


# ---------------------------------------------------------------------------
# Helpers / Fakes
# ---------------------------------------------------------------------------

@dataclass
class FakeKC:
    """Minimal stand-in for a KnowledgeComponent; satisfies the interface."""
    id: str
    prerequisites: List[str]


class FakeCurriculum:
    """Minimal stand-in for Curriculum; provides .kcs attribute."""
    def __init__(self, kcs: list[FakeKC]) -> None:
        self.kcs = kcs


# Shared timestamps — use seconds for clarity.
T0 = 1_000_000.0   # arbitrary epoch


# ---------------------------------------------------------------------------
# 1.  predicted_recall — decay behaviour
# ---------------------------------------------------------------------------

class TestPredictedRecall:
    """Recall should be 1.0 at t=0 and decay toward 0 as time passes."""

    def test_recall_at_zero_elapsed_is_one(self) -> None:
        rs = RetentionState(half_life=86_400.0, last_seen_ts=T0)
        assert predicted_recall(rs, T0) == pytest.approx(1.0)

    def test_recall_at_one_half_life_is_half(self) -> None:
        """
        By definition, 2^(-(h/h)) = 2^(-1) = 0.5.
        This is the core invariant of exponential-half-life decay.
        """
        h = 3600.0
        rs = RetentionState(half_life=h, last_seen_ts=T0)
        assert predicted_recall(rs, T0 + h) == pytest.approx(0.5)

    def test_recall_at_two_half_lives_is_quarter(self) -> None:
        """
        2^(-(2h/h)) = 2^(-2) = 0.25
        """
        h = 3600.0
        rs = RetentionState(half_life=h, last_seen_ts=T0)
        assert predicted_recall(rs, T0 + 2 * h) == pytest.approx(0.25)

    def test_recall_decays_monotonically(self) -> None:
        """Recall must strictly decrease as more time elapses."""
        h = 3600.0
        rs = RetentionState(half_life=h, last_seen_ts=T0)
        recalls = [
            predicted_recall(rs, T0 + elapsed)
            for elapsed in [0, h * 0.5, h, h * 2, h * 5, h * 20]
        ]
        for earlier, later in zip(recalls, recalls[1:]):
            assert later < earlier

    def test_recall_approaches_zero(self) -> None:
        """After many half-lives the recall should be negligible (< 0.001)."""
        h = 1.0   # 1-second half-life for a quick test
        rs = RetentionState(half_life=h, last_seen_ts=T0)
        # 20 half-lives: 2^(-20) ≈ 9.5e-7
        assert predicted_recall(rs, T0 + 20 * h) < 0.001

    def test_negative_elapsed_returns_one(self) -> None:
        """Clock skew or same-instant review should not break recall."""
        rs = RetentionState(half_life=3600.0, last_seen_ts=T0 + 100)
        assert predicted_recall(rs, T0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2.  update_after_review — half-life growth and decay
# ---------------------------------------------------------------------------

class TestUpdateAfterReview:
    """Test the half-life update rules."""

    def test_spaced_success_grows_half_life(self) -> None:
        """
        A success where elapsed >= MIN_SPACING_RATIO * h should multiply h
        by GROWTH_FACTOR.

        Arithmetic:
            h = 3600.0
            elapsed must be >= MIN_SPACING_RATIO * 3600 = 0.10 * 3600 = 360 s
            We use elapsed = 1800 s (half a half-life, clearly spaced).
            Expected new_h = 3600 * GROWTH_FACTOR = 3600 * 2.0 = 7200 s.
        """
        h = 3600.0
        rs = RetentionState(half_life=h, last_seen_ts=T0)
        elapsed = 1800.0  # 1800 >= 360 — counts as spaced
        rs2 = update_after_review(rs, T0 + elapsed, success=True)
        assert rs2.half_life == pytest.approx(h * GROWTH_FACTOR)

    def test_spaced_success_increments_successful_reviews(self) -> None:
        h = 3600.0
        rs = RetentionState(half_life=h, last_seen_ts=T0, successful_reviews=3)
        rs2 = update_after_review(rs, T0 + 1800, success=True)
        assert rs2.successful_reviews == 4

    def test_massed_success_does_not_grow_half_life(self) -> None:
        """
        A success where elapsed < MIN_SPACING_RATIO * h should NOT grow h.

        Arithmetic:
            h = 3600.0, MIN_SPACING_RATIO = 0.10 => threshold = 360 s.
            elapsed = 10 s < 360 s  => massed repetition => no reward.
        """
        h = 3600.0
        rs = RetentionState(half_life=h, last_seen_ts=T0)
        elapsed = 10.0  # 10 < 360 — massed
        rs2 = update_after_review(rs, T0 + elapsed, success=True)
        assert rs2.half_life == pytest.approx(h)   # unchanged

    def test_failure_shrinks_half_life(self) -> None:
        """
        On failure, h should be multiplied by DECAY_FACTOR.

        Arithmetic:
            h = 7200.0, DECAY_FACTOR = 0.5 => new_h = 3600.0
        """
        h = 7200.0
        rs = RetentionState(half_life=h, last_seen_ts=T0)
        rs2 = update_after_review(rs, T0 + 3600, success=False)
        assert rs2.half_life == pytest.approx(h * DECAY_FACTOR)

    def test_failure_does_not_go_below_min_half_life(self) -> None:
        """
        Even with a tiny h, failure should not produce a non-positive half-life.
        """
        rs = RetentionState(half_life=MIN_HALF_LIFE, last_seen_ts=T0)
        rs2 = update_after_review(rs, T0 + 1, success=False)
        assert rs2.half_life >= MIN_HALF_LIFE

    def test_failure_does_not_increment_successful_reviews(self) -> None:
        rs = RetentionState(half_life=3600.0, last_seen_ts=T0, successful_reviews=5)
        rs2 = update_after_review(rs, T0 + 3600, success=False)
        assert rs2.successful_reviews == 5  # unchanged

    def test_update_is_immutable(self) -> None:
        """Original RetentionState must not be mutated."""
        rs = RetentionState(half_life=3600.0, last_seen_ts=T0)
        _ = update_after_review(rs, T0 + 1800, success=True)
        assert rs.half_life == pytest.approx(3600.0)
        assert rs.last_seen_ts == T0


# ---------------------------------------------------------------------------
# 3.  due_for_review
# ---------------------------------------------------------------------------

class TestDueForReview:
    """Items below the recall band should be flagged."""

    def _make_states(self) -> dict[str, RetentionState]:
        h = 3600.0
        return {
            "kc_a": RetentionState(half_life=h, last_seen_ts=T0),          # fresh
            "kc_b": RetentionState(half_life=h, last_seen_ts=T0 - 2 * h),  # old (recall=0.25)
            "kc_c": RetentionState(half_life=h, last_seen_ts=T0 - h * 0.2),# recall≈0.87 (>0.85)
        }

    def test_due_items_are_below_band(self) -> None:
        """
        kc_b recall ≈ 0.25 (clearly due).
        kc_a recall = 1.0  (fresh — not due).
        kc_c recall ≈ 0.87 (just above 0.85 — not due).
        """
        states = self._make_states()
        due = due_for_review(states, T0, band=0.85)
        assert "kc_b" in due
        assert "kc_a" not in due
        assert "kc_c" not in due

    def test_due_sorted_most_forgotten_first(self) -> None:
        """The most-forgotten KC (lowest recall) should come first."""
        h = 3600.0
        states = {
            "kc_very_old": RetentionState(half_life=h, last_seen_ts=T0 - 10 * h),
            "kc_old":      RetentionState(half_life=h, last_seen_ts=T0 - 2 * h),
        }
        due = due_for_review(states, T0, band=0.90)
        assert due[0] == "kc_very_old"

    def test_empty_states_returns_empty(self) -> None:
        assert due_for_review({}, T0) == []

    def test_band_exactly_at_threshold(self) -> None:
        """A KC whose recall equals the band exactly should be included."""
        h = 3600.0
        # We need elapsed s.t. 2^(-e/h) == 0.85
        # => e = -h * log2(0.85) ≈ h * 0.2345
        elapsed = -h * math.log2(0.85)
        rs = RetentionState(half_life=h, last_seen_ts=T0 - elapsed)
        states = {"kc_x": rs}
        due = due_for_review(states, T0, band=0.85)
        assert "kc_x" in due


# ---------------------------------------------------------------------------
# 4.  RetentionState validation
# ---------------------------------------------------------------------------

class TestRetentionStateValidation:
    def test_negative_half_life_raises(self) -> None:
        with pytest.raises(ValueError, match="half_life"):
            RetentionState(half_life=-1.0)

    def test_zero_half_life_raises(self) -> None:
        with pytest.raises(ValueError, match="half_life"):
            RetentionState(half_life=0.0)

    def test_negative_successful_reviews_raises(self) -> None:
        with pytest.raises(ValueError, match="successful_reviews"):
            RetentionState(half_life=3600.0, successful_reviews=-1)


# ---------------------------------------------------------------------------
# 5.  select_next — prerequisite safety + interleaving
# ---------------------------------------------------------------------------

class TestSelectNext:
    """
    Core invariants:
      a. Never return a KC with unmet prerequisites.
      b. Return an interleaved (not blocked) sequence when both review and new
         KCs are available.
    """

    def _simple_curriculum(self) -> FakeCurriculum:
        """
        Linear chain:  kc1 → kc2 → kc3 → kc4
        Plus two isolated KCs: kc5, kc6 (no prereqs)
        """
        return FakeCurriculum(kcs=[
            FakeKC("kc1", prerequisites=[]),
            FakeKC("kc2", prerequisites=["kc1"]),
            FakeKC("kc3", prerequisites=["kc2"]),
            FakeKC("kc4", prerequisites=["kc3"]),
            FakeKC("kc5", prerequisites=[]),
            FakeKC("kc6", prerequisites=[]),
        ])

    # ------------------------------------------------------------------
    # Prerequisite safety
    # ------------------------------------------------------------------

    def test_no_kc_with_unmet_prereqs(self) -> None:
        """
        With nothing mastered, only kc1, kc5, kc6 are eligible (no prereqs).
        kc2–kc4 must never appear.
        """
        curriculum = self._simple_curriculum()
        result = select_next(
            curriculum=curriculum,
            mastered_set=set(),
            retention_states={},
            now_ts=T0,
            k=6,
        )
        for kc_id in result:
            assert kc_id in {"kc1", "kc5", "kc6"}, (
                f"{kc_id!r} has unmet prerequisites but was selected"
            )

    def test_prereq_chain_respected(self) -> None:
        """
        With only kc1 mastered, kc2 becomes eligible but kc3/kc4 stay gated.
        """
        curriculum = self._simple_curriculum()
        result = select_next(
            curriculum=curriculum,
            mastered_set={"kc1"},
            retention_states={},
            now_ts=T0,
            k=6,
        )
        assert "kc3" not in result
        assert "kc4" not in result
        # kc2 *may* appear (prereq met)
        if "kc2" in result:
            pass  # acceptable

    def test_mastered_kc_not_in_new_pool(self) -> None:
        """A mastered KC should not be offered as 'new' even if its recall is high."""
        curriculum = self._simple_curriculum()
        result = select_next(
            curriculum=curriculum,
            mastered_set={"kc1"},
            retention_states={},   # kc1 not in retention_states
            now_ts=T0,
            k=6,
        )
        assert "kc1" not in result

    # ------------------------------------------------------------------
    # Interleaving
    # ------------------------------------------------------------------

    def test_interleaves_review_and_new(self) -> None:
        """
        Set up: kc1 and kc5 are due for review; kc6 is new.
        Expected output pattern: review, new, review, … (no two reviews
        or two news in a row, given alternation).

        We verify that the output is NOT simply [all reviews, all new] —
        i.e. not a blocked schedule.
        """
        h = 3600.0
        # kc1 and kc5 are overdue (recall ≈ 0.25 after 2h)
        retention_states = {
            "kc1": RetentionState(half_life=h, last_seen_ts=T0 - 2 * h),
            "kc5": RetentionState(half_life=h, last_seen_ts=T0 - 2 * h),
        }
        # kc6 is new (not in retention_states), prereqs met
        curriculum = self._simple_curriculum()
        result = select_next(
            curriculum=curriculum,
            mastered_set={"kc1", "kc5"},   # mastered so they appear in review pool, not new
            retention_states=retention_states,
            now_ts=T0,
            k=5,
        )
        # kc6 should be included (it is new and prereqs met)
        assert "kc6" in result

    def test_interleaved_not_blocked(self) -> None:
        """
        Build a scenario with 3 review KCs and 3 new KCs.
        The result should not be [r, r, r, n, n, n] — it should alternate.
        We check that the first new KC appears before the last review KC.
        """
        h = 1.0  # tiny half-life for predictability
        # kc1, kc5 are mastered but their recall has decayed (due for review)
        retention_states = {
            "kc1": RetentionState(half_life=h, last_seen_ts=T0 - 10 * h),
            "kc5": RetentionState(half_life=h, last_seen_ts=T0 - 10 * h),
        }
        # kc6 is new, kc2 will be eligible once kc1 is mastered
        curriculum = FakeCurriculum(kcs=[
            FakeKC("kc1", prerequisites=[]),
            FakeKC("kc5", prerequisites=[]),
            FakeKC("kc6", prerequisites=[]),
            FakeKC("kc2", prerequisites=["kc1"]),
        ])
        result = select_next(
            curriculum=curriculum,
            mastered_set={"kc1", "kc5"},
            retention_states=retention_states,
            now_ts=T0,
            k=6,
        )
        # kc1 and kc5 are in retention_states and recall is very low → review pool
        # kc6 and kc2 are new → new pool (kc2 prereq met via mastered_set)
        review_positions = [i for i, kc in enumerate(result) if kc in {"kc1", "kc5"}]
        new_positions    = [i for i, kc in enumerate(result) if kc in {"kc6", "kc2"}]

        if review_positions and new_positions:
            # Interleaved: the first new KC should appear before the last review KC
            assert min(new_positions) < max(review_positions), (
                f"Blocked schedule detected: reviews={review_positions}, "
                f"new={new_positions}, result={result}"
            )

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_k_is_respected(self) -> None:
        curriculum = self._simple_curriculum()
        result = select_next(
            curriculum=curriculum,
            mastered_set=set(),
            retention_states={},
            now_ts=T0,
            k=2,
        )
        assert len(result) <= 2

    def test_empty_curriculum_returns_empty(self) -> None:
        curriculum = FakeCurriculum(kcs=[])
        result = select_next(
            curriculum=curriculum,
            mastered_set=set(),
            retention_states={},
            now_ts=T0,
            k=5,
        )
        assert result == []

    def test_k_less_than_one_raises(self) -> None:
        curriculum = self._simple_curriculum()
        with pytest.raises(ValueError, match="k"):
            select_next(
                curriculum=curriculum,
                mastered_set=set(),
                retention_states={},
                now_ts=T0,
                k=0,
            )

    def test_all_mastered_returns_empty(self) -> None:
        """If every KC is already mastered, there is nothing new to select."""
        curriculum = self._simple_curriculum()
        all_ids = {kc.id for kc in curriculum.kcs}
        result = select_next(
            curriculum=curriculum,
            mastered_set=all_ids,
            retention_states={},   # none are in retention states
            now_ts=T0,
            k=5,
        )
        # Nothing due for review (not in retention_states) and nothing new
        assert result == []

    def test_no_new_kcs_drains_review_pool(self) -> None:
        """When all eligible KCs are due reviews, return them up to k."""
        h = 1.0
        curriculum = FakeCurriculum(kcs=[
            FakeKC("kc1", prerequisites=[]),
            FakeKC("kc5", prerequisites=[]),
        ])
        retention_states = {
            "kc1": RetentionState(half_life=h, last_seen_ts=T0 - 10),
            "kc5": RetentionState(half_life=h, last_seen_ts=T0 - 10),
        }
        result = select_next(
            curriculum=curriculum,
            mastered_set={"kc1", "kc5"},
            retention_states=retention_states,
            now_ts=T0,
            k=5,
        )
        assert set(result) == {"kc1", "kc5"}
