"""
mathtutor/learner/scheduling.py
================================
Half-Life Regression (HLR) style forgetting model and spaced-repetition
scheduler.

References
----------
Settles, B. & Meeder, B. (2016). A Trainable Spaced Repetition Model for
Language Learning. ACL 2016.

SPEC §7.4–7.5 — Forgetting + Scheduling

Forgetting model
----------------
Recall is modelled as exponential decay:

    p(t) = 2 ** (-elapsed / h)

where:
  - elapsed  = now_ts - last_seen_ts  (same time unit; seconds recommended)
  - h        = half_life              (same unit; at elapsed=h recall = 0.50)

Half-life grows with successful, *spaced* retrievals (the "spacing" part of
HLR): if the student answered correctly AND the elapsed time is at least
MIN_SPACING_RATIO * current half_life, the half_life is multiplied by
GROWTH_FACTOR.  On failure, the half_life is multiplied by DECAY_FACTOR
(shrunk) and floored at MIN_HALF_LIFE.

Scheduling policy
-----------------
1. Mastery gate  — a KC is never selected unless all its prerequisites appear
   in mastered_set.
2. Spacing       — schedule for review when predicted recall ≤ review_band
   (default 0.85 — the "desirable difficulty" sweet spot: still mostly
   remembered, but retrieval effort drives consolidation).
3. Interleaving  — `select_next` mixes due-review KCs with new (unseen/
   unmastered) KCs in a round-robin pattern so no single KC dominates a
   session (the "interleaving effect").

DESIGN INVARIANTS
-----------------
* Pure functions — no global mutable state.  `update_after_review` returns a
  *new* `RetentionState` rather than mutating in-place.
* Half-life is always a positive float; this is enforced in
  `update_after_review`.
* `predicted_recall` returns a float in (0, 1].  At elapsed=0 it is exactly
  1.0; it approaches 0 asymptotically but never reaches it.
* `confidence == 1.0` only when `decidable is True` (BKT invariant from the
  rest of the system; scheduling doesn't produce Judgment objects but follows
  the same honesty policy: never claim certainty we don't have).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

# ---------------------------------------------------------------------------
# Imports from sibling modules (do not redefine)
# ---------------------------------------------------------------------------
from mathtutor.learner.bkt import BKTLearnerState          # noqa: F401 (used by callers)
from mathtutor.domain.curriculum import Curriculum


# ---------------------------------------------------------------------------
# Hyperparameters (module-level constants, not global mutable state)
# ---------------------------------------------------------------------------

#: Half-life grows by this factor after a spaced successful retrieval.
GROWTH_FACTOR: float = 2.0

#: Half-life shrinks by this factor after a failed retrieval.
DECAY_FACTOR: float = 0.5

#: Minimum half-life floor (prevents collapse to 0).
MIN_HALF_LIFE: float = 1.0   # same unit as timestamps (e.g. seconds)

#: Default initial half-life for an unseen KC.
DEFAULT_HALF_LIFE: float = 86_400.0   # 1 day in seconds

#: A retrieval counts as "spaced" only if elapsed ≥ this fraction of h.
#: Immediately re-reading something does not earn a half-life boost.
MIN_SPACING_RATIO: float = 0.10


# ---------------------------------------------------------------------------
# RetentionState
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetentionState:
    """
    Per-knowledge-component forgetting state.

    Attributes
    ----------
    half_life : float
        Current memory half-life in the same unit as timestamps (e.g. seconds).
        A larger value means the student forgets more slowly.
    last_seen_ts : float
        Unix timestamp (or any monotonic counter) of the most recent review
        of this KC.
    successful_reviews : int
        Cumulative count of successful (correct) retrievals for this KC.
        Used for diagnostics and can gate more aggressive half-life growth in
        future versions.
    """

    half_life: float = DEFAULT_HALF_LIFE
    last_seen_ts: float = 0.0
    successful_reviews: int = 0

    def __post_init__(self) -> None:
        if self.half_life <= 0.0:
            raise ValueError(f"half_life must be > 0, got {self.half_life!r}")
        if self.successful_reviews < 0:
            raise ValueError(
                f"successful_reviews must be >= 0, got {self.successful_reviews!r}"
            )


# ---------------------------------------------------------------------------
# Core recall formula
# ---------------------------------------------------------------------------

def predicted_recall(rs: RetentionState, now_ts: float) -> float:
    """
    Predict the probability of successful free recall at time ``now_ts``.

    Formula
    -------
    ::

        p = 2 ** (-(now_ts - last_seen_ts) / half_life)

    Step-by-step example
    --------------------
    Suppose ``half_life = 86_400`` (one day in seconds) and the student last
    reviewed 12 hours ago (elapsed = 43_200 s):

        p = 2 ** (-43_200 / 86_400)
          = 2 ** (-0.5)
          ≈ 0.707

    So after half a half-life the recall is ≈ 71 %.

    Parameters
    ----------
    rs : RetentionState
        Current retention state for the KC.
    now_ts : float
        Current timestamp (same unit as ``rs.last_seen_ts``).

    Returns
    -------
    float
        Predicted recall probability in (0, 1].  Returns 1.0 when
        ``now_ts <= rs.last_seen_ts`` (elapsed ≤ 0).
    """
    elapsed = now_ts - rs.last_seen_ts
    if elapsed <= 0.0:
        return 1.0
    return 2.0 ** (-elapsed / rs.half_life)


# ---------------------------------------------------------------------------
# State update after a review attempt
# ---------------------------------------------------------------------------

def update_after_review(
    rs: RetentionState,
    now_ts: float,
    success: bool,
) -> RetentionState:
    """
    Return a new ``RetentionState`` incorporating the outcome of a review.

    Growth rule (success)
    ---------------------
    A successful retrieval grows the half-life *only* if the retrieval was
    sufficiently spaced (elapsed ≥ MIN_SPACING_RATIO × h).  Massed (cramming)
    repetitions are detected and do not earn a bonus.

    ::

        if elapsed >= MIN_SPACING_RATIO * h:
            new_h = h * GROWTH_FACTOR      # e.g. h → 2h
        else:
            new_h = h                       # no reward for cramming

    Decay rule (failure)
    --------------------
    ::

        new_h = max(h * DECAY_FACTOR, MIN_HALF_LIFE)   # e.g. h → h/2, ≥ 1

    Parameters
    ----------
    rs : RetentionState
        Current state.
    now_ts : float
        Timestamp of this review event.
    success : bool
        True if the student recalled correctly, False otherwise.

    Returns
    -------
    RetentionState
        New (immutable) state.  The original is never mutated.
    """
    elapsed = max(now_ts - rs.last_seen_ts, 0.0)

    if success:
        spaced = elapsed >= MIN_SPACING_RATIO * rs.half_life
        new_h = rs.half_life * GROWTH_FACTOR if spaced else rs.half_life
        return replace(
            rs,
            half_life=new_h,
            last_seen_ts=now_ts,
            successful_reviews=rs.successful_reviews + 1,
        )
    else:
        new_h = max(rs.half_life * DECAY_FACTOR, MIN_HALF_LIFE)
        return replace(
            rs,
            half_life=new_h,
            last_seen_ts=now_ts,
            # successful_reviews unchanged — failure doesn't reset the count
        )


# ---------------------------------------------------------------------------
# Which KCs are due for review?
# ---------------------------------------------------------------------------

def due_for_review(
    states: Mapping[str, RetentionState],
    now_ts: float,
    band: float = 0.85,
) -> list[str]:
    """
    Return KC ids whose predicted recall has dropped to or below ``band``.

    The default band of 0.85 targets the "desirable difficulty" sweet spot:
    the student still mostly remembers the material, but the retrieval
    effort is high enough to drive long-term consolidation.

    Parameters
    ----------
    states : Mapping[str, RetentionState]
        Map from kc_id → RetentionState for every KC the learner has seen.
        KC ids not present in this mapping are considered unseen and are
        *not* returned (they belong to ``select_next``'s "new KC" pool).
    now_ts : float
        Current timestamp.
    band : float
        Recall threshold.  KCs with recall ≤ band are returned.
        Default 0.85.

    Returns
    -------
    list[str]
        KC ids that are due, sorted by ascending recall (most-forgotten first)
        so callers can prioritise the KC most at risk of being lost.
    """
    due: list[tuple[float, str]] = []
    for kc_id, rs in states.items():
        recall = predicted_recall(rs, now_ts)
        if recall <= band + 1e-9:
            due.append((recall, kc_id))
    # Most-forgotten first (lowest recall = highest urgency)
    due.sort(key=lambda t: t[0])
    return [kc_id for _, kc_id in due]


# ---------------------------------------------------------------------------
# Main scheduler
# ---------------------------------------------------------------------------

def select_next(
    curriculum: Curriculum,
    mastered_set: set[str],
    retention_states: Mapping[str, RetentionState],
    now_ts: float,
    k: int = 5,
) -> list[str]:
    """
    Choose up to ``k`` KC ids for the next session, interleaved.

    Algorithm
    ---------
    Step 1 — Prerequisite filter
        Collect *all* candidate KCs from the curriculum.  A KC is a candidate
        only if every KC in its ``prerequisites`` list is in ``mastered_set``.
        Already-mastered KCs are excluded from the "new" pool (they may still
        appear in the due-review pool).

    Step 2 — Split into two pools
        a. **Review pool** — candidates whose predicted recall ≤ ``band``
           (default 0.85) *and* that appear in ``retention_states`` (i.e. the
           learner has seen them before).  Sorted most-forgotten first.
        b. **New pool** — candidates not yet in ``retention_states`` (never
           seen), sorted by curriculum order (shallow first).

    Step 3 — Interleave
        Alternate: one review KC, one new KC, one review KC, …, until ``k``
        slots are filled or both pools are exhausted.  If one pool runs out,
        continue drawing from the other.

    This interleaving prevents "blocking" (spending the whole session on one
    KC) and mixes retrieval practice with new learning.

    Parameters
    ----------
    curriculum : Curriculum
        The prerequisite DAG.  Expected to have a ``kcs`` attribute that is
        an iterable of objects with ``.id`` (str) and ``.prerequisites``
        (list[str]).
    mastered_set : set[str]
        KC ids the learner has mastered (``bkt.mastered(kc_id)`` returned
        True at some point).
    retention_states : Mapping[str, RetentionState]
        Current forgetting state for every KC the learner has encountered.
    now_ts : float
        Current timestamp.
    k : int
        Maximum number of KCs to return.

    Returns
    -------
    list[str]
        Interleaved KC ids, length ≤ k.  Never contains a KC whose
        prerequisites are not all in ``mastered_set``.

    Raises
    ------
    ValueError
        If k < 1.
    """
    if k < 1:
        raise ValueError(f"k must be ≥ 1, got {k!r}")

    REVIEW_BAND = 0.85

    # ------------------------------------------------------------------
    # Step 1 — build the candidate set (prereqs met, not mastered)
    # ------------------------------------------------------------------
    all_kc_ids_in_order: list[str] = [kc.id for kc in curriculum.kcs]
    prereqs_by_id: dict[str, list[str]] = {
        kc.id: list(kc.prerequisites) for kc in curriculum.kcs
    }

    def prereqs_met(kc_id: str) -> bool:
        return all(p in mastered_set for p in prereqs_by_id.get(kc_id, []))

    # ------------------------------------------------------------------
    # Step 2 — split into review vs new pools
    # ------------------------------------------------------------------
    review_pool: list[tuple[float, str]] = []  # (recall, kc_id)
    new_pool: list[str] = []                    # curriculum-ordered

    for kc_id in all_kc_ids_in_order:
        if not prereqs_met(kc_id):
            continue  # blocked by unmet prereqs — never select

        if kc_id in retention_states:
            # Seen before — eligible for review if recall ≤ band
            recall = predicted_recall(retention_states[kc_id], now_ts)
            if recall <= REVIEW_BAND:
                review_pool.append((recall, kc_id))
        elif kc_id not in mastered_set:
            # Never seen and not mastered — eligible as a new KC
            new_pool.append(kc_id)

    # Sort review pool: most-forgotten first
    review_pool.sort(key=lambda t: t[0])
    review_ids = [kc_id for _, kc_id in review_pool]

    # ------------------------------------------------------------------
    # Step 3 — interleave review and new KCs (round-robin)
    # ------------------------------------------------------------------
    result: list[str] = []
    ri, ni = 0, 0
    # Alternate: review, new, review, new, …
    take_review = True   # start with review if available, else new

    while len(result) < k and (ri < len(review_ids) or ni < len(new_pool)):
        if take_review and ri < len(review_ids):
            result.append(review_ids[ri])
            ri += 1
        elif not take_review and ni < len(new_pool):
            result.append(new_pool[ni])
            ni += 1
        elif ri < len(review_ids):
            # new pool exhausted — drain review
            result.append(review_ids[ri])
            ri += 1
        else:
            # review pool exhausted — drain new
            result.append(new_pool[ni])
            ni += 1
        take_review = not take_review   # flip for next slot

    return result
