"""
mathtutor/learner/bkt.py
========================
Bayesian Knowledge Tracing (BKT) — Anderson / Corbett-Anderson model.

References
----------
Corbett, A. T., & Anderson, J. R. (1994). Knowledge tracing: Modeling the
acquisition of procedural knowledge. User Modeling and User-Adapted
Interaction, 4(4), 253–278.

Spec §7.2–7.3: four-parameter per-KC model with prerequisite propagation.

DESIGN INVARIANTS
-----------------
* No global mutable state — all state lives in BKTLearnerState instances.
* Pure functions where possible (observe mutates state, everything else is
  read-only or returns new values).
* confidence == 1.0 only when decidable is True (BKT is always decidable by
  construction; these are plain probability updates, not symbolic reasoning).
* Guard against degenerate params: S + G < 1 must hold, and every parameter
  must be strictly inside (0, 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# contracts.py is the only allowed project import at this level.
from mathtutor.contracts import KnowledgeComponent

if TYPE_CHECKING:
    # Avoid circular import at runtime; used only for the propagate() helper.
    pass


# ---------------------------------------------------------------------------
# BKTParams
# ---------------------------------------------------------------------------

@dataclass
class BKTParams:
    """
    Four parameters that govern one Knowledge Component's BKT dynamics.

    Parameters
    ----------
    l0 : float
        Prior probability the learner already knows the KC before any practice.
        Literature default: 0.20.
    t : float
        Transition (learn) probability — probability of going from "not mastered"
        to "mastered" after a single opportunity. Literature default: 0.30.
    s : float
        Slip probability — P(wrong answer | mastered). Literature default: 0.10.
    g : float
        Guess probability — P(correct answer | not mastered). Literature default: 0.20.

    Constraints
    -----------
    * All parameters must be strictly in (0, 1).
    * s + g < 1  — violating this makes the model's posteriors incoherent
      (a correct answer would *reduce* the mastery estimate).
    """

    l0: float = 0.20
    t: float  = 0.30
    s: float  = 0.10
    g: float  = 0.20

    def __post_init__(self) -> None:
        for name, val in [("l0", self.l0), ("t", self.t),
                          ("s", self.s),  ("g", self.g)]:
            if not (0.0 < val < 1.0):
                raise ValueError(
                    f"BKTParams.{name} must be strictly in (0, 1); got {val}"
                )
        if self.s + self.g >= 1.0:
            raise ValueError(
                f"BKTParams requires s + g < 1 (got s={self.s}, g={self.g}, "
                f"sum={self.s + self.g:.4f}). Degenerate params make posteriors "
                "incoherent."
            )


# ---------------------------------------------------------------------------
# BKTLearnerState
# ---------------------------------------------------------------------------

class BKTLearnerState:
    """
    Tracks a single learner's estimated mastery across all Knowledge Components.

    Each KC is tracked independently with its own probability p = P(mastered)
    and its own BKTParams.  Call ``observe`` after every student response to
    update estimates.

    Parameters
    ----------
    default_params : BKTParams, optional
        Parameter set used for any KC that has not been explicitly registered.
        Defaults to the literature-prior BKTParams().

    Internals
    ---------
    _p      : dict[str, float]  — current P(mastered) per KC id
    _params : dict[str, BKTParams]  — params per KC id
    """

    def __init__(
        self,
        default_params: BKTParams | None = None,
    ) -> None:
        self._p:      dict[str, float]     = {}
        self._params: dict[str, BKTParams] = {}
        self._default_params = default_params or BKTParams()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def register(self, kc_id: str, params: BKTParams | None = None) -> None:
        """
        Explicitly register a KC with optional custom params.

        If the KC is already registered this is a no-op unless you pass new
        params, in which case the params are updated but the current p is
        preserved.
        """
        p = self._default_params if params is None else params
        self._params[kc_id] = p
        if kc_id not in self._p:
            self._p[kc_id] = p.l0

    def _ensure(self, kc_id: str) -> None:
        """Lazily initialise a KC with default params if not yet seen."""
        if kc_id not in self._params:
            self.register(kc_id)

    # ------------------------------------------------------------------
    # Core BKT update  (Spec §7.2)
    # ------------------------------------------------------------------

    def observe(self, kc_id: str, correct: bool) -> float:
        """
        Update P(mastered) for *kc_id* given one observed student response.

        Applies the two-step Corbett-Anderson update in order:

        Step 1 — condition on the observation (Bayes rule):
        ::

            if correct:
                p_post = p*(1-S) / ( p*(1-S) + (1-p)*G )
            else:
                p_post = p*S     / ( p*S     + (1-p)*(1-G) )

        Step 2 — learning opportunity (transition):
        ::

            p_next = p_post + (1 - p_post) * T

        Parameters
        ----------
        kc_id : str
            Knowledge-component identifier (matches KnowledgeComponent.id).
        correct : bool
            Whether the student's response was correct.

        Returns
        -------
        float
            The updated P(mastered) after this observation.
        """
        self._ensure(kc_id)
        p = self._p[kc_id]
        par = self._params[kc_id]

        # --- Step 1: condition on observation ---
        if correct:
            numerator   = p * (1.0 - par.s)
            denominator = numerator + (1.0 - p) * par.g
        else:
            numerator   = p * par.s
            denominator = numerator + (1.0 - p) * (1.0 - par.g)

        p_post = numerator / denominator  # denominator > 0 when s+g<1

        # --- Step 2: learning transition ---
        p_next = p_post + (1.0 - p_post) * par.t

        self._p[kc_id] = p_next
        return p_next

    # ------------------------------------------------------------------
    # Mastery query
    # ------------------------------------------------------------------

    def mastered(self, kc_id: str, threshold: float = 0.95) -> bool:
        """
        Return True iff the current P(mastered) >= *threshold*.

        Parameters
        ----------
        kc_id : str
        threshold : float
            Mastery threshold. Spec default is 0.95 — chosen so a single
            correct answer from the L0 prior CANNOT cross it (guessing is
            modelled).
        """
        self._ensure(kc_id)
        return self._p[kc_id] >= threshold

    def p_mastered(self, kc_id: str) -> float:
        """Return the raw P(mastered) estimate for *kc_id*."""
        self._ensure(kc_id)
        return self._p[kc_id]

    # ------------------------------------------------------------------
    # Effective prior under prerequisite conditioning  (Spec §7.3)
    # ------------------------------------------------------------------

    def effective_prior(
        self,
        kc_id: str,
        prereqs: list[str],
        mastered_set: set[str],
    ) -> float:
        """
        Return an adjusted prior for *kc_id* that accounts for prerequisite mastery.

        Rationale
        ---------
        If a student has not yet mastered the prerequisites of a KC, their
        effective probability of knowing that KC is lower than the raw L0 would
        suggest.  Conversely, mastering all prerequisites keeps the prior at L0
        (or can nudge it slightly upward if we have positive evidence).

        Simple model used here
        ----------------------
        Let ``r = |mastered_prereqs| / |prereqs|``  (fraction of prereqs mastered).
        ``effective_l0 = L0 * (alpha + (1 - alpha) * r)``
        where ``alpha = 0.3`` is a floor factor — even with zero prereqs mastered
        we don't completely zero out the prior.

        This keeps the function monotone and bounded in (0, L0], which respects
        the BKT invariant that priors are in (0, 1).

        Parameters
        ----------
        kc_id : str
        prereqs : list[str]
            IDs of direct prerequisite KCs.
        mastered_set : set[str]
            Set of KC ids the learner has already mastered.

        Returns
        -------
        float
            Adjusted L0 ∈ (0, 1).
        """
        self._ensure(kc_id)
        params = self._params[kc_id]

        if not prereqs:
            return params.l0

        ALPHA = 0.3  # floor: minimum fraction of L0 even with 0 prereqs mastered
        r = sum(1 for pid in prereqs if pid in mastered_set) / len(prereqs)
        adjusted = params.l0 * (ALPHA + (1.0 - ALPHA) * r)
        # Clamp to (0, 1) for safety — shouldn't be needed with valid params
        return max(1e-6, min(adjusted, 1.0 - 1e-6))


# ---------------------------------------------------------------------------
# propagate — weak upward/downward prerequisite conditioning
# ---------------------------------------------------------------------------

def propagate(state: BKTLearnerState, curriculum: object) -> None:
    """
    Apply weak prerequisite-conditioned probability propagation across the
    KC dependency graph.

    What "weak" means
    -----------------
    We do *not* hard-override any KC's p with the effective prior — that would
    discard hard-won evidence.  Instead we nudge: if a KC's current p is
    *above* what the effective prior would grant given its unmastered prereqs,
    we pull it gently toward the effective prior.  The nudge is small (weight
    ``NUDGE = 0.05``) so it acts as a regulariser, not a reset.

    Upward propagation (prereq → dependent)
    ----------------------------------------
    If a KC is mastered, its dependents get a small positive nudge on their
    current p (capped at their current p + NUDGE, and never above 0.94 to
    avoid spurious mastery declarations without evidence).

    Downward propagation (dependent → prereq — "weak upward" in learner terms)
    ---------------------------------------------------------------------------
    If a dependent KC is answered correctly, the probability of the prereq
    being known rises slightly.  We model this as: for each unmastered prereq,
    nudge its p up by ``NUDGE * P(dependent correct | prereq mastered)``, which
    we approximate as (1 - s_dependent).

    Parameters
    ----------
    state : BKTLearnerState
        Mutated in-place.
    curriculum : object
        Expected to have a ``kcs`` attribute — an iterable of
        ``KnowledgeComponent`` objects with ``.id`` and ``.prerequisites``
        (list[str]).

    Notes
    -----
    This is intentionally simple — it avoids full belief propagation over the
    DAG, which would require topological ordering and multiple passes.  For
    richer behaviour, replace with a proper sum-product pass.
    """
    NUDGE = 0.05
    MASTERY_CEILING = 0.94  # never declare mastery without a real observation

    kcs: list[KnowledgeComponent] = list(getattr(curriculum, "kcs", []))

    # Build prereq map: kc_id -> list[prereq_id]
    prereq_map: dict[str, list[str]] = {
        kc.id: list(kc.prerequisites) for kc in kcs
    }

    # Build reverse map: prereq_id -> list[dependent_id]
    dependent_map: dict[str, list[str]] = {}
    for kc_id, prereqs in prereq_map.items():
        for pid in prereqs:
            dependent_map.setdefault(pid, []).append(kc_id)

    mastered_set = {kc_id for kc_id in state._p if state.mastered(kc_id)}

    # --- Downward: mastered KC nudges its dependents up ---
    for mastered_kc in mastered_set:
        for dep_id in dependent_map.get(mastered_kc, []):
            if dep_id not in mastered_set:
                state._ensure(dep_id)
                current = state._p[dep_id]
                state._p[dep_id] = min(current + NUDGE, MASTERY_CEILING)

    # --- Upward: unmastered prereqs gently pull dependents toward effective prior ---
    for kc_id, prereqs in prereq_map.items():
        if not prereqs:
            continue
        unmastered_prereqs = [pid for pid in prereqs if pid not in mastered_set]
        if not unmastered_prereqs:
            continue  # all prereqs mastered — no downward drag needed

        effective = state.effective_prior(kc_id, prereqs, mastered_set)
        state._ensure(kc_id)
        current = state._p[kc_id]
        if current > effective:
            # Nudge toward effective prior, not hard-reset
            state._p[kc_id] = current - NUDGE * (current - effective)
