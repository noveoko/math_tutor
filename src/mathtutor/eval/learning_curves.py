# mathtutor/eval/learning_curves.py

"""Learning-curve analysis based on the Additive Factor Model (AFM).

Background — Additive Factor Model
------------------------------------
The AFM (Cen, Koedinger & Junker 2006) models the probability that a
student *s* answers a problem requiring knowledge component *kc* correctly
on their *n*-th opportunity (0-indexed) as::

    P(correct | s, kc, n) = σ( β_s + δ_kc + γ_kc · n )

where

* σ(z) = 1 / (1 + exp(−z))  is the logistic function
* β_s   is the **student ability** intercept (one per student)
* δ_kc  is the **KC easiness** intercept (one per KC; higher ⇒ easier)
* γ_kc  is the **learning rate** for KC (one per KC; should be > 0 if
         practice helps)

Fitting is done by minimising binary cross-entropy with L-BFGS-B via
``scipy.optimize.minimize``.  Parameters are initialised to zero.

Interpretation
--------------
* A *well-specified* KC shows a smooth, monotone **decline** in *error*
  rate (= 1 − P(correct)) as opportunity index increases.
* A *flat or noisy* curve suggests the KC conflates sub-skills and should
  be **split**.

Learning-gain summary
---------------------
Hake's normalised gain (Hake 1998)::

    g = (post − pre) / (100 − pre)

ranges from 0 (no gain) to 1 (perfect gain).  Guard: if pre == 100,
gain is undefined (ceiling); we return ``float('nan')``.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from typing import Sequence

import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize
    _SCIPY = True
except ImportError:                     # pragma: no cover
    _SCIPY = False

from mathtutor.contracts import TelemetryEvent


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sigma(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function σ(z) = 1/(1+e^{-z})."""
    # np.where avoids overflow in exp for large |z|
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def _bce_loss_and_grad(
    params: np.ndarray,
    y: np.ndarray,
    student_idx: np.ndarray,
    kc_idx: np.ndarray,
    opp: np.ndarray,
    n_students: int,
    n_kcs: int,
) -> tuple[float, np.ndarray]:
    """Binary cross-entropy loss + gradient for AFM."""
    beta  = params[:n_students]                        # (S,)
    delta = params[n_students : n_students + n_kcs]    # (K,)
    gamma = params[n_students + n_kcs :]               # (K,)

    # Logit for each observation
    z = beta[student_idx] + delta[kc_idx] + gamma[kc_idx] * opp   # (N,)
    p = _sigma(z)                                                 # (N,)

    # ---- Loss: mean binary cross-entropy ----
    # Numerically stable BCE from logits avoids eps capping issues
    loss_vec = np.maximum(z, 0) - y * z + np.log1p(np.exp(-np.abs(z)))
    loss = float(np.mean(loss_vec))

    # ---- Gradient ----
    # ∂L/∂z_i = (p_i − y_i) / N
    residual = (p - y) / len(y)                                   # (N,)

    grad_beta  = np.zeros(n_students)
    grad_delta = np.zeros(n_kcs)
    grad_gamma = np.zeros(n_kcs)

    np.add.at(grad_beta,  student_idx, residual)
    np.add.at(grad_delta, kc_idx,      residual)
    np.add.at(grad_gamma, kc_idx,      residual * opp)

    grad = np.concatenate([grad_beta, grad_delta, grad_gamma])
    return loss, grad


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_afm(events: Sequence[TelemetryEvent]) -> dict:
    """Fit the Additive Factor Model to a sequence of telemetry events.

    Only events whose ``verdict`` field is ``"correct"`` or ``"wrong"``
    (case-insensitive) *and* whose ``kc_id`` is non-null contribute to the
    fit.  All other events are silently ignored.

    Parameters
    ----------
    events:
        Flat sequence of :class:`~mathtutor.contracts.TelemetryEvent`.

    Returns
    -------
    dict with keys:

    ``"kc_easiness"`` : dict[str, float]
        δ_kc for every KC that appeared in the data.  Higher ⇒ easier.
    ``"kc_learning_rate"`` : dict[str, float]
        γ_kc for every KC.  Positive means practice helps.
    ``"student_ability"`` : dict[str, float]
        β_s for every student pseudonym.
    ``"converged"`` : bool
        Whether the optimiser reported successful convergence.
    ``"n_obs"`` : int
        Number of observations used in the fit.

    Notes
    -----
    With zero or one observations the model cannot be identified; an
    empty/trivial result is returned with ``converged=False``.
    """
    # ---- Collect observations ----
    student_ids: list[str] = []
    kc_ids_obs:  list[str] = []
    opps:        list[int] = []
    labels:      list[int] = []

    for ev in events:
        if ev.kc_id is None:
            continue
        v = (ev.verdict or "").lower()
        if v not in ("correct", "wrong"):
            continue
        student_ids.append(ev.user_pseudonym)
        kc_ids_obs.append(ev.kc_id)
        opps.append(int(ev.opportunity_index))
        labels.append(1 if v == "correct" else 0)

    n_obs = len(labels)
    empty: dict = {
        "kc_easiness": {}, "kc_learning_rate": {},
        "student_ability": {}, "converged": False, "n_obs": 0,
    }
    if n_obs < 2:
        return empty

    # ---- Build index maps ----
    s_vocab  = {s: i for i, s in enumerate(sorted(set(student_ids)))}
    kc_vocab = {k: i for i, k in enumerate(sorted(set(kc_ids_obs)))}
    n_s, n_k = len(s_vocab), len(kc_vocab)

    s_arr   = np.array([s_vocab[s]  for s in student_ids], dtype=np.int32)
    kc_arr  = np.array([kc_vocab[k] for k in kc_ids_obs],  dtype=np.int32)
    opp_arr = np.array(opps,                                dtype=np.float64)
    y_arr   = np.array(labels,                              dtype=np.float64)

    # ---- Optimise ----
    n_params = n_s + 2 * n_k
    x0 = np.zeros(n_params)

    converged = False
    params = x0.copy()

    if _SCIPY:
        result = _scipy_minimize(
            fun=_bce_loss_and_grad,
            x0=x0,
            method="L-BFGS-B",
            jac=True,
            args=(y_arr, s_arr, kc_arr, opp_arr, n_s, n_k),
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        params    = result.x
        converged = bool(result.success)
    else:
        # Fallback: simple gradient descent (no scipy)        # pragma: no cover
        lr_opt = 0.1
        for _ in range(2000):
            _, grad = _bce_loss_and_grad(
                params, y_arr, s_arr, kc_arr, opp_arr, n_s, n_k)
            params -= lr_opt * grad
            if np.max(np.abs(grad)) < 1e-6:
                converged = True
                break

    # ---- Unpack ----
    beta  = params[:n_s]
    delta = params[n_s : n_s + n_k]
    gamma = params[n_s + n_k :]

    inv_s  = {i: s for s, i in s_vocab.items()}
    inv_kc = {i: k for k, i in kc_vocab.items()}

    return {
        "kc_easiness":      {inv_kc[i]: float(delta[i]) for i in range(n_k)},
        "kc_learning_rate": {inv_kc[i]: float(gamma[i]) for i in range(n_k)},
        "student_ability":  {inv_s[i]:  float(beta[i])  for i in range(n_s)},
        "converged":        converged,
        "n_obs":            n_obs,
    }


def error_rate_curve(
    events: Sequence[TelemetryEvent],
    kc_id: str,
) -> list[float]:
    """Compute per-opportunity error rate for a single KC.

    Each position in the returned list corresponds to one opportunity
    index (0, 1, 2, …).  The value is the fraction of attempts at that
    opportunity that were *incorrect* (verdict == ``"wrong"``).

    Positions with no observations are set to ``float('nan')``.

    Parameters
    ----------
    events:
        All telemetry events (any KC; irrelevant ones are filtered out).
    kc_id:
        The knowledge component to analyse.

    Returns
    -------
    list[float]
        Error rates indexed by opportunity number.  Length equals
        ``max(opportunity_index) + 1`` across matching events, or ``[]``
        if no matching events exist.
    """
    # Bucket by opportunity index
    buckets: dict[int, list[int]] = defaultdict(list)
    for ev in events:
        if ev.kc_id != kc_id:
            continue
        v = (ev.verdict or "").lower()
        if v not in ("correct", "wrong"):
            continue
        buckets[int(ev.opportunity_index)].append(0 if v == "correct" else 1)

    if not buckets:
        return []

    max_opp = max(buckets)
    curve: list[float] = []
    for i in range(max_opp + 1):
        if i in buckets and buckets[i]:
            curve.append(float(np.mean(buckets[i])))
        else:
            curve.append(float("nan"))
    return curve


def flag_misspecified_kcs(
    events: Sequence[TelemetryEvent],
    *,
    min_decline: float = 0.0,
) -> list[str]:
    """Return KC IDs whose error-rate curves fail to decline monotonically.

    A KC is flagged when its error-rate curve is *not* non-increasing by
    at least *min_decline* per step on average.  The check:

    1. Compute the error-rate curve for each KC.
    2. Ignore positions where the rate is ``nan`` (no observations).
    3. Compute ``Δ[i] = curve[i] − curve[i−1]`` for consecutive
       non-nan positions.
    4. A KC is **well-specified** if ``mean(Δ) ≤ −min_decline``
       (average step goes down).
    5. A KC is **flagged** (mis-specified / should be split) if
       ``mean(Δ) > −min_decline`` OR the curve has fewer than 2
       non-nan points (can't judge).

    Parameters
    ----------
    events:
        Telemetry events.
    min_decline:
        Minimum average per-step decline required to avoid flagging.
        Default ``0.0`` means *any* non-increasing average passes;
        raise it (e.g. ``0.01``) for stricter criteria.

    Returns
    -------
    list[str]
        Sorted list of flagged KC IDs.
    """
    # Gather all unique KC ids that have scoreable events
    kc_ids = {
        ev.kc_id for ev in events
        if ev.kc_id is not None
        and (ev.verdict or "").lower() in ("correct", "wrong")
    }

    flagged: list[str] = []
    for kc_id in sorted(kc_ids):
        curve = error_rate_curve(events, kc_id)

        # Keep only non-nan consecutive pairs
        non_nan = [(i, v) for i, v in enumerate(curve) if not math.isnan(v)]
        if len(non_nan) < 2:
            # Not enough data to assess — conservatively flag it
            flagged.append(kc_id)
            continue

        deltas = [
            non_nan[j][1] - non_nan[j - 1][1]
            for j in range(1, len(non_nan))
        ]
        # Non-negative mean delta => curve is flat or rising => mis-specified.
        # We require mean(Δ) < -min_decline for the KC to be considered
        # well-specified.  A flat curve (mean == 0) is flagged by default
        # because it provides no learning signal.
        if float(np.mean(deltas)) >= -min_decline:
            flagged.append(kc_id)

    return flagged


def normalized_gain(pre: float, post: float) -> float:
    """Compute Hake's normalised gain.

    Formula::

        g = (post − pre) / (100 − pre)

    Parameters
    ----------
    pre:
        Pre-test score as a percentage in [0, 100].
    post:
        Post-test score as a percentage in [0, 100].

    Returns
    -------
    float
        Normalised gain *g*.  Returns ``float('nan')`` when ``pre == 100``
        (ceiling effect; gain is undefined).

    Examples
    --------
    >>> normalized_gain(40, 70)
    0.5
    >>> import math; math.isnan(normalized_gain(100, 100))
    True
    """
    if pre == 100.0:
        return float("nan")
    return (post - pre) / (100.0 - pre)