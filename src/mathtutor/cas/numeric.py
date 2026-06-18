from __future__ import annotations

import random
from typing import Tuple

import sympy as sp


def numeric_equivalent(
    a,
    b,
    *,
    n_points: int = 40,
    tol: float = 1e-9,
    seed: int = 0,
) -> Tuple[bool, float]:
    """
    Probabilistic equivalence test for semi-decidable expressions.

    Returns:
        (equivalent, confidence)

    Confidence is always strictly < 1.0.
    """
    rng = random.Random(seed)

    expr = sp.simplify(a - b)
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)

    if not symbols:
        try:
            val = abs(complex(expr.evalf()))
            return (val < tol, 0.999)
        except Exception:
            return (False, 0.1)

    valid_samples = 0

    for _ in range(n_points * 5):  # allow retries for invalid points
        subs = {s: rng.uniform(-10, 10) for s in symbols}

        try:
            v = expr.subs(subs).evalf()

            if v.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
                continue

            c = complex(v)
            if abs(c.imag) > tol:
                continue

            valid_samples += 1

            if abs(c.real) > tol:
                confidence = min(0.999, 0.2 + 0.7 * valid_samples / n_points)
                return (False, confidence)

            if valid_samples >= n_points:
                confidence = min(0.999, 0.5 + 0.499 * valid_samples / n_points)
                return (True, confidence)

        except Exception:
            continue

    confidence = min(0.999, valid_samples / max(1, n_points) * 0.5)
    return (False, confidence)
