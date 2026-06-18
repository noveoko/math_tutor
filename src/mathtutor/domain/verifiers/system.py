# mathtutor/domain/verifiers/system.py

from __future__ import annotations

from sympy import linsolve, FiniteSet

from mathtutor.contracts import Verifier, Artifact, Canonical, Target, Judgment
from mathtutor.cas.parsing import parse_math


class SystemVerifier(Verifier):
    """Verifier for systems of equations using exact solution-set equality."""

    domain = "system"

    def parse(self, raw: str) -> Artifact:
        return parse_math(raw)

    def canonical(self, a: Artifact) -> Canonical:
        eqs = a if isinstance(a, (list, tuple)) else [a]
        symbols = sorted(
            set().union(*(eq.free_symbols for eq in eqs)),
            key=lambda s: s.name,
        )
        return linsolve(eqs, symbols)

    def accepts(self, student: Artifact, target: Target) -> Judgment:
        try:
            s = self.canonical(student)
            t = self.canonical(target.payload["answer"])
        except Exception:
            return Judgment(False, False, False, False, False, True, 1.0, {})

        value_equivalent = s == t

        return Judgment(
            True,
            value_equivalent,
            True,
            value_equivalent,
            False,
            True,
            1.0,
            {},
        )
