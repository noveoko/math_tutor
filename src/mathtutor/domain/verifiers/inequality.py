# mathtutor/domain/verifiers/inequality.py

from __future__ import annotations

from sympy import S
from sympy.solvers.solveset import solveset

from mathtutor.contracts import Verifier, Artifact, Canonical, Target, Judgment
from mathtutor.cas.parsing import parse_math


class InequalityVerifier(Verifier):
    """Verifier for inequality solution sets."""

    domain = "inequality"

    def parse(self, raw: str) -> Artifact:
        return parse_math(raw)

    def canonical(self, a: Artifact) -> Canonical:
        symbol = next(iter(a.free_symbols))
        return solveset(a, symbol, domain=S.Reals)

    def accepts(self, student: Artifact, target: Target) -> Judgment:
        try:
            s = self.canonical(student)
            t = self.canonical(target.payload["answer"])
        except Exception:
            return Judgment(False, False, False, False, False, True, 1.0, {})

        value_equivalent = s == t
        form_ok = getattr(target, "form", None) != "interval" or value_equivalent

        return Judgment(
            True,
            value_equivalent,
            form_ok,
            value_equivalent and form_ok,
            False,
            True,
            1.0,
            {},
        )
