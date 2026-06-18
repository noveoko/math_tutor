# mathtutor/domain/verifiers/fraction.py

from __future__ import annotations

from math import gcd
from sympy import Rational

from mathtutor.contracts import Verifier, Artifact, Canonical, Target, Judgment, ParseError
from mathtutor.cas.parsing import parse_math


class FractionVerifier(Verifier):
    """Verifier for exact rational numbers with reduced-form requirement."""

    domain = "fraction"

    def parse(self, raw: str) -> Artifact:
        return parse_math(raw)

    def canonical(self, a: Artifact) -> Canonical:
        return Rational(a)

    def _is_reduced(self, r: Rational) -> bool:
        return gcd(abs(r.p), abs(r.q)) == 1

    def accepts(self, student: Artifact, target: Target) -> Judgment:
        try:
            s = self.canonical(student)
            t = self.canonical(target.payload["answer"])
        except Exception:
            return Judgment(False, False, False, False, False, True, 1.0, {})

        value_equivalent = s == t
        form_ok = self._is_reduced(student if isinstance(student, Rational) else s)

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
