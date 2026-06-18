# mathtutor/domain/verifiers/polynomial.py

from __future__ import annotations

from sympy import expand, factor, Mul, simplify

from mathtutor.contracts import Verifier, Artifact, Canonical, Target, Judgment
from mathtutor.cas.parsing import parse_math


class PolynomialVerifier(Verifier):
    """Verifier for polynomial equivalence and structural form."""

    domain = "polynomial"

    def parse(self, raw: str) -> Artifact:
        return parse_math(raw)

    def canonical(self, a: Artifact) -> Canonical:
        return expand(a)

    def _is_expanded(self, expr) -> bool:
        return expand(expr) == expr

    def _is_fully_factored(self, expr) -> bool:
        return factor(expr) == expr

    def accepts(self, student: Artifact, target: Target) -> Judgment:
        try:
            value_equivalent = simplify(student - target.payload["answer"]) == 0
        except Exception:
            return Judgment(False, False, False, False, False, True, 1.0, {})

        required_form = getattr(target, "form", None)
        if required_form == "expanded":
            form_ok = self._is_expanded(student)
        elif required_form == "factored":
            form_ok = self._is_fully_factored(student)
        else:
            form_ok = True

        return Judgment(
            True,
            value_equivalent,
            form_ok,
            value_equivalent and form_ok,
            False,
            True,
            1.0,
            {"expected_form": required_form},
        )
