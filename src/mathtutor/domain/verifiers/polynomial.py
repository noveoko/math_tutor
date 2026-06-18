# mathtutor/domain/verifiers/polynomial.py

from __future__ import annotations

from sympy import expand, factor, Mul, simplify
import sympy as sp
from sympy import FiniteSet, Eq
from mathtutor.contracts import Verifier, Artifact, Canonical, Target, Judgment
from mathtutor.cas.parsing import parse_math
from mathtutor.cas.equivalence import value_equivalent

def normalize_answer(verifier, answer_str: str):
    """Parse and normalize answers for comparison (handles sets, equations, etc.)."""
    try:
        if isinstance(answer_str, str):
            # Handle set notation like '{2, 2}' or '{-7, 0}'
            if answer_str.startswith('{') and answer_str.endswith('}'):
                content = answer_str[1:-1].strip()
                if content:
                    items = [sp.sympify(item.strip()) for item in content.split(',')]
                    return FiniteSet(*items)
            # Try parsing as sympy expression/equation
            return sp.sympify(answer_str)
        return answer_str
    except Exception:
        return answer_str  # fallback
    
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
        student = student.expr if isinstance(student, Artifact) else student
        answer = normalize_answer(self, target.payload["answer"])  # now defined
        try:
            ve = value_equivalent(student, answer)
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
            ve,
            form_ok,
            ve and form_ok,
            False,
            True,
            1.0,
            {"expected_form": required_form},
        )
