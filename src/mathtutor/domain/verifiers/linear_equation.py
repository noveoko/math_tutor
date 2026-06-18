# mathtutor/domain/verifiers/equation.py

from __future__ import annotations

from typing import Any
from sympy import Eq, FiniteSet, Symbol, S
from sympy.solvers.solveset import solveset
from sympy.sets.sets import Set

from mathtutor.contracts import Verifier, Artifact, Canonical, Target, Judgment, ParseError
from mathtutor.cas.parsing import parse_math
from sympy import FiniteSet
from sympy.sets.sets import Set
from mathtutor.cas.equivalence import normalize_answer

class EquationVerifier(Verifier):
    """Verifier for algebraic equations via exact solution-set equality."""

    domain = "equation"

    def parse(self, raw: str) -> Artifact:
        return parse_math(raw)

    def canonical(self, a: Artifact) -> Canonical:
        if isinstance(a, Eq):
            symbol = next(iter(a.free_symbols), Symbol("x"))
            return solveset(a, symbol, domain=S.Reals)
        if isinstance(a, Set):
            return a
        return FiniteSet(a)        # bare value → singleton solution set {a}

    def accepts(self, student: Artifact, target: Target) -> Judgment:
        student = student.expr if isinstance(student, Artifact) else student
        answer = normalize_answer(self, target.payload["answer"])
        try:
            target_set = self.canonical(answer)
            student_set = self.canonical(student)
        except ParseError:
            return Judgment(False, False, False, False, False, True, 1.0, {})
        except Exception:
            return Judgment(False, False, False, False, False, True, 1.0, {})

        value_equivalent = student_set == target_set
        missing = list(target_set - student_set) if isinstance(target_set, Set) else []
        extra = list(student_set - target_set) if isinstance(student_set, Set) else []
        partial = bool(missing) and not extra and len(student_set) > 0

        return Judgment(
            parsed_ok=True,
            value_equivalent=value_equivalent,
            form_ok=True,
            correct=value_equivalent,
            partial=partial,
            decidable=True,
            confidence=1.0,
            detail={"missing": missing, "extra": extra},
        )
