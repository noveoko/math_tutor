from __future__ import annotations
from sympy import sympify, simplify
from sympy.parsing.sympy_parser import parse_expr
from mathtutor.contracts import (
    Artifact, Canonical, Judgment, ParseError
)

class SympyVerifierBase:
    domain = "generic"

    def parse(self, raw: str) -> Artifact:
        try:
            expr = parse_expr(raw.replace("^", "**"))
            return Artifact(
                kind="expression",
                expr=expr,
                raw=raw
            )
        except Exception as e:
            raise ParseError(str(e))

    def canonical(self, a: Artifact) -> Canonical:
        return Canonical(key=simplify(a.expr))

    def _judgment(self, correct: bool, detail=None) -> Judgment:
        return Judgment(
            parsed_ok=True,
            value_equivalent=correct,
            form_ok=True,
            correct=correct,
            partial=False,
            decidable=True,
            confidence=1.0,
            detail=detail or {}
        )