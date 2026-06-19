from sympy import sympify, Rational, simplify
from dataclasses import dataclass

from contracts import Artifact, Target, Judgment, ParseError


def _normalize(raw: str) -> str:
    return raw.strip().replace("^", "**")


def _parse_mixed_fraction(raw: str):
    """
    Supports:
    - "1 3/4"
    - "7/4"
    - "1.75"
    """
    raw = _normalize(raw)

    # Mixed number: "1 3/4"
    if " " in raw and "/" in raw:
        whole, frac = raw.split()
        return sympify(whole) + sympify(frac)

    return sympify(raw)


class MixedFractionVerifier:
    domain = "mixed_fraction"

    def parse(self, raw: str) -> Artifact:
        try:
            expr = _parse_mixed_fraction(raw)
            return Artifact(kind="mixed_fraction", expr=expr, raw=raw, meta={})
        except Exception as e:
            raise ParseError(str(e))

    def canonical(self, a: Artifact):
        return a.expr

    def accepts(self, student: Artifact, target: Target):
        expected = sympify(target.payload["answer"])

        correct = simplify(student.expr - expected) == 0

        return Judgment(
            parsed_ok=True,
            value_equivalent=correct,
            form_ok=True,
            correct=correct,
            partial=False,
            decidable=True,
            confidence=1.0,
            detail={}
        )