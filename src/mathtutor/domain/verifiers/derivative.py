from sympy import sympify, diff, simplify
from sympy.abc import x

from contracts import Artifact, Target, Judgment, ParseError


def _normalize(raw: str) -> str:
    return raw.strip().replace("^", "**")


class DerivativeVerifier:
    domain = "derivative"

    def parse(self, raw: str) -> Artifact:
        try:
            expr = sympify(_normalize(raw))
            return Artifact(kind="derivative", expr=expr, raw=raw, meta={})
        except Exception as e:
            raise ParseError(str(e))

    def canonical(self, a: Artifact):
        return a.expr

    def accepts(self, student: Artifact, target: Target):
        if "expr" in target.payload:
            expected = diff(sympify(target.payload["expr"]), sympify(target.payload["variable"]))
        else:
            expected = sympify(target.payload["answer"])

        correct = simplify(student.expr - expected) == 0

        return Judgment(
            True,
            correct,
            True,
            correct,
            False,
            True,
            1.0,
            {}
        )