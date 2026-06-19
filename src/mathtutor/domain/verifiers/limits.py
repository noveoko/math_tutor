from sympy import sympify, limit, oo, simplify
from sympy.abc import x

from contracts import Artifact, Target, Judgment, ParseError


def _normalize(raw: str) -> str:
    return raw.strip().replace("^", "**")


def _parse_value(raw: str):
    raw = raw.strip()
    if raw == "oo":
        return oo
    if raw == "-oo":
        return -oo
    return sympify(raw)


class LimitVerifier:
    domain = "limits"

    def parse(self, raw: str) -> Artifact:
        try:
            val = _parse_value(_normalize(raw))
            return Artifact(kind="limit", expr=val, raw=raw, meta={})
        except Exception as e:
            raise ParseError(str(e))

    def canonical(self, a: Artifact):
        return a.expr

    def accepts(self, student: Artifact, target: Target):
        expr = sympify(target.payload["expr"])
        var = sympify(target.payload.get("variable", "x"))
        point = target.payload["approaches"]
        direction = target.payload.get("dir")

        if direction:
            computed = limit(expr, var, point, dir=direction)
        else:
            computed = limit(expr, var, point)

        correct = simplify(student.expr - computed) == 0

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