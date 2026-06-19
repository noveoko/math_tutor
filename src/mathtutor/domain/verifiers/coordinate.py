from sympy import sympify
from mathtutor.contracts import *

class CoordinateVerifier:
    domain = "coordinate"

    def parse(self, raw):
        try:
            raw = raw.strip().replace("(", "").replace(")", "")
            xs, ys = raw.split(",")
            x = sympify(xs)
            y = sympify(ys)
            return Artifact(
                kind="point",
                expr=(x, y),
                raw=raw
            )
        except Exception as e:
            raise ParseError(str(e))

    def canonical(self, a):
        return Canonical(key=a.expr)

    def accepts(self, student, target):
        tx, ty = target.payload["point"]
        sx, sy = student.expr
        correct = sx == tx and sy == ty

        return Judgment(
            parsed_ok=True,
            value_equivalent=correct,
            form_ok=True,
            correct=correct,
            partial=False,
            decidable=True,
            confidence=1.0
        )