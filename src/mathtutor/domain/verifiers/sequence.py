from sympy import sympify, simplify
from sympy.abc import n

from contracts import Artifact, Target, Judgment, ParseError


def _normalize(raw: str) -> str:
    return raw.strip().replace("^", "**")


class SequenceVerifier:
    domain = "sequence"

    def parse(self, raw: str) -> Artifact:
        try:
            raw_n = _normalize(raw)

            if "," in raw_n:
                items = [sympify(x.strip()) for x in raw_n.split(",")]
                return Artifact(
                    kind="sequence",
                    expr=items,
                    raw=raw,
                    meta={"mode": "list"}
                )

            expr = sympify(raw_n)
            return Artifact(
                kind="sequence",
                expr=expr,
                raw=raw,
                meta={"mode": "formula"}
            )

        except Exception as e:
            raise ParseError(str(e))

    def canonical(self, a: Artifact):
        return a.expr

    def accepts(self, student: Artifact, target: Target):
        mode = target.payload["mode"]

        # MODE A: explicit list
        if mode == "list":
            expected = target.payload["answer"]

            if len(student.expr) != len(expected):
                return Judgment(False, False, False, False, False, True, 1.0, {})

            correct = all(
                simplify(sympify(student.expr[i]) - sympify(expected[i])) == 0
                for i in range(len(expected))
            )

            return Judgment(True, correct, True, correct, False, True, 1.0, {})

        # MODE B: formula
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