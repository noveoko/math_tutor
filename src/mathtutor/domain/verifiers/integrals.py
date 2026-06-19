from sympy import sympify, integrate, diff, simplify
from sympy.abc import x

from contracts import Artifact, Target, Judgment, ParseError


def _normalize(raw: str) -> str:
    return raw.strip().replace("^", "**")


def _strip_constant(expr_str: str) -> str:
    return expr_str.replace("+ C", "").replace("+C", "").replace("+ c", "").replace("+c", "")


class IntegralVerifier:
    domain = "integral"

    def parse(self, raw: str) -> Artifact:
        try:
            cleaned = _strip_constant(_normalize(raw))
            expr = sympify(cleaned)
            return Artifact(kind="integral", expr=expr, raw=raw, meta={})
        except Exception as e:
            raise ParseError(str(e))

    def canonical(self, a: Artifact):
        return a.expr

    def accepts(self, student: Artifact, target: Target):
        mode = target.payload["mode"]

        # DEFINTIE
        if mode == "definite":
            expected = sympify(target.payload["answer"])
            correct = simplify(student.expr - expected) == 0

            return Judgment(True, correct, True, correct, False, True, 1.0, {})

        # INDEFINITE
        if "integrand" in target.payload:
            integrand = sympify(target.payload["integrand"])
            var = sympify(target.payload.get("variable", "x"))

            # derivative check (best method)
            correct = simplify(diff(student.expr, var) - integrand) == 0
        else:
            expected = sympify(target.payload["answer"])
            correct = simplify(student.expr - expected) == 0

        return Judgment(True, correct, True, correct, False, True, 1.0, {})