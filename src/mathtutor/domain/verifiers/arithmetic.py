from sympy import simplify
from mathtutor.contracts import Target
from .base import SympyVerifierBase

class ArithmeticVerifier(SympyVerifierBase):
    domain = "arithmetic"

    def accepts(self, student, target: Target):
        expected = simplify(target.payload["answer"])
        actual = simplify(student.expr)
        return self._judgment(simplify(actual - expected) == 0)