from sympy import simplify
from mathtutor.contracts import Target
from .base import SympyVerifierBase

class ExpressionVerifier(SympyVerifierBase):
    domain = "expression"

    def accepts(self, student, target):
        expected = simplify(target.payload["answer"])
        correct = simplify(student.expr - expected) == 0
        return self._judgment(correct)