from sympy import simplify
from .base import SympyVerifierBase

class TrigVerifier(SympyVerifierBase):
    domain = "trig"

    def accepts(self, student, target):
        expected = target.payload["answer"]
        correct = simplify(student.expr - expected) == 0
        return self._judgment(correct)