from sympy import simplify
from .base import SympyVerifierBase

class GraphEquationVerifier(SympyVerifierBase):
    domain = "graph_equation"

    def accepts(self, student, target):
        expected = target.payload["answer"]
        correct = simplify(student.expr - expected) == 0
        return self._judgment(correct)