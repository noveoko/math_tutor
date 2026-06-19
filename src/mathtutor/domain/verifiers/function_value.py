from sympy import simplify
from .base import SympyVerifierBase

class FunctionValueVerifier(SympyVerifierBase):
    domain = "function_value"

    def accepts(self, student, target):
        expected = simplify(target.payload["answer"])
        actual = simplify(student.expr)
        return self._judgment(actual == expected)