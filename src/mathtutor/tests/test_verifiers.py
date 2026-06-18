# tests/test_verifiers.py

import pytest
from sympy import Eq, symbols, Rational

from mathtutor.domain.verifiers.linear_equation import EquationVerifier
from mathtutor.domain.verifiers.fraction import FractionVerifier
from mathtutor.domain.verifiers.polynomial import PolynomialVerifier
from mathtutor.domain.verifiers.inequality import InequalityVerifier
from mathtutor.domain.verifiers.system import SystemVerifier


class Target:
    """Lightweight stand-in for contracts.Target used in verifier tests.

    Verifiers read the expected answer via ``target.payload["answer"]``,
    so the shim must expose a ``payload`` dict — not a bare attribute.
    """
    def __init__(self, answer, form=None):
        self.payload = {"answer": answer}
        if form is not None:
            self.payload["form"] = form


x, y = symbols("x y")


def test_equation():
    v = EquationVerifier()
    t = Target(Eq(x**2 - 5*x + 6, 0))
    assert v.accepts(Eq(x**2 - 5*x + 6, 0), t).correct


def test_fraction():
    v = FractionVerifier()
    t = Target(Rational(5, 6))
    assert v.accepts(Rational(5, 6), t).correct
    j = v.accepts(Rational(10, 12), t)
    assert j.value_equivalent


def test_polynomial():
    v = PolynomialVerifier()
    t = Target((x + 1) * (x + 2), form="expanded")
    assert v.accepts(x**2 + 3*x + 2, t).correct
    assert not v.accepts((x + 1) * (x + 2), t).form_ok


def test_inequality():
    v = InequalityVerifier()
    t = Target(x > 2)
    assert v.accepts(x > 2, t).correct


def test_system():
    v = SystemVerifier()
    t = Target([Eq(x + y, 3), Eq(x - y, 1)])
    assert v.accepts([Eq(x + y, 3), Eq(x - y, 1)], t).correct
