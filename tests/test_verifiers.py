# tests/test_verifiers.py

import pytest
from sympy import Eq, symbols, Rational
from sympy import symbols, Eq
from mathtutor.domain.verifiers.linear_equation import EquationVerifier
from mathtutor.domain.verifiers.fraction import FractionVerifier
from mathtutor.domain.verifiers.polynomial import PolynomialVerifier
from mathtutor.domain.verifiers.inequality import InequalityVerifier
from mathtutor.domain.verifiers.system import SystemVerifier
from mathtutor.contracts import Target as _Target


x, y = symbols('x y')

def Target(answer, form=None, *, domain="expression"):
    """Build a real contract Target for verifier tests.

    Exposes `form` as a top-level field (matching contracts.Target) and the
    answer under payload['answer'] — the two things verifiers actually read.
    """
    return _Target(domain=domain, payload={"answer": answer}, form=form)


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
