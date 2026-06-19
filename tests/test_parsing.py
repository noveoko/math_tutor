import sympy as sp
import pytest

from mathtutor.cas.parsing import parse_math
from mathtutor.contracts import ParseError


def test_equation_parses():
    a = parse_math("2x^2-5x+6=0")
    assert a.kind == "equation"
    assert isinstance(a.expr, sp.Equality)


def test_fraction_parses_to_rational():
    a = parse_math("3/4")
    assert a.kind == "value"
    assert a.expr == sp.Rational(3, 4)


def test_garbage_raises():
    with pytest.raises(ParseError):
        parse_math("2x +")
