import sympy as sp

from mathtutor.cas.numeric import numeric_equivalent


def test_trig_identity():
    x = sp.Symbol("x")
    eq, conf = numeric_equivalent(sp.sin(x) ** 2 + sp.cos(x) ** 2, 1)
    assert eq is True
    assert conf < 1.0


def test_non_equivalent():
    x = sp.Symbol("x")
    eq, _ = numeric_equivalent(x, x + 1)
    assert eq is False
