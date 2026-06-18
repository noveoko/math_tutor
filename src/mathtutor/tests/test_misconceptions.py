# tests/test_misconceptions.py

import pytest
from sympy import symbols, Eq, StrictLessThan, Add, Pow, Mul
from mathtutor.cas.parsing import parse_math
from mathtutor.tutoring.misconceptions import (
    diagnose, classify_error, exact_match, BUGGY_RULES,
    CancelTermAcrossAddition, ClearsFractionOneTerm
)

x, y = symbols('x y')

def test_exact_match():
    # Commutativity should match
    assert exact_match(x**2 + y**2, y**2 + x**2) is True
    # Structural difference shouldn't match (factored vs expanded)
    assert exact_match(x*(x + 1), x**2 + x) is False
    # Swapped equations
    assert exact_match(Eq(x, 5), Eq(5, x)) is True

def test_distributes_exponent_over_sum():
    P = (x + y)**2
    S = x**2 + y**2
    assert diagnose(P, S) == ["distributes_exponent_over_sum"]

def test_moves_term_across_equals_without_sign_flip():
    P = Eq(x + 3, 5)
    S = Eq(x, 5 + 3)
    assert diagnose(P, S) == ["moves_term_across_equals_without_sign_flip"]

def test_cancels_term_across_addition():
    # (x + y) / x
    P = Mul(Add(x, y), Pow(x, -1)) 
    S = y 
    assert diagnose(P, S) == ["cancels_term_across_addition"]

def test_clears_fraction_one_term():
    # x/2 + y = 5
    P = Eq(Mul(x, Pow(2, -1)) + y, 5)
    # x + y = 10 (forgot to multiply y by 2)
    S = Eq(x + y, 10)
    assert diagnose(P, S) == ["clears_fraction_multiplying_only_one_term"]

def test_forgets_to_flip_inequality():
    P = StrictLessThan(-x, 5)
    S = StrictLessThan(x, -5)
    assert diagnose(P, S) == ["forgets_to_flip_inequality_on_negative_multiply"]

def test_multiple_matches():
    # Construct a state that can plausibly hit two isolated/mocked buggy rules 
    # to test the disambiguation contract.
    class DummyRuleA:
        id = "rule_a"
        def applies_to(self, P): return True
        def transform(self, P): return [S_target]

    class DummyRuleB:
        id = "rule_b"
        def applies_to(self, P): return True
        def transform(self, P): return [S_target]

    P_dummy = Eq(x, 1)
    S_target = Eq(x, 2)
    
    rules = [DummyRuleA(), DummyRuleB()]
    assert diagnose(P_dummy, S_target, rules=rules) == ["rule_a", "rule_b"]

def test_novel_wrong_line_unknown():
    P = Eq(x, 2)
    S = Eq(x, 5)
    assert diagnose(P, S) == []
    assert classify_error(P, S) == "unknown"

def test_novel_wrong_line_sign_error():
    P = Eq(x, 2)
    S = Eq(-x, 2)
    assert diagnose(P, S) == []
    assert classify_error(P, S) == "sign_error"

def test_novel_wrong_line_off_by_one():
    P = Eq(x, 2)
    S = Eq(x, 3)
    assert diagnose(P, S) == []
    assert classify_error(P, S) == "off_by_one"
