# mathtutor/cas/equivalence.py
"""CAS-level equivalence shared by verifiers and the claim certifier.

Handles heterogeneous answer representations — bare values/expressions,
equations (Eq), and solution sets (FiniteSet / other Sets). This is general
CAS equivalence, not specific to any single verifier domain.
"""
from __future__ import annotations

from sympy import Eq, FiniteSet, S, simplify
from sympy.sets.sets import Set
from sympy.solvers.solveset import solveset


def _as_set_or_value(obj):
    """Classify obj as a solution Set or a bare value.

    Returns (is_set, payload):
      * (True, <Set>)    obj is already a Set, or an equation we can solve
      * (False, <expr>)  a number/expression to be compared by value
    """
    if isinstance(obj, Set):
        return True, obj
    if isinstance(obj, Eq):
        sym = next(iter(obj.free_symbols), None)
        if sym is None:                       # degenerate, e.g. Eq(3, 3)
            return False, obj.lhs - obj.rhs
        return True, solveset(obj, sym, domain=S.Reals)
    return False, obj

def normalize_answer(verifier, answer):
    """payload['answer'] may be a raw string (generators) or a pre-parsed
    SymPy object (unit tests / certify). Parse strings via the verifier's own
    parser; pass everything else through unchanged."""
    if isinstance(answer, str):
        parsed = verifier.parse(answer)
        return getattr(parsed, "expr", parsed)   # unwrap Artifact if needed
    return answer

def value_equivalent(student, answer) -> bool:
    """True iff `student` and `answer` denote the same value or solution set.

      set   vs set   -> set equality
      set   vs value -> lift the value to the singleton {value}, compare sets
      value vs value -> simplify(difference) == 0
    """
    s_set, s_val = _as_set_or_value(student)
    a_set, a_val = _as_set_or_value(answer)
    if s_set and a_set:
        return s_val == a_val
    if s_set != a_set:
        the_set = s_val if s_set else a_val
        the_val = a_val if s_set else s_val
        return the_set == FiniteSet(the_val)
    return simplify(s_val - a_val) == 0