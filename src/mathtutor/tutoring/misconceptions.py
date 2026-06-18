# mathtutor/tutoring/misconceptions.py

from typing import Any, List
from sympy import Expr, Eq, Add, Mul, Pow, Rel, simplify
from mathtutor.contracts import BuggyRule


def exact_match(a: Any, b: Any) -> bool:
    """
    Checks for canonical structural equality via SymPy.
    Ensures mathematical equivalence AND structural identity so that
    aggressive simplification doesn't erase the buggy form.
    """
    if type(a) != type(b):
        return False
    
    # SymPy's `==` handles structural identity (and basic commutativity).
    if a == b:
        return True
        
    # Handle swapped equations
    if hasattr(a, 'lhs') and hasattr(b, 'lhs'):
        if isinstance(a, Eq) and a.lhs == b.rhs and a.rhs == b.lhs:
            return True
        # If it's a relational or un-swapped Eq, we want structural match of sides.
        return False
        
    # Fallback: check mathematical equivalence combined with structural bounds.
    # We use a.equals(b) but restrict matched forms to have identical operation counts.
    try:
        if hasattr(a, 'equals') and hasattr(b, 'equals') and a.equals(b):
            return (a.count_ops() == b.count_ops()) and (len(a.args) == len(b.args))
    except Exception:
        pass
        
    return False


class DistributeExponentOverSum:
    id = "distributes_exponent_over_sum"
    
    def applies_to(self, previous: Any) -> bool:
        return previous.has(Pow)
        
    def transform(self, previous: Any) -> List[Any]:
        """(a+b)**2 -> a**2 + b**2"""
        try:
            res = previous.replace(
                lambda x: x.is_Pow and x.base.is_Add,
                lambda x: Add(*[Pow(arg, x.exp) for arg in x.base.args])
            )
            return [res] if res != previous else []
        except Exception:
            return []


class MoveTermWithoutSignFlip:
    id = "moves_term_across_equals_without_sign_flip"
    
    def applies_to(self, previous: Any) -> bool:
        return isinstance(previous, Eq) and (previous.lhs.is_Add or previous.rhs.is_Add)
        
    def transform(self, previous: Any) -> List[Any]:
        """Eq(LHS + term, RHS) -> Eq(LHS, RHS + term)"""
        results = []
        if not isinstance(previous, Eq):
            return results
            
        if previous.lhs.is_Add:
            for arg in previous.lhs.args:
                results.append(Eq(previous.lhs - arg, previous.rhs + arg))
        if previous.rhs.is_Add:
            for arg in previous.rhs.args:
                results.append(Eq(previous.lhs + arg, previous.rhs - arg))
        return results


class CancelTermAcrossAddition:
    id = "cancels_term_across_addition"
    
    def applies_to(self, previous: Any) -> bool:
        return previous.has(Mul)
        
    def transform(self, previous: Any) -> List[Any]:
        """(a+b)/a -> b"""
        results = []
        def replacer(expr: Expr) -> Expr:
            if expr.is_Mul:
                adds = [a for a in expr.args if a.is_Add]
                pows = [a for a in expr.args if a.is_Pow and a.exp == -1]
                if adds and pows:
                    for add in adds:
                        for p in pows:
                            denom = p.base
                            for arg in add.args:
                                if arg == denom:
                                    return add - arg # Drops denominator and 'a'
            return expr
            
        try:
            new_expr = previous.replace(lambda x: x.is_Mul, replacer)
            if new_expr != previous:
                results.append(new_expr)
        except Exception:
            pass
        return results


class ClearsFractionOneTerm:
    id = "clears_fraction_multiplying_only_one_term"
    
    def applies_to(self, previous: Any) -> bool:
        return isinstance(previous, Eq) and (previous.lhs.has(Mul) or previous.rhs.has(Mul))
        
    def transform(self, previous: Any) -> List[Any]:
        """Eq(A/C + B, D) -> Eq(A + B, D*C)"""
        results = []
        if not isinstance(previous, Eq):
            return results
            
        for side, other in [(previous.lhs, previous.rhs), (previous.rhs, previous.lhs)]:
            if side.is_Add:
                for arg in side.args:
                    if arg.is_Mul:
                        pows = [a for a in arg.args if a.is_Pow and a.exp == -1]
                        if pows:
                            denom = pows[0].base
                            buggy_arg = arg * denom
                            buggy_side = side - arg + buggy_arg
                            buggy_other = other * denom
                            results.append(Eq(buggy_side, buggy_other))
        return results


class InequalityMultiplyNegativeNoFlip:
    id = "forgets_to_flip_inequality_on_negative_multiply"
    
    def applies_to(self, previous: Any) -> bool:
        return isinstance(previous, Rel)
        
    def transform(self, previous: Any) -> List[Any]:
        """-x < 5 -> x < -5"""
        if isinstance(previous, Rel):
            try:
                # Mimic multiplying by -1 without changing the relational operator
                return [type(previous)(previous.lhs * -1, previous.rhs * -1)]
            except Exception:
                pass
        return []


BUGGY_RULES: List[BuggyRule] = [
    DistributeExponentOverSum(),
    MoveTermWithoutSignFlip(),
    CancelTermAcrossAddition(),
    ClearsFractionOneTerm(),
    InequalityMultiplyNegativeNoFlip()
]


def diagnose(previous: Any, student_line: Any, rules: List[BuggyRule] = None) -> List[str]:
    """
    Evaluates known misconceptions by structurally verifying the transformation against the student line.
    """
    if rules is None:
        rules = BUGGY_RULES
        
    matched_rules = []
    for r in rules:
        try:
            if not r.applies_to(previous):
                continue
            transforms = r.transform(previous)
            for t in transforms:
                if exact_match(t, student_line):
                    matched_rules.append(r.id)
                    break
        except Exception:
            continue
            
    return matched_rules


def classify_error(previous: Any, student_line: Any) -> str:
    """
    Symptom-level fallback classification when exact misconception diagnosis fails.
    Returns: "sign_error" | "off_by_one" | "dropped_root" | "unknown"
    """
    try:
        if isinstance(previous, Eq) and isinstance(student_line, Eq):
            # Check for sign error on one of the sides
            if exact_match(previous.lhs * -1, student_line.lhs) or \
               exact_match(previous.rhs * -1, student_line.rhs):
                return "sign_error"
                
            # Check for simple arithmetic off-by-one
            diff = simplify((previous.lhs - previous.rhs) - (student_line.lhs - student_line.rhs))
            if diff.is_number and abs(diff) == 1:
                return "off_by_one"
    except Exception:
        pass
        
    return "unknown"
