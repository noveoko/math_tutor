from __future__ import annotations

import re
from typing import List

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from mathtutor.contracts import Artifact, ParseError

_TRANSFORMS = (
    standard_transformations
    + (implicit_multiplication_application, convert_xor)
)


def _normalize(raw: str) -> str:
    """Normalize loose student input into something SymPy can parse."""
    s = raw.strip()
    if not s:
        raise ParseError("Empty input.")

    s = s.replace("∞", "inf")
    s = s.replace("−", "-")
    s = s.replace("\n", ";")

    # Interval notation like (-inf, 3]
    s = re.sub(r"\binf\b", "oo", s)

    return s


def _parse_single(expr: str):
    expr = expr.strip()
    if not expr:
        raise ParseError("Empty expression in system.")

    # Set literal
    if expr.startswith("{") and expr.endswith("}"):
        inner = expr[1:-1].strip()
        if not inner:
            return sp.FiniteSet()
        parts = [parse_expr(x.strip(), transformations=_TRANSFORMS) for x in inner.split(",")]
        return sp.FiniteSet(*parts)

    # Interval literal
    interval_match = re.match(r"^([\(\[])\s*(.+?)\s*,\s*(.+?)\s*([\)\]])$", expr)
    if interval_match:
        left_br, a_raw, b_raw, right_br = interval_match.groups()
        a = sp.sympify(a_raw)
        b = sp.sympify(b_raw)
        return sp.Interval(
            a,
            b,
            left_open=(left_br == "("),
            right_open=(right_br == ")"),
        )

    # Equation
    if "=" in expr and not any(op in expr for op in ("<=", ">=", "<", ">")):
        lhs, rhs = expr.split("=", 1)
        if not lhs.strip() or not rhs.strip():
            raise ParseError(f"Malformed equation: {expr}")
        return sp.Eq(
            parse_expr(lhs, transformations=_TRANSFORMS),
            parse_expr(rhs, transformations=_TRANSFORMS),
        )

    # Inequalities
    for op in ("<=", ">=", "<", ">"):
        if op in expr:
            lhs, rhs = expr.split(op, 1)
            if not lhs.strip() or not rhs.strip():
                raise ParseError(f"Malformed inequality: {expr}")
            l = parse_expr(lhs, transformations=_TRANSFORMS)
            r = parse_expr(rhs, transformations=_TRANSFORMS)
            return {
                "<=": sp.Le,
                ">=": sp.Ge,
                "<": sp.Lt,
                ">": sp.Gt,
            }[op](l, r)

    return parse_expr(expr, transformations=_TRANSFORMS)


def parse_math(raw: str) -> Artifact:
    """
    Parse loose student math input into an Artifact containing a SymPy object.

    Supported:
      - implicit multiplication: 2x, 3(x+1)
      - ^ exponent
      - equations, inequalities
      - systems via newline or ';'
      - sets {1,2}
      - intervals (-inf,3]
    """
    s = _normalize(raw)

    try:
        if ";" in s:
            parts = [p.strip() for p in s.split(";") if p.strip()]
            objs = [_parse_single(p) for p in parts]
            return Artifact(kind="system", expr=tuple(objs), raw="system")

        obj = _parse_single(s)

        if isinstance(obj, sp.Equality):
            kind = "equation"
        elif isinstance(obj, sp.core.relational.Relational):
            kind = "inequality"
        elif isinstance(obj, (sp.Set, sp.FiniteSet, sp.Interval)):
            kind = "set"
        elif obj.is_number:
            kind = "value"
        else:
            kind = "expression"

        return Artifact(kind=kind, expr=obj, raw=s)

    except ParseError:
        raise
    except Exception as e:
        raise ParseError(f"Could not parse input: {raw!r}. Reason: {e}") from e


def echo_latex(a: Artifact) -> str:
    """Return LaTeX rendering of parsed artifact."""
    if a.kind == "system":
        return r"\left\{" + ", ".join(sp.latex(x) for x in a.obj) + r"\right."
    return sp.latex(a.obj)
