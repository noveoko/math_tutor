# mathtutor/domain/generators.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any
import random
import uuid
from fractions import Fraction

from mathtutor.contracts import Target
from mathtutor.domain.verifiers.linear_equation import EquationVerifier as LinearEquationVerifier
from mathtutor.domain.verifiers.polynomial import PolynomialVerifier as QuadraticEquationVerifier
from mathtutor.domain.verifiers.fraction import FractionVerifier as FractionAdditionVerifier


@dataclass(frozen=True)
class Problem:
    id: str
    kc_id: str
    domain: str
    prompt_text: str
    parsed_target: Target
    reference_answer: str
    difficulty_band: int
    meta: dict


class GeneratorError(Exception):
    """Raised when a generated problem fails self-verification."""


_REGISTRY: Dict[str, Callable[..., Problem]] = {}


def register(name: str) -> Callable:
    """Decorator to register a problem generator."""

    def deco(fn: Callable[..., Problem]) -> Callable[..., Problem]:
        _REGISTRY[name] = fn
        return fn

    return deco


def generate(name: str, *, difficulty_band: int, seed: int) -> Problem:
    """Generate a problem deterministically from a registered generator."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown generator: {name}")
    return _REGISTRY[name](difficulty_band=difficulty_band, seed=seed)


def _difficulty_range(band: int) -> tuple[int, int]:
    """Cheap solve-step proxy range."""
    ranges = {
        1: (1, 3),
        2: (3, 6),
        3: (6, 10),
    }
    if band not in ranges:
        raise GeneratorError(f"Unsupported difficulty band: {band}")
    return ranges[band]


def _assert_self_verified(
    problem: Problem,
    verifier: Any,
    solve_steps: int,
) -> None:
    """Run verifier and difficulty checks."""
    student = verifier.parse(problem.reference_answer)
    judgment = verifier.accepts(student, problem.parsed_target)

    if not judgment.correct:
        raise GeneratorError(
            f"Verifier rejected generated answer: {problem.reference_answer}"
        )

    low, high = _difficulty_range(problem.difficulty_band)
    if not (low <= solve_steps <= high):
        raise GeneratorError(
            f"Solve-step proxy {solve_steps} out of band {problem.difficulty_band}"
        )


@register("linear_equation")
def generate_linear_equation(*, difficulty_band: int, seed: int) -> Problem:
    rng = random.Random(seed)

    coeff_max = {1: 5, 2: 20, 3: 50}[difficulty_band]
    x_solution = rng.randint(-coeff_max, coeff_max)
    a = rng.choice([i for i in range(-coeff_max, coeff_max + 1) if i not in (0,)])
    b = rng.randint(-coeff_max, coeff_max)
    c = a * x_solution + b

    prompt = f"Solve for x: {a}x + {b} = {c}"
    reference = str(x_solution)

    target = Target(domain="linear_equation", payload={"answer": reference})

    problem = Problem(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"linear-{seed}-{difficulty_band}")),
        kc_id="solve_linear_eq",
        domain="linear_equation",
        prompt_text=prompt,
        parsed_target=target,
        reference_answer=reference,
        difficulty_band=difficulty_band,
        meta={"a": a, "b": b, "c": c},
    )

    solve_steps = 2 if difficulty_band == 1 else 4 if difficulty_band == 2 else 7
    _assert_self_verified(problem, LinearEquationVerifier(), solve_steps)
    return problem


@register("quadratic_equation")
def generate_quadratic_equation(*, difficulty_band: int, seed: int) -> Problem:
    rng = random.Random(seed)

    root_max = {1: 4, 2: 8, 3: 15}[difficulty_band]

    r1 = rng.randint(-root_max, root_max)
    r2 = rng.randint(-root_max, root_max)

    a = 1
    b = -(r1 + r2)
    c = r1 * r2

    prompt = f"Solve for x: x^2 + ({b})x + ({c}) = 0"
    reference = f"{{{min(r1,r2)}, {max(r1,r2)}}}"

    target = Target(domain="quadratic_equation", payload={"answer": reference})

    problem = Problem(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"quad-{seed}-{difficulty_band}")),
        kc_id="solve_quadratic_eq",
        domain="quadratic_equation",
        prompt_text=prompt,
        parsed_target=target,
        reference_answer=reference,
        difficulty_band=difficulty_band,
        meta={"roots": (r1, r2)},
    )

    solve_steps = 3 if difficulty_band == 1 else 5 if difficulty_band == 2 else 8
    _assert_self_verified(problem, QuadraticEquationVerifier(), solve_steps)
    return problem


@register("fraction_addition")
def generate_fraction_addition(*, difficulty_band: int, seed: int) -> Problem:
    rng = random.Random(seed)

    term_count = {1: 2, 2: 3, 3: 4}[difficulty_band]
    denom_max = {1: 6, 2: 12, 3: 20}[difficulty_band]

    fractions = []
    total = Fraction(0, 1)

    for _ in range(term_count):
        denom = rng.randint(2, denom_max)
        numer = rng.randint(1, denom - 1)
        frac = Fraction(numer, denom)
        fractions.append(frac)
        total += frac

    prompt = "Compute: " + " + ".join(f"{f.numerator}/{f.denominator}" for f in fractions)
    reference = f"{total.numerator}/{total.denominator}"

    target = Target(domain="fraction_addition", payload={"answer": reference})

    problem = Problem(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"frac-{seed}-{difficulty_band}")),
        kc_id="fraction_addition",
        domain="fraction_addition",
        prompt_text=prompt,
        parsed_target=target,
        reference_answer=reference,
        difficulty_band=difficulty_band,
        meta={"terms": fractions},
    )

    solve_steps = 2 if difficulty_band == 1 else 4 if difficulty_band == 2 else 8
    _assert_self_verified(problem, FractionAdditionVerifier(), solve_steps)
    return problem
