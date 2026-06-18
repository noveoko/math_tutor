# tests/test_generators.py

import pytest

from mathtutor.domain.generators import generate, GeneratorError
import mathtutor.domain.generators as generators


GENS = [
    "linear_equation",
    "quadratic_equation",
    "fraction_addition",
]


def test_200_generations():
    count = 0
    for gen in GENS:
        for band in [1, 2, 3]:
            for seed in range(25):
                p = generate(gen, difficulty_band=band, seed=seed)
                assert p is not None
                count += 1
    assert count >= 200


def test_determinism():
    for gen in GENS:
        p1 = generate(gen, difficulty_band=2, seed=123)
        p2 = generate(gen, difficulty_band=2, seed=123)
        assert p1 == p2


def test_broken_reference_answer_raises(monkeypatch):
    original = generators.generate_linear_equation

    def broken(*, difficulty_band, seed):
        p = original(difficulty_band=difficulty_band, seed=seed)
        bad = p.__class__(
            id=p.id,
            kc_id=p.kc_id,
            domain=p.domain,
            prompt_text=p.prompt_text,
            parsed_target=p.parsed_target,
            reference_answer="999999",
            difficulty_band=p.difficulty_band,
            meta=p.meta,
        )
        generators._assert_self_verified(
            bad,
            generators.LinearEquationVerifier(),
            2,
        )
        return bad

    monkeypatch.setitem(generators._REGISTRY, "linear_equation", broken)

    with pytest.raises(GeneratorError):
        generate("linear_equation", difficulty_band=1, seed=1)
