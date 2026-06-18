import pytest

from mathtutor.domain.curriculum import (
    Curriculum,
    CurriculumError,
    build_sample_curriculum,
)
from mathtutor.contracts import KnowledgeComponent


def test_cycle_insertion_raises() -> None:
    curriculum = Curriculum()

    curriculum.add(
        KnowledgeComponent(
            id="a",
            name="A",
            prerequisites=[],
            verifier_domain="x",
            generator="g1",
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="b",
            name="B",
            prerequisites=["a"],
            verifier_domain="x",
            generator="g2",
        )
    )

    with pytest.raises(CurriculumError):
        curriculum.add(
            KnowledgeComponent(
                id="a2",
                name="A2",
                prerequisites=["b", "a2"],  # self-cycle
                verifier_domain="x",
                generator="g3",
            )
        )


def test_topological_order_respects_edges() -> None:
    curriculum = build_sample_curriculum()
    order = curriculum.topological_order()
    pos = {kc: i for i, kc in enumerate(order)}

    for kc_id in order:
        for prereq in curriculum.prerequisites(kc_id):
            assert pos[prereq] < pos[kc_id]


def test_unmet_prerequisites() -> None:
    curriculum = build_sample_curriculum()

    mastered = {"fraction_basics", "fraction_operations"}

    unmet = curriculum.unmet_prerequisites(
        "linear_one_step",
        mastered,
    )

    assert unmet == ["simplify_expressions"]


def test_ready_kcs() -> None:
    curriculum = build_sample_curriculum()

    ready = curriculum.ready_kcs(set())
    assert ready == ["fraction_basics"]

    mastered = {"fraction_basics"}
    ready = curriculum.ready_kcs(mastered)
    assert "fraction_operations" in ready
    assert "linear_one_step" not in ready

    mastered = {
        "fraction_basics",
        "fraction_operations",
        "simplify_expressions",
        "linear_one_step",
        "linear_multi_step",
        "distributive_property",
    }
    ready = curriculum.ready_kcs(mastered)
    assert "factoring_quadratics" in ready
