# tests/test_curriculum.py
#
# Renamed from tests/curriculum.py so pytest actually collects it (the default
# collection pattern is test_*.py). The KnowledgeComponent constructions are
# fixed to match contracts.KnowledgeComponent: `generators` is a list and
# `difficulty_band` is required. Two integration tests at the bottom exercise a
# REAL Curriculum through select_next and propagate, which only works now that
# Curriculum exposes a `.kcs` property.

import pytest

from mathtutor.domain.curriculum import (
    Curriculum,
    CurriculumError,
    build_sample_curriculum,
)
from mathtutor.contracts import KnowledgeComponent
from mathtutor.learner.bkt import BKTLearnerState, propagate
from mathtutor.learner.scheduling import select_next


def _kc(kc_id: str, prereqs: list[str], band: int = 1) -> KnowledgeComponent:
    """Build a contract-correct KnowledgeComponent for tests."""
    return KnowledgeComponent(
        id=kc_id,
        name=kc_id.replace("_", " ").title(),
        prerequisites=prereqs,
        verifier_domain="x",
        difficulty_band=band,
        generators=[f"gen_{kc_id}"],
    )


def test_cycle_insertion_raises() -> None:
    curriculum = Curriculum()
    curriculum.add(_kc("a", []))
    curriculum.add(_kc("b", ["a"]))

    with pytest.raises(CurriculumError):
        curriculum.add(_kc("a2", ["b", "a2"]))  # self-cycle / forward-ref


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
    unmet = curriculum.unmet_prerequisites("linear_one_step", mastered)
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


# ---------------------------------------------------------------------------
# Regression tests: build_sample_curriculum must not raise, and the real
# Curriculum must drive scheduling + BKT propagation via the .kcs property.
# These are exactly the paths the broken contract used to crash on.
# ---------------------------------------------------------------------------

def test_build_sample_curriculum_does_not_raise() -> None:
    """The contract-mismatch bug raised TypeError here; it must not anymore."""
    curriculum = build_sample_curriculum()
    assert len(curriculum.kcs) == 8
    # .kcs yields real KnowledgeComponent objects with the fields collaborators read
    first = curriculum.kcs[0]
    assert first.id == "fraction_basics"
    assert first.prerequisites == []


def test_select_next_with_real_curriculum() -> None:
    """select_next reads curriculum.kcs; with nothing mastered/seen, only the
    single no-prerequisite KC is offered as new work."""
    curriculum = build_sample_curriculum()
    result = select_next(
        curriculum=curriculum,
        mastered_set=set(),
        retention_states={},
        now_ts=1_000.0,
        k=5,
    )
    assert result == ["fraction_basics"]


def test_select_next_respects_real_prereq_chain() -> None:
    """Once fraction_basics is mastered, fraction_operations becomes eligible
    but deeper KCs stay gated."""
    curriculum = build_sample_curriculum()
    result = select_next(
        curriculum=curriculum,
        mastered_set={"fraction_basics"},
        retention_states={},
        now_ts=1_000.0,
        k=5,
    )
    assert "simplify_expressions" not in result   # prereq not yet met
    assert "linear_one_step" not in result
    # fraction_operations may appear (its only prereq is mastered)
    for kc_id in result:
        assert kc_id in {"fraction_operations"}

def test_registered_generators_are_referenced_by_curriculum():
    """Generators that exist must be claimed by at least one KC.
    The reverse is allowed: a KC may reference a generator not yet built
    (that gap is the roadmap, and is intentional)."""
    from mathtutor.domain import generators as gen
    curriculum = build_sample_curriculum()
    referenced = {name for kc in curriculum.kcs for name in kc.generators}
    registered = set(gen._REGISTRY)
    orphans = registered - referenced
    assert not orphans, f"Registered generators no KC references: {orphans}"
    
def test_propagate_with_real_curriculum_nudges_dependents() -> None:
    """propagate iterates curriculum.kcs; a mastered prerequisite should nudge
    its direct dependent's mastery probability upward."""
    curriculum = build_sample_curriculum()
    state = BKTLearnerState()

    # Master fraction_basics (3 correct answers cross the 0.95 threshold).
    state.register("fraction_basics")
    for _ in range(3):
        state.observe("fraction_basics", correct=True)
    assert state.mastered("fraction_basics")

    propagate(state, curriculum)

    # fraction_operations (direct dependent) is nudged from L0=0.20 by NUDGE=0.05
    p_ops = state.p_mastered("fraction_operations")
    assert p_ops == pytest.approx(0.25, abs=1e-9)