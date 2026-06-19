# tests/test_curriculum_loader.py
#
# Tests for the data-driven Curriculum loaders (from_dict / from_json /
# from_json_file) added so any subject can be authored as data rather than
# hand-coded Python. These are independent of test_curriculum.py.

import json

import pytest

from mathtutor.domain.curriculum import Curriculum, CurriculumError, build_sample_curriculum
from mathtutor.learner.scheduling import select_next


# A small calculus graph, deliberately listed OUT of dependency order and with
# two "join" nodes (definite_integral and u_substitution each need two prereqs)
# to prove the loader reorders correctly.
CALCULUS = {
    "subject": "calculus_intro",
    "knowledge_components": [
        {"id": "u_substitution", "prerequisites": ["definite_integral", "chain_rule"]},
        {"id": "limits_intro", "prerequisites": []},
        {"id": "definite_integral", "prerequisites": ["antiderivatives", "continuity"]},
        {"id": "chain_rule", "prerequisites": ["power_rule"]},
        {"id": "continuity", "prerequisites": ["limits_intro"]},
        {"id": "derivative_definition", "prerequisites": ["limits_intro"]},
        {"id": "power_rule", "prerequisites": ["derivative_definition"]},
        {"id": "antiderivatives", "prerequisites": ["power_rule"]},
    ],
}


class TestFromDict:
    def test_loads_all_kcs(self):
        c = Curriculum.from_dict(CALCULUS)
        assert len(c.kcs) == 8
        assert c.subject == "calculus_intro"

    def test_reorders_to_valid_dependency_order(self):
        """Scrambled input is reordered so prereqs precede dependents."""
        c = Curriculum.from_dict(CALCULUS)
        order = [kc.id for kc in c.kcs]
        pos = {kc_id: i for i, kc_id in enumerate(order)}
        for kc in c.kcs:
            for prereq in kc.prerequisites:
                assert pos[prereq] < pos[kc.id]

    def test_topological_order_succeeds(self):
        c = Curriculum.from_dict(CALCULUS)
        # Should not raise (graph is acyclic) and cover every KC.
        assert len(c.topological_order()) == 8

    def test_ready_at_start_is_only_root(self):
        c = Curriculum.from_dict(CALCULUS)
        assert c.ready_kcs(set()) == ["limits_intro"]

    def test_defaults_applied_for_omitted_fields(self):
        c = Curriculum.from_dict({"knowledge_components": [{"id": "solo"}]})
        kc = c.get("solo")
        assert kc.name == "solo"            # name defaults to id
        assert kc.prerequisites == []
        assert kc.difficulty_band == 1
        assert kc.generators == []
        assert kc.verifier_domain == ""

    def test_bare_list_accepted(self):
        c = Curriculum.from_dict([{"id": "a"}, {"id": "b", "prerequisites": ["a"]}])
        assert {kc.id for kc in c.kcs} == {"a", "b"}

    def test_single_generator_string_is_wrapped(self):
        c = Curriculum.from_dict([{"id": "a", "generators": "gen_a"}])
        assert c.get("a").generators == ["gen_a"]

    def test_subject_kwarg_overrides_file_metadata(self):
        c = Curriculum.from_dict(CALCULUS, subject="my_calc")
        assert c.subject == "my_calc"


class TestLoaderErrors:
    def test_cycle_raises(self):
        with pytest.raises(CurriculumError, match="Cycle"):
            Curriculum.from_dict([
                {"id": "a", "prerequisites": ["b"]},
                {"id": "b", "prerequisites": ["a"]},
            ])

    def test_unknown_prerequisite_raises(self):
        with pytest.raises(CurriculumError, match="unknown prerequisites"):
            Curriculum.from_dict([{"id": "a", "prerequisites": ["ghost"]}])

    def test_duplicate_id_raises(self):
        with pytest.raises(CurriculumError, match="Duplicate"):
            Curriculum.from_dict([{"id": "a"}, {"id": "a"}])

    def test_missing_id_raises(self):
        with pytest.raises(CurriculumError, match="'id'"):
            Curriculum.from_dict([{"name": "no id here"}])

    def test_bad_difficulty_band_raises(self):
        with pytest.raises(CurriculumError, match="difficulty_band"):
            Curriculum.from_dict([{"id": "a", "difficulty_band": "hard"}])

    def test_bad_prerequisites_type_raises(self):
        with pytest.raises(CurriculumError, match="prerequisites"):
            Curriculum.from_dict([{"id": "a", "prerequisites": "b"}])

    def test_missing_knowledge_components_key_raises(self):
        with pytest.raises(CurriculumError, match="knowledge_components"):
            Curriculum.from_dict({"subject": "x"})


class TestFromJson:
    def test_from_json_string(self):
        c = Curriculum.from_json(json.dumps(CALCULUS))
        assert len(c.kcs) == 8

    def test_invalid_json_raises(self):
        with pytest.raises(CurriculumError, match="Invalid JSON"):
            Curriculum.from_json("{not valid json")

    def test_from_json_file(self, tmp_path):
        p = tmp_path / "trig.json"
        p.write_text(json.dumps(CALCULUS), encoding="utf-8")
        c = Curriculum.from_json_file(p)
        assert len(c.kcs) == 8
        # No explicit subject passed, so the file STEM wins: "trig"
        # (from_json_file resolves subject=subject or p.stem before parsing).
        assert c.subject == "trig"

    def test_from_json_file_uses_stem_when_no_subject(self, tmp_path):
        p = tmp_path / "trigonometry.json"
        p.write_text(json.dumps({"knowledge_components": [{"id": "a"}]}), encoding="utf-8")
        c = Curriculum.from_json_file(p)
        assert c.subject == "trigonometry"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(CurriculumError, match="Could not read"):
            Curriculum.from_json_file(tmp_path / "does_not_exist.json")


class TestRoundTrip:
    def test_to_dict_from_dict_round_trip(self):
        original = build_sample_curriculum()
        rebuilt = Curriculum.from_dict(original.to_dict())
        assert {kc.id for kc in rebuilt.kcs} == {kc.id for kc in original.kcs}
        assert rebuilt.subject == original.subject

    def test_to_json_from_json_round_trip(self):
        original = Curriculum.from_dict(CALCULUS)
        rebuilt = Curriculum.from_json(original.to_json())
        # prerequisite edges survive the round trip
        for kc in original.kcs:
            assert set(rebuilt.get(kc.id).prerequisites) == set(kc.prerequisites)


class TestSchedulerIntegration:
    """A data-loaded curriculum must drive select_next just like a hand-built one."""

    def test_select_next_on_loaded_curriculum(self):
        c = Curriculum.from_dict(CALCULUS)
        result = select_next(
            curriculum=c,
            mastered_set=set(),
            retention_states={},
            now_ts=1_000.0,
            k=5,
        )
        assert result == ["limits_intro"]

    def test_deeper_kcs_gated_until_prereqs_met(self):
        c = Curriculum.from_dict(CALCULUS)
        result = select_next(
            curriculum=c,
            mastered_set={"limits_intro"},
            retention_states={},
            now_ts=1_000.0,
            k=5,
        )
        # With only limits_intro mastered, integration/chain-rule KCs stay gated.
        assert "u_substitution" not in result
        assert "definite_integral" not in result
        for kc_id in result:
            assert kc_id in {"continuity", "derivative_definition"}