# tests/test_orchestrator.py
"""End-to-end tests for the Orchestrator using the offline (deterministic) tutor."""
from __future__ import annotations
import sys, os, types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The package layout already lives at ROOT/mathtutor/, so normal imports work.
import pytest
from mathtutor.contracts import (
    Artifact, Judgment, ParseError, SupportLevel,
    Target, TelemetryEvent, Verdict,
)
from mathtutor.orchestrator import Orchestrator, Session, TurnResult
from mathtutor.llm.offline import CoachingContext, OfflineTutor


# ── stubs ────────────────────────────────────────────────────────────────────

def _make_judgment(correct=False, partial=False, parsed_ok=True, detail=None):
    return Judgment(
        parsed_ok=parsed_ok, value_equivalent=correct, form_ok=correct,
        correct=correct, partial=partial, decidable=True,
        confidence=1.0 if parsed_ok else 0.0, detail=detail or {},
    )


class _StubVerifier:
    domain = "stub"
    def __init__(self, correct_raw, partial_raws=None, misconception_map=None):
        self._correct = {correct_raw} if isinstance(correct_raw, str) else set(correct_raw)
        self._partial = set(partial_raws or [])
        self._misconceptions = misconception_map or {}
    def parse(self, raw: str) -> Artifact:
        if raw.strip() == "UNPARSEABLE":
            raise ParseError("Cannot parse input")
        return Artifact(kind="value", expr=raw.strip(), raw=raw)
    def canonical(self, a): return a.expr
    def accepts(self, student, target):
        raw = student.expr
        correct = raw in self._correct
        partial = (not correct) and raw in self._partial
        detail = {}
        if raw in self._misconceptions:
            detail["misconception_id"] = self._misconceptions[raw]
        return _make_judgment(correct=correct, partial=partial, detail=detail)


def _make_session(correct_raws="x=3", problem="Solve x^2 - 5x + 6 = 0",
                  kc_name="quadratic equations", correct_answer_str="x=2 or x=3"):
    verifier = _StubVerifier(
        correct_raw=correct_raws,
        misconception_map={"x=9": "forgot_to_take_square_root"},
    )
    target = Target(domain="stub", payload={"solutions": [2, 3]})
    return Session(
        user_pseudonym="test_student", kc_id="quad_eq", kc_name=kc_name,
        problem_id="prob_001", problem_statement=problem,
        target=target, verifier=verifier, correct_answer_str=correct_answer_str,
    )


# ── Test 1: correct answer ────────────────────────────────────────────────────

class TestCorrectAnswer:
    def test_verdict_is_correct(self):
        session = _make_session(correct_raws=["x=2", "x=3"])
        result = Orchestrator().handle_turn(session, "x=2")
        assert result.verdict == Verdict.CORRECT

    def test_p_known_increases_after_correct(self):
        session = _make_session(correct_raws=["x=2", "x=3"])
        p_before = session.p_known
        result = Orchestrator().handle_turn(session, "x=2")
        assert result.p_known_after > p_before

    def test_judgment_attached(self):
        session = _make_session(correct_raws=["x=3"])
        result = Orchestrator().handle_turn(session, "x=3")
        assert result.judgment is not None and result.judgment.correct is True

    def test_coaching_message_non_empty(self):
        result = Orchestrator().handle_turn(_make_session(correct_raws=["x=3"]), "x=3")
        assert len(result.coaching_message.strip()) > 0


# ── Test 2: wrong answer, diagnosis, no leak ──────────────────────────────────

class TestWrongAnswerDiagnosisNoLeak:
    def test_verdict_is_wrong(self):
        result = Orchestrator().handle_turn(_make_session(), "x=99")
        assert result.verdict == Verdict.WRONG

    def test_p_known_decreases_after_wrong(self):
        session = _make_session()
        p_before = session.p_known
        result = Orchestrator().handle_turn(session, "x=99")
        assert result.p_known_after < p_before

    def test_no_correct_roots_leaked_in_independent_mode(self):
        session = _make_session(correct_answer_str="x=2 or x=3")
        result = Orchestrator().handle_turn(session, "x=99")
        msg = result.coaching_message
        assert "x=2" not in msg and "x=3" not in msg, f"Roots leaked: {msg!r}"

    def test_misconception_diagnosis_returned(self):
        import mathtutor.orchestrator as orch_mod
        orig = orch_mod._safe_diagnose
        def _fake(artifact, target):
            if artifact.expr == "x=9":
                return "forgot_to_take_square_root", "You squared instead of square-rooted."
            return None, None
        orch_mod._safe_diagnose = _fake
        try:
            result = Orchestrator().handle_turn(_make_session(), "x=9")
        finally:
            orch_mod._safe_diagnose = orig
        assert result.misconception_id == "forgot_to_take_square_root"

    def test_misconception_message_hint_not_value(self):
        import mathtutor.orchestrator as orch_mod
        orig = orch_mod._safe_diagnose
        def _fake(artifact, target):
            return "forgot_to_take_square_root", "You squared instead of square-rooted."
        orch_mod._safe_diagnose = _fake
        try:
            session = _make_session(correct_answer_str="x=2 or x=3")
            result = Orchestrator().handle_turn(session, "x=9")
        finally:
            orch_mod._safe_diagnose = orig
        msg = result.coaching_message
        assert "square" in msg.lower(), f"Expected hint in: {msg!r}"
        assert "x=2" not in msg and "x=3" not in msg, f"Roots leaked: {msg!r}"


# ── Test 3: LLM failure → offline fallback ───────────────────────────────────

class _BrokenLLM:
    def coach(self, context):
        raise RuntimeError("LLM service unavailable")

class TestLLMFailureFallback:
    def test_no_exception_propagates(self):
        result = Orchestrator(llm=_BrokenLLM()).handle_turn(_make_session(), "x=99")
        assert result is not None

    def test_used_offline_tutor_flag_set(self):
        result = Orchestrator(llm=_BrokenLLM()).handle_turn(_make_session(), "x=99")
        assert result.used_offline_tutor is True

    def test_coaching_message_still_present(self):
        result = Orchestrator(llm=_BrokenLLM()).handle_turn(_make_session(), "x=99")
        assert len(result.coaching_message.strip()) > 0

    def test_correct_answer_with_broken_llm(self):
        session = _make_session(correct_raws=["x=3"])
        result = Orchestrator(llm=_BrokenLLM()).handle_turn(session, "x=3")
        assert result.verdict == Verdict.CORRECT and result.used_offline_tutor is True


# ── Test 4: telemetry emitted per turn ────────────────────────────────────────

class TestTelemetryEmittedPerTurn:
    def test_telemetry_attached(self):
        result = Orchestrator().handle_turn(_make_session(), "x=3")
        assert result.telemetry is not None

    def test_telemetry_session_id(self):
        session = _make_session()
        result = Orchestrator().handle_turn(session, "x=3")
        assert result.telemetry.session_id == session.session_id

    def test_telemetry_verdict(self):
        result = Orchestrator().handle_turn(_make_session(), "x=99")
        assert result.telemetry.verdict == Verdict.WRONG.value

    def test_telemetry_p_known_fields(self):
        result = Orchestrator().handle_turn(_make_session(), "x=3")
        assert result.telemetry.p_known_before is not None
        assert result.telemetry.p_known_after is not None

    def test_telemetry_kc_and_problem_id(self):
        result = Orchestrator().handle_turn(_make_session(), "x=3")
        assert result.telemetry.kc_id == "quad_eq"
        assert result.telemetry.problem_id == "prob_001"

    def test_one_event_per_turn(self):
        import mathtutor.orchestrator as orch_mod
        emitted = []
        orig = orch_mod._emit_telemetry
        orch_mod._emit_telemetry = emitted.append
        try:
            session = _make_session()
            orch = Orchestrator()
            orch.handle_turn(session, "x=99")
            orch.handle_turn(session, "x=3")
        finally:
            orch_mod._emit_telemetry = orig
        assert len(emitted) == 2
        assert len({e.event_id for e in emitted}) == 2

    def test_telemetry_serialisable(self):
        result = Orchestrator().handle_turn(_make_session(), "x=3")
        restored = TelemetryEvent.from_json(result.telemetry.to_json())
        assert restored.session_id == result.telemetry.session_id
        assert restored.verdict == result.telemetry.verdict


# ── Bonus: parse failure ──────────────────────────────────────────────────────

class TestParseFailure:
    def test_unparseable_gives_abstain(self):
        result = Orchestrator().handle_turn(_make_session(), "UNPARSEABLE")
        assert result.verdict == Verdict.ABSTAIN

    def test_parse_error_string_attached(self):
        result = Orchestrator().handle_turn(_make_session(), "UNPARSEABLE")
        assert result.parse_error and len(result.parse_error) > 0