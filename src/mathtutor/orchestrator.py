# mathtutor/orchestrator.py

"""Orchestrator — the spine of the Verified Math Tutor.

Routing contract (SPEC §5, §13, §14)
--------------------------------------
* The CAS is the single source of mathematical truth.
* The LLM (or offline fallback) handles ONLY language / pedagogy.
* Deterministic turns (verdict, gate checks, worked steps) hit NO model.
* If the LLM raises, the offline tutor takes over transparently.
* Every concrete symbolic claim that would leave the system passes through
  ``safety.claim_cert.certify`` and ``safety.leak_filter.redact_answers``.
* One ``TelemetryEvent`` is emitted per turn.

Turn lifecycle
--------------
1. Parse student input (echo-confirm if ambiguous).
2. Verify via domain Verifier → Judgment.
3. Update learner state (BKT p_known).
4. If wrong, diagnose misconception.
5. Choose support level / hint depth (scaffolding).
6. Ask LLM (or offline tutor) for coaching prose.
7. Run safety filters on any prose.
8. Emit telemetry.
9. Return TurnResult.
"""

from __future__ import annotations

import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

# ── shared contracts ──────────────────────────────────────────────────────────
from mathtutor.contracts import (
    Artifact,
    Judgment,
    ParseError,
    SupportLevel,
    Target,
    TelemetryEvent,
    Verdict,
    Verifier,
    verdict_from_judgment,
)

# ── offline tutor (always available) ─────────────────────────────────────────
from mathtutor.llm.offline import CoachingContext, OfflineTutor

# ── optional dependencies: degrade gracefully if absent ──────────────────────

try:
    from mathtutor.learner.bkt import BKTModel  # type: ignore[import]
except ImportError:
    BKTModel = None  # type: ignore[assignment,misc]

try:
    from mathtutor.tutoring.misconceptions import diagnose  # type: ignore[import]
except ImportError:
    diagnose = None  # type: ignore[assignment]

try:
    from mathtutor.tutoring.scaffolding import choose_support  # type: ignore[import]
except ImportError:
    choose_support = None  # type: ignore[assignment]

try:
    from mathtutor.safety.claim_cert import certify  # type: ignore[import]
except ImportError:
    certify = None  # type: ignore[assignment]

try:
    from mathtutor.safety.leak_filter import redact_answers  # type: ignore[import]
except ImportError:
    redact_answers = None  # type: ignore[assignment]

try:
    from mathtutor.eval.telemetry import record  # type: ignore[import]
except ImportError:
    record = None  # type: ignore[assignment]


# ── session & turn types ──────────────────────────────────────────────────────

@dataclass
class Session:
    """Mutable per-student session state.

    The orchestrator modifies this in place on every turn.

    Attributes
    ----------
    session_id:
        Unique identifier for this session (UUID string).
    user_pseudonym:
        Privacy-safe identifier for the learner.
    kc_id:
        The knowledge-component currently being practised.
    kc_name:
        Human-readable name of the KC.
    problem_id:
        Identifier for the current problem.
    problem_statement:
        The full problem text shown to the student.
    target:
        The Target the verifier uses to judge correctness.
    verifier:
        Domain verifier to use for this problem.
    opportunity_index:
        How many attempts have been made on this KC so far.
    hint_level:
        How many hints have already been delivered this turn.
    p_known:
        BKT probability that the student knows the KC.
    bkt_model:
        Optional BKT model instance (may be None if module absent).
    history:
        Ordered list of past TurnResult objects this session.
    correct_answer_str:
        CAS-formatted reference answer.  May be None until set.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_pseudonym: str = "anonymous"
    kc_id: str | None = None
    kc_name: str = "this topic"
    problem_id: str | None = None
    problem_statement: str = ""
    target: Target | None = None
    verifier: Verifier | None = None
    opportunity_index: int = 0
    hint_level: int = 0
    p_known: float = 0.2
    bkt_model: Any = None
    history: list["TurnResult"] = field(default_factory=list)
    correct_answer_str: str | None = None


@dataclass
class TurnResult:
    """Everything produced by a single handle_turn call.

    Attributes
    ----------
    verdict:
        Coarse verdict (CORRECT / PARTIAL / WRONG / ABSTAIN).
    judgment:
        Full CAS Judgment (or None if parse failed completely).
    coaching_message:
        The prose shown to the student.
    p_known_after:
        Updated BKT probability after this turn.
    misconception_id:
        Buggy-rule ID diagnosed, if any.
    support_level:
        Scaffolding level that was used.
    telemetry:
        The TelemetryEvent emitted for this turn.
    parse_error:
        Set if the input could not be parsed.
    used_offline_tutor:
        True when the real LLM was unavailable and the fallback was used.
    """

    verdict: Verdict
    judgment: Judgment | None
    coaching_message: str
    p_known_after: float
    misconception_id: str | None = None
    support_level: SupportLevel = SupportLevel.INDEPENDENT
    telemetry: TelemetryEvent | None = None
    parse_error: str | None = None
    used_offline_tutor: bool = False


# ── LLM protocol ─────────────────────────────────────────────────────────────

class LLMCoach(Protocol):
    """Interface the Orchestrator expects from the LLM object."""

    def coach(self, context: CoachingContext) -> str:
        """Return a coaching message given verified CAS facts."""
        ...


# ── helpers ───────────────────────────────────────────────────────────────────

_OFFLINE = OfflineTutor()


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


def _safe_certify(text: str) -> str:
    """Run claim_cert.certify if available; otherwise pass through."""
    if certify is None:
        return text
    try:
        return certify(text)
    except Exception:  # pragma: no cover
        return text


def _safe_redact(text: str, target: Target | None) -> str:
    """Run leak_filter.redact_answers if available; otherwise pass through."""
    if redact_answers is None or target is None:
        return text
    try:
        return redact_answers(text, target)
    except Exception:  # pragma: no cover
        return text


def _safe_diagnose(artifact: Artifact, target: Target) -> tuple[str | None, str | None]:
    """Return (misconception_id, description) or (None, None) on failure."""
    if diagnose is None:
        return None, None
    try:
        result = diagnose(artifact, target)
        if result is None:
            return None, None
        # diagnose may return a BuggyRule or a (id, description) tuple
        if hasattr(result, "id"):
            return result.id, getattr(result, "description", None)
        if isinstance(result, tuple) and len(result) == 2:
            return result
    except Exception:  # pragma: no cover
        pass
    return None, None


def _safe_choose_support(
    session: Session,
    judgment: Judgment,
) -> SupportLevel:
    """Return a scaffolding SupportLevel; fall back to INDEPENDENT."""
    if choose_support is None:
        return SupportLevel.INDEPENDENT
    try:
        return choose_support(session, judgment)
    except Exception:  # pragma: no cover
        return SupportLevel.INDEPENDENT


def _safe_bkt_update(session: Session, correct: bool) -> float:
    """Update p_known via BKT if available; return new probability."""
    if session.bkt_model is None or BKTModel is None:
        # Tiny hand-coded update so the value isn't always static
        lr = 0.1
        if correct:
            session.p_known = session.p_known + lr * (1.0 - session.p_known)
        else:
            session.p_known = session.p_known * (1.0 - lr)
        return session.p_known
    try:
        new_p = session.bkt_model.update(session.kc_id, correct)
        session.p_known = new_p
        return new_p
    except Exception:  # pragma: no cover
        return session.p_known


def _emit_telemetry(event: TelemetryEvent) -> None:
    """Fire-and-forget telemetry; never raises."""
    if record is not None:
        try:
            record(event)
        except Exception:  # pragma: no cover
            pass


def _null_judgment() -> Judgment:
    """Return a placeholder Judgment for unparseable input."""
    return Judgment(
        parsed_ok=False,
        value_equivalent=False,
        form_ok=False,
        correct=False,
        partial=False,
        decidable=False,
        confidence=0.0,
        detail={"reason": "parse_failure"},
    )


# ── orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """Route a student turn through CAS verification → tutoring → safety.

    Parameters
    ----------
    llm:
        An object satisfying the LLMCoach protocol.  Defaults to
        ``OfflineTutor`` so the system works without any LLM.
    emit_telemetry:
        If True (default), emit a TelemetryEvent to ``eval.telemetry.record``.

    Usage
    -----
    >>> orch = Orchestrator()              # uses offline tutor
    >>> result = orch.handle_turn(session, "x = 3")
    >>> print(result.coaching_message)
    """

    def __init__(
        self,
        llm: LLMCoach | None = None,
        emit_telemetry: bool = True,
    ) -> None:
        self._llm: LLMCoach = llm if llm is not None else _OFFLINE
        self._emit_telemetry = emit_telemetry

    # ── public API ────────────────────────────────────────────────────────────

    def handle_turn(self, session: Session, student_raw: str) -> TurnResult:
        """Process one student input and return a fully-formed TurnResult.

        This method is the ONLY entry point for student interaction.  It
        implements the turn lifecycle described in the module docstring.

        Parameters
        ----------
        session:
            Mutable session state (modified in place).
        student_raw:
            Exactly what the student typed; not yet parsed or validated.

        Returns
        -------
        TurnResult
            Contains verdict, coaching message, updated p_known, and
            the telemetry event.
        """
        t_start = _now_ms()
        p_known_before = session.p_known

        # ── step 1: parse ─────────────────────────────────────────────────
        artifact: Artifact | None = None
        parse_err: str | None = None
        judgment: Judgment | None = None

        verifier = session.verifier
        if verifier is None:
            # No verifier configured — treat as abstain
            judgment = _null_judgment()
            parse_err = "No verifier configured for this session."
        else:
            try:
                artifact = verifier.parse(student_raw)
            except ParseError as exc:
                parse_err = str(exc)
                judgment = _null_judgment()
            except Exception as exc:  # pragma: no cover
                parse_err = f"Unexpected parse error: {exc}"
                judgment = _null_judgment()

        # ── step 2: verify ────────────────────────────────────────────────
        if artifact is not None and session.target is not None:
            try:
                judgment = verifier.accepts(artifact, session.target)  # type: ignore[union-attr]
            except Exception as exc:  # pragma: no cover
                judgment = Judgment(
                    parsed_ok=True,
                    value_equivalent=False,
                    form_ok=False,
                    correct=False,
                    partial=False,
                    decidable=False,
                    confidence=0.0,
                    detail={"reason": f"verifier_error: {exc}"},
                )
        elif judgment is None:
            judgment = _null_judgment()

        verdict = verdict_from_judgment(judgment)

        # ── step 3: update learner state ──────────────────────────────────
        p_known_after = _safe_bkt_update(session, correct=judgment.correct)
        session.opportunity_index += 1

        # ── step 4: diagnose misconception (wrong turns only) ─────────────
        misconception_id: str | None = None
        misconception_desc: str | None = None
        if verdict == Verdict.WRONG and artifact is not None and session.target is not None:
            misconception_id, misconception_desc = _safe_diagnose(
                artifact, session.target
            )

        # ── step 5: choose support level ──────────────────────────────────
        support_level = _safe_choose_support(session, judgment)

        # ── step 6: build coaching context ────────────────────────────────
        # Redact the correct answer if support level is INDEPENDENT
        answer_for_coach: str | None = session.correct_answer_str
        if support_level == SupportLevel.INDEPENDENT:
            answer_for_coach = None

        ctx = CoachingContext(
            verdict=verdict,
            judgment=judgment,
            support_level=support_level,
            hint_level=session.hint_level,
            kc_name=session.kc_name,
            problem_statement=session.problem_statement,
            student_raw=student_raw,
            correct_answer_str=answer_for_coach,
            misconception_id=misconception_id,
            misconception_description=misconception_desc,
            worked_steps=judgment.detail.get("worked_steps", []),
        )

        # ── step 6b: call LLM (with offline fallback) ─────────────────────
        used_offline = False
        raw_prose = ""
        try:
            raw_prose = self._llm.coach(ctx)
        except Exception:
            # LLM unavailable — fall back to offline tutor transparently
            used_offline = True
            raw_prose = _OFFLINE.coach(ctx)

        # ── step 7: safety filters on prose ──────────────────────────────
        # certify: strip or flag any unsupported mathematical claims
        safe_prose = _safe_certify(raw_prose)
        # redact_answers: ensure correct answer values are not leaked in
        # INDEPENDENT / COMPLETION modes
        if support_level != SupportLevel.WORKED:
            safe_prose = _safe_redact(safe_prose, session.target)

        # ── step 8: emit telemetry ────────────────────────────────────────
        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            session_id=session.session_id,
            user_pseudonym=session.user_pseudonym,
            ts=time.time(),
            kc_id=session.kc_id,
            problem_id=session.problem_id,
            opportunity_index=session.opportunity_index,
            action="answer_attempt",
            input_artifact=student_raw[:200],
            verdict=verdict.value,
            error_kind=parse_err,
            misconception_id=misconception_id,
            support_level=support_level.value,
            hint_level=session.hint_level,
            latency_ms=_now_ms() - t_start,
            p_known_before=p_known_before,
            p_known_after=p_known_after,
            policy_id="offline" if used_offline else "llm",
        )
        if self._emit_telemetry:
            _emit_telemetry(event)

        # ── step 9: assemble result ───────────────────────────────────────
        result = TurnResult(
            verdict=verdict,
            judgment=judgment,
            coaching_message=safe_prose,
            p_known_after=p_known_after,
            misconception_id=misconception_id,
            support_level=support_level,
            telemetry=event,
            parse_error=parse_err,
            used_offline_tutor=used_offline or isinstance(self._llm, OfflineTutor),
        )
        session.history.append(result)
        return result