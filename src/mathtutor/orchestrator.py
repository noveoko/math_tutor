# mathtutor/orchestrator.py

"""Orchestrator — the spine of the Verified Math Tutor.

This is a corrected rewrite. The previous version imported every collaborator
under the wrong name or called it with the wrong signature, then swallowed the
resulting ImportError / TypeError in a bare ``except`` — so the system silently
degraded to "parse -> verify -> crude p_known nudge -> offline template" while
the test suite stayed green.

What changed (and why)
----------------------
* BKT: import ``BKTLearnerState`` (not ``BKTModel``) and call ``observe`` (not
  ``update``). A learner state is now genuinely usable when attached to a
  Session; absent one, a documented crude fallback is used.
* Scaffolding: use ``support_level(p_known)`` (not the non-existent
  ``choose_support``).
* Safety: call ``certify(text, verifier, target)`` and
  ``redact_answers(text, answers, gated)`` with their real signatures.
* Misconceptions: the real ``diagnose(previous_line, student_line)`` compares
  two *consecutive* lines of student working, so the Session now carries
  ``previous_artifact`` and we run diagnosis against it. The module-level
  ``_safe_diagnose(artifact, target)`` seam is preserved verbatim so existing
  monkeypatch-based tests keep working.
* Telemetry: ``eval.telemetry`` exposes ``TelemetrySink`` / ``pseudonymize``
  (there is no ``record``). Persistence now goes through an optional injected
  ``TelemetrySink``; the ``_emit_telemetry(event)`` hook is kept as the
  monkeypatchable, single-argument seam the tests expect.

Imports are now HARD (no ``try/except ImportError``). If a collaborator is
missing or renamed again, you will see it immediately rather than having the
system quietly disable a whole subsystem.

Routing contract (SPEC §5, §13, §14)
--------------------------------------
* The CAS is the single source of mathematical truth.
* The LLM (or offline fallback) handles ONLY language / pedagogy.
* Deterministic turns (verdict, gate checks, worked steps) hit NO model.
* If the LLM raises, the offline tutor takes over transparently.
* Every concrete symbolic claim that would leave the system passes through
  ``safety.claim_cert.certify``; any non-revealed answer is run through
  ``safety.leak_filter.redact_answers``.
* One ``TelemetryEvent`` is emitted per turn.

Turn lifecycle
--------------
1. Parse student input.
2. Verify via the domain Verifier -> Judgment.
3. Update learner state (BKT p_known).
4. If wrong, diagnose a misconception (vs the previous student line).
5. Choose support level from mastery.
6. Build a CoachingContext and ask the LLM (or offline tutor) for prose.
7. Run safety filters on the prose.
8. Emit telemetry.
9. Return a TurnResult.
"""

from __future__ import annotations

import re
import time
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

# ── collaborators (hard imports — fail loudly if renamed) ────────────────────
from mathtutor.llm.offline import CoachingContext, OfflineTutor
from mathtutor.learner.bkt import BKTLearnerState
from mathtutor.tutoring.scaffolding import support_level
from mathtutor.tutoring.misconceptions import diagnose as _diagnose_lines
from mathtutor.safety.claim_cert import certify as _certify
from mathtutor.safety.leak_filter import redact_answers as _redact_answers
from mathtutor.eval.telemetry import TelemetrySink  # noqa: F401 (used for typing/injection)


# ── session & turn types ──────────────────────────────────────────────────────

@dataclass
class Session:
    """Mutable per-student session state.

    The orchestrator modifies this in place on every turn.

    Attributes
    ----------
    bkt_model:
        Optional ``BKTLearnerState``. When provided, mastery is tracked by the
        real four-parameter BKT model via ``observe(kc_id, correct)``. When
        ``None``, a crude monotone fallback nudge is used (kept so the default
        test fixtures behave deterministically).
    previous_artifact:
        The student's most recently parsed line of working, used as the
        "previous" argument to the misconception diagnoser. ``None`` on the
        first turn of a problem.
    gate_open:
        When ``True`` the literal correct answer may appear in coaching prose
        (e.g. once the worked-solution gate has opened). When ``False`` (the
        default) the answer is withheld and ``redact_answers`` runs as a
        backstop — so no literal answer ever leaks while the student is still
        working, regardless of support level.
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
    bkt_model: BKTLearnerState | None = None
    previous_artifact: Artifact | None = None
    gate_open: bool = False
    history: list["TurnResult"] = field(default_factory=list)
    correct_answer_str: str | None = None


@dataclass
class TurnResult:
    """Everything produced by a single handle_turn call."""

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


# ── module-level singletons / seams ──────────────────────────────────────────

_OFFLINE = OfflineTutor()


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


# Human-readable descriptions for the buggy-rule ids emitted by
# tutoring.misconceptions.diagnose. Unknown ids are humanised on the fly.
_MISCONCEPTION_DESCRIPTIONS: dict[str, str] = {
    "distributes_exponent_over_sum":
        "an exponent was distributed over a sum, e.g. (a+b)^2 treated as a^2+b^2",
    "moves_term_across_equals_without_sign_flip":
        "a term was moved across the equals sign without flipping its sign",
    "cancels_term_across_addition":
        "a term was cancelled across an addition rather than a common factor",
    "clears_fraction_multiplying_only_one_term":
        "only the fractional term was multiplied out when clearing a denominator",
    "forgets_to_flip_inequality_on_negative_multiply":
        "the inequality sign was not flipped after multiplying by a negative",
}


def _humanise(rule_id: str) -> str:
    return _MISCONCEPTION_DESCRIPTIONS.get(rule_id, rule_id.replace("_", " "))


# ── safety / collaborator wrappers (runtime-robust, never import-masking) ─────

def _safe_certify(text: str, verifier: Verifier | None, target: Target | None) -> str:
    """Run claim_cert.certify with its real (text, verifier, target) signature.

    Returns the text unchanged if there is nothing to certify against.
    """
    if verifier is None or target is None:
        return text
    try:
        return _certify(text, verifier, target)
    except Exception:  # pragma: no cover — defensive: a bad claim never crashes a turn
        return text


def _answers_for_redaction(session: Session) -> list[str]:
    """Best-effort extraction of literal answer tokens for the leak filter.

    The CAS — never the LLM — is the source. We pull tokens from
    ``correct_answer_str`` and from ``target.payload['answer']``, splitting set
    notation and "or"-lists, and taking the right-hand side of any "x = 3"
    form. Returned strings are de-duplicated while preserving order.
    """
    tokens: list[str] = []

    def _split(raw: str) -> None:
        cleaned = raw.replace("{", "").replace("}", "")
        for chunk in re.split(r",|\bor\b", cleaned):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" in chunk:
                chunk = chunk.split("=")[-1].strip()
            if chunk:
                tokens.append(chunk)

    if session.correct_answer_str:
        _split(session.correct_answer_str)

    if session.target is not None:
        ans = session.target.payload.get("answer")
        if ans is not None:
            _split(str(ans))

    # de-dupe, preserve order
    return list(dict.fromkeys(tokens))


def _safe_redact(text: str, answers: list[str], gated: bool) -> str:
    """Run leak_filter.redact_answers with its real (text, answers, gated) signature."""
    if not gated or not answers:
        return text
    try:
        return _redact_answers(text, answers, gated)
    except Exception:  # pragma: no cover
        return text


def _safe_diagnose(artifact: Artifact, target: Target) -> tuple[str | None, str | None]:
    """Monkeypatchable diagnosis seam (signature preserved for existing tests).

    The real line-to-line diagnoser needs the *previous* student line, which
    this 2-argument seam does not carry — so the default implementation returns
    ``(None, None)`` and the orchestrator falls back to
    :func:`_diagnose_with_previous`. Tests that need a deterministic
    misconception monkeypatch this function directly.
    """
    return None, None


def _diagnose_with_previous(
    previous: Artifact,
    student: Artifact,
) -> tuple[str | None, str | None]:
    """Run the real CAS misconception diagnoser against two consecutive lines.

    ``tutoring.misconceptions.diagnose`` operates on SymPy expressions and
    returns a list of matched buggy-rule ids. We take the first match and pair
    it with a human-readable description.
    """
    try:
        prev_expr = getattr(previous, "expr", previous)
        cur_expr = getattr(student, "expr", student)
        matched = _diagnose_lines(prev_expr, cur_expr)
    except Exception:  # pragma: no cover — a non-SymPy stub artifact, etc.
        return None, None

    if not matched:
        return None, None
    rule_id = matched[0]
    return rule_id, _humanise(rule_id)


def _safe_choose_support(session: Session) -> SupportLevel:
    """Map current mastery to a SupportLevel; fall back to INDEPENDENT on error."""
    try:
        return support_level(session.p_known)
    except Exception:  # pragma: no cover — p_known out of range, etc.
        return SupportLevel.INDEPENDENT


def _safe_bkt_update(session: Session, correct: bool) -> float:
    """Update p_known and return the new probability.

    With a real ``BKTLearnerState`` attached, this calls ``observe`` — note
    that, per the BKT model, a *wrong* answer can still raise p_known slightly
    because the learning-transition term fires regardless of correctness. The
    crude fallback below is strictly monotone (correct raises, wrong lowers)
    and is what the default test fixtures rely on.
    """
    if session.bkt_model is not None and session.kc_id is not None:
        try:
            new_p = session.bkt_model.observe(session.kc_id, correct)
            session.p_known = new_p
            return new_p
        except Exception:  # pragma: no cover
            return session.p_known

    # Crude monotone fallback (no model attached).
    lr = 0.1
    if correct:
        session.p_known = session.p_known + lr * (1.0 - session.p_known)
    else:
        session.p_known = session.p_known * (1.0 - lr)
    return session.p_known


def _emit_telemetry(event: TelemetryEvent) -> None:
    """Single-argument, monkeypatchable telemetry hook.

    The default implementation is a no-op; durable persistence is handled by an
    injected ``TelemetrySink`` on the Orchestrator (see ``handle_turn``). Tests
    replace this function to count emissions.
    """
    return None


def _null_judgment() -> Judgment:
    """Placeholder Judgment for unparseable / unverifiable input."""
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
    """Route a student turn through CAS verification -> tutoring -> safety.

    Parameters
    ----------
    llm:
        An object satisfying the LLMCoach protocol. Defaults to ``OfflineTutor``
        so the system works without any LLM.
    emit_telemetry:
        If True (default), fire the ``_emit_telemetry`` hook and persist to the
        sink (if one was provided) once per turn.
    telemetry_sink:
        Optional ``TelemetrySink`` for durable, append-only persistence of every
        ``TelemetryEvent``.

    Usage
    -----
    >>> orch = Orchestrator()              # uses offline tutor, no persistence
    >>> result = orch.handle_turn(session, "x = 3")
    >>> print(result.coaching_message)
    """

    def __init__(
        self,
        llm: LLMCoach | None = None,
        emit_telemetry: bool = True,
        telemetry_sink: "TelemetrySink | None" = None,
    ) -> None:
        self._llm: LLMCoach = llm if llm is not None else _OFFLINE
        self._emit_telemetry = emit_telemetry
        self._sink = telemetry_sink

    # ── public API ────────────────────────────────────────────────────────────

    def handle_turn(self, session: Session, student_raw: str) -> TurnResult:
        """Process one student input and return a fully-formed TurnResult."""
        t_start = _now_ms()
        p_known_before = session.p_known

        # ── step 1: parse ─────────────────────────────────────────────────
        artifact: Artifact | None = None
        parse_err: str | None = None
        judgment: Judgment | None = None

        verifier = session.verifier
        if verifier is None:
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
            # Seam first (monkeypatchable); fall back to real line-to-line CAS
            # diagnosis against the student's previous line when available.
            misconception_id, misconception_desc = _safe_diagnose(artifact, session.target)
            if misconception_id is None and session.previous_artifact is not None:
                misconception_id, misconception_desc = _diagnose_with_previous(
                    session.previous_artifact, artifact
                )

        # ── step 5: choose support level ──────────────────────────────────
        support = _safe_choose_support(session)

        # ── step 6: build coaching context ────────────────────────────────
        # The literal answer is only ever placed in the context when the gate
        # is open; otherwise it is withheld in every support level so it cannot
        # surface in prose. redact_answers (step 7) is a defence-in-depth
        # backstop for the gated case.
        reveal_answer = session.gate_open
        answer_for_coach = session.correct_answer_str if reveal_answer else None

        ctx = CoachingContext(
            verdict=verdict,
            judgment=judgment,
            support_level=support,
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
        try:
            raw_prose = self._llm.coach(ctx)
        except Exception:
            used_offline = True
            raw_prose = _OFFLINE.coach(ctx)

        # ── step 7: safety filters on prose ──────────────────────────────
        # certify: keep only CAS-verified <<claim>> spans.
        safe_prose = _safe_certify(raw_prose, session.verifier, session.target)
        # redact: when not revealing, strip any literal answer occurrences.
        if not reveal_answer:
            answers = _answers_for_redaction(session)
            safe_prose = _safe_redact(safe_prose, answers, gated=True)

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
            support_level=support.value,
            hint_level=session.hint_level,
            latency_ms=_now_ms() - t_start,
            p_known_before=p_known_before,
            p_known_after=p_known_after,
            policy_id="offline" if used_offline else "llm",
        )
        if self._emit_telemetry:
            _emit_telemetry(event)               # monkeypatchable hook
            if self._sink is not None:           # durable persistence
                try:
                    self._sink.emit(event)
                except Exception:  # pragma: no cover
                    pass

        # ── step 9: assemble result ───────────────────────────────────────
        result = TurnResult(
            verdict=verdict,
            judgment=judgment,
            coaching_message=safe_prose,
            p_known_after=p_known_after,
            misconception_id=misconception_id,
            support_level=support,
            telemetry=event,
            parse_error=parse_err,
            used_offline_tutor=used_offline or isinstance(self._llm, OfflineTutor),
        )

        # Remember this line so the next turn can diagnose against it.
        if artifact is not None:
            session.previous_artifact = artifact

        session.history.append(result)
        return result