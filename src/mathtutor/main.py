# mathtutor/main.py
"""Minimal driver: a real, runnable tutoring loop.

This shows the intended way to assemble the pieces — pick a verifier, wrap the
answer in a Target under payload['answer'], build a Session, and loop
handle_turn over student inputs. Attaching a BKTLearnerState turns on real
mastery tracking; attaching a TelemetrySink turns on durable logging.

Run:  python -m mathtutor.main      (or: python main.py from this folder)
"""

from __future__ import annotations

from mathtutor.contracts import Target
from mathtutor.domain.verifiers.polynomial import PolynomialVerifier
from mathtutor.learner.bkt import BKTLearnerState
from mathtutor.orchestrator import Orchestrator, Session


def build_session() -> Session:
    """A quadratic problem with roots {2, 3}."""
    verifier = PolynomialVerifier()
    target = Target(domain="polynomial", payload={"answer": "{2, 3}"})
    return Session(
        user_pseudonym="demo-student",
        kc_id="solve_quadratic_eq",
        kc_name="quadratic equations",
        problem_id="p1",
        problem_statement="Solve x^2 - 5x + 6 = 0",
        target=target,
        verifier=verifier,
        correct_answer_str="{2, 3}",
        bkt_model=BKTLearnerState(),   # real BKT mastery tracking
        gate_open=False,               # answer stays hidden until the gate opens
    )


def run_once(orch: Orchestrator, session: Session, student_input: str) -> None:
    result = orch.handle_turn(session, student_input)
    print(f"\n  you typed     : {student_input!r}")
    print(f"  verdict       : {result.verdict.value}")
    print(f"  support level : {result.support_level.value}")
    print(f"  p_known       : {session.p_known:.3f}")
    if result.misconception_id:
        print(f"  misconception : {result.misconception_id}")
    print(f"  tutor says    : {result.coaching_message}")


def main() -> None:
    orch = Orchestrator()              # defaults to the offline template tutor
    session = build_session()

    print("Problem:", session.problem_statement)
    print("(type a solution, e.g. {2, 3}; blank line to quit)")

    # A scripted demo so the file runs non-interactively too:
    for attempt in ("{2, 4}", "{2, 3}"):
        run_once(orch, session, attempt)

    # Uncomment for an interactive loop:
    # while True:
    #     try:
    #         line = input("\n> ").strip()
    #     except (EOFError, KeyboardInterrupt):
    #         break
    #     if not line:
    #         break
    #     run_once(orch, session, line)


if __name__ == "__main__":
    main()