"""
demo.py — a scripted, end-to-end tour of the whole MathTutor stack.

Run it from the project root (with your venv active):

    python demo.py

It is non-interactive: it auto-answers problems (one deliberate wrong answer to
show feedback, then correct answers) so you can just watch the pipeline work.

What it exercises, in order:
  1. The data layer        — list & load a bundled curriculum
  2. Scheduling            — select_next picks the next eligible KC
  3. Generators + verifiers— a CAS-verified problem per KC
  4. The orchestrator turn — parse -> verify -> coach -> safety filters
  5. BKT mastery           — mastery climbs; the scheduler unlocks the next KC
  6. Telemetry + analytics — events are persisted and summarised
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from mathtutor.domain.curriculum import (
    Curriculum,
    CurriculumError,
    load_curriculum,
    list_curricula,
)
from mathtutor.domain.generators import generate
from mathtutor.learner.bkt import BKTLearnerState
from mathtutor.learner.scheduling import RetentionState, update_after_review, select_next
from mathtutor.orchestrator import Orchestrator, Session
from mathtutor.contracts import Verdict
from mathtutor.eval.telemetry import TelemetrySink, pseudonymize

# Map a generated problem's domain to the verifier that judges it.
from mathtutor.domain.verifiers.linear_equation import EquationVerifier
from mathtutor.domain.verifiers.polynomial import PolynomialVerifier
from mathtutor.domain.verifiers.fraction import FractionVerifier

VERIFIER_FOR_DOMAIN = {
    "linear_equation": EquationVerifier,
    "quadratic_equation": PolynomialVerifier,
    "fraction_addition": FractionVerifier,
}

# A small practice curriculum whose KCs map onto the three registered
# generators, so every KC can actually produce problems.
PRACTICE = {
    "subject": "practice",
    "knowledge_components": [
        {"id": "fractions",  "name": "Adding Fractions",       "generators": ["fraction_addition"]},
        {"id": "linear",     "name": "Linear Equations",       "prerequisites": ["fractions"], "generators": ["linear_equation"]},
        {"id": "quadratics", "name": "Quadratic Equations",    "prerequisites": ["linear"],    "generators": ["quadratic_equation"]},
    ],
}

RULE = "─" * 64


def banner(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def wrong_answer(reference: str) -> str:
    """A clearly-incorrect but parseable answer for any of our domains."""
    return "1" if reference.strip() in {"0", "1"} else "0"


def show_data_layer() -> None:
    banner("1. DATA LAYER — bundled curricula")
    available = list_curricula()
    print("Bundled curricula:", available or "(none found under mathtutor/data/curricula)")
    if not available:
        print("Drop a JSON file there to see it here; skipping bundled load.")
        return
    try:
        c = load_curriculum(available[0])
        print(f"Loaded '{c.subject}' with {len(c.kcs)} knowledge components.")
        print("Teaching order:", " -> ".join(c.topological_order()))
        print("Can start with:", c.ready_kcs(set()))
    except CurriculumError as exc:
        print("Could not load bundled curriculum:", exc)


def run_session() -> Path:
    banner("2-5. ADAPTIVE SESSION — scheduling, generation, mastery")

    curriculum = Curriculum.from_dict(PRACTICE)
    bkt = BKTLearnerState()

    events_path = Path(tempfile.gettempdir()) / "mathtutor_demo_events.jsonl"
    if events_path.exists():
        events_path.unlink()
    sink = TelemetrySink(events_path)
    orch = Orchestrator(telemetry_sink=sink)

    mastered: set[str] = set()
    retention: dict[str, RetentionState] = {}
    clock = 0.0
    seed = 0
    used_first_wrong = False

    while True:
        chosen = select_next(curriculum, mastered, retention, now_ts=clock, k=1)
        if not chosen:
            break
        kc_id = chosen[0]
        kc_name = curriculum.get(kc_id).name
        gen_name = curriculum.get(kc_id).generators[0]
        print(f"\n► Now studying: {kc_name}  (prereqs satisfied)")

        attempts = 0
        while not bkt.mastered(kc_id) and attempts < 8:
            problem = generate(gen_name, difficulty_band=1, seed=seed)
            seed += 1
            verifier = VERIFIER_FOR_DOMAIN[problem.domain]()

            session = Session(
                user_pseudonym=pseudonymize("demo-student", salt="demo-salt"),
                kc_id=kc_id,
                kc_name=kc_name,
                problem_id=problem.id,
                problem_statement=problem.prompt_text,
                target=problem.parsed_target,
                verifier=verifier,
                correct_answer_str=problem.reference_answer,
                bkt_model=bkt,                      # real BKT mastery tracking
            )

            # First problem of the whole demo: answer wrong to show feedback.
            if not used_first_wrong:
                answer = wrong_answer(problem.reference_answer)
                used_first_wrong = True
            else:
                answer = problem.reference_answer

            result = orch.handle_turn(session, answer)   # BKT update happens inside

            print(f"    {problem.prompt_text}")
            print(f"      answered {answer!r} -> {result.verdict.value}"
                  f"  (mastery {bkt.p_mastered(kc_id):.2f})")
            # The answer must NOT leak in coaching prose while the student works:
            leaked = problem.reference_answer in result.coaching_message
            print(f"      tutor: {result.coaching_message.splitlines()[0]}")
            print(f"      [answer leaked into prose? {leaked}]")

            clock += 3600.0
            prev = retention.get(kc_id, RetentionState(last_seen_ts=clock))
            retention[kc_id] = update_after_review(
                prev, now_ts=clock, success=(result.verdict == Verdict.CORRECT)
            )
            attempts += 1

        status = "MASTERED" if bkt.mastered(kc_id) else "advancing (cap reached)"
        print(f"  ✓ {kc_name}: {status}  (mastery {bkt.p_mastered(kc_id):.2f})")
        mastered.add(kc_id)

    print("\nAll knowledge components complete.")
    return events_path


def show_analytics(events_path: Path) -> None:
    banner("6. TELEMETRY — what was logged")
    events = TelemetrySink.read_all(events_path)
    print(f"Persisted {len(events)} telemetry events to {events_path}")

    correct = sum(1 for e in events if e.verdict == Verdict.CORRECT.value)
    wrong = sum(1 for e in events if e.verdict == Verdict.WRONG.value)
    print(f"Verdicts: {correct} correct, {wrong} wrong")

    # Confirm raw identity never hit disk (pseudonymous by construction).
    raw_present = any("demo-student" in e.user_pseudonym for e in events)
    print(f"Raw student id present in logs? {raw_present}  (should be False)")


def main() -> None:
    print("MathTutor — full stack demo")
    show_data_layer()
    events_path = run_session()
    show_analytics(events_path)
    print("\nDone.")


if __name__ == "__main__":
    main()