# MathTutor

A verified, adaptive math tutoring engine. It generates practice problems,
judges student answers with a computer algebra system (CAS), tracks what each
learner has mastered, schedules review using a forgetting model, diagnoses
specific misconceptions, and wraps everything in coaching language — without
ever letting the language layer decide whether a math answer is right.

The guiding principle: **the CAS is the single source of mathematical truth.**
The language model (or its offline template fallback) only phrases feedback. It
never adjudicates correctness, and every concrete claim it makes is
re-verified before it reaches a student.

---

## Core idea in one breath

A "turn" is one student submission. It flows through nine deterministic steps:

```
parse → verify (CAS) → update mastery (BKT) → diagnose misconception
      → choose support level → ask tutor for prose → run safety filters
      → emit telemetry → return result
```

Steps that decide truth or progress (verify, mastery, gate checks) touch no
model. Only step 6 — turning verified facts into a sentence — uses the tutor,
and if it fails, an offline template tutor takes over transparently.

---

## Project layout

```
mathtutor/
├── pyproject.toml
├── conftest.py
├── src/mathtutor/
│   ├── orchestrator.py            # the spine: handle_turn()
│   ├── main.py                    # runnable demo (python -m mathtutor.main)
│   ├── contracts.py               # shared dataclasses / protocols (no deps)
│   ├── cas/                       # parsing + equivalence (SymPy)
│   ├── domain/
│   │   ├── curriculum.py          # prerequisite DAG + JSON loaders
│   │   ├── generators.py          # problem generators (self-verifying)
│   │   └── verifiers/             # per-domain CAS verifiers
│   ├── learner/
│   │   ├── bkt.py                 # Bayesian Knowledge Tracing
│   │   └── scheduling.py          # forgetting model + spaced repetition
│   ├── tutoring/
│   │   ├── scaffolding.py         # support levels, hint ladder, gate
│   │   └── misconceptions.py      # buggy-rule diagnosis
│   ├── safety/
│   │   ├── claim_cert.py          # re-verify LLM claims
│   │   └── leak_filter.py         # redact answers while gated
│   ├── llm/offline.py             # deterministic template tutor
│   ├── eval/                      # telemetry sink + learning-curve analytics
│   └── data/curricula/            # author your subjects here (JSON)
└── tests/
```

---

## 1. Setup

**Prerequisites:** Python 3.12. The engine depends on SymPy (the CAS), NumPy and
SciPy (learning-curve fitting), and pytest (tests).

```bash
# from the project root
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# install the package plus dev tools (pytest) in editable mode
pip install -e ".[dev]"

# to install only the runtime engine (no test tooling):
#   pip install -e .
```

Confirm everything works by running the test suite:

```bash
pytest -q
```

You should see all tests pass. If you instead see
`No files were found in testpaths`, see [Troubleshooting](#troubleshooting).

---

## 2. Quick demo

The fastest way to see a turn end-to-end:

```bash
python -m mathtutor.main
```

This builds a quadratic problem (`x² − 5x + 6 = 0`, answer `{2, 3}`), submits a
wrong answer then the right one, and prints the verdict, support level, updated
mastery, and the tutor's message for each. No LLM key needed — it uses the
offline template tutor.

---

## 3. How it works (including the math)

You don't need this section to use the engine, but it explains the two
quantitative models you'll see in telemetry and scheduling.

### Mastery: Bayesian Knowledge Tracing (BKT)

Each knowledge component (KC) has a probability `p = P(mastered)` per student,
updated after every answer using four parameters: prior `L0`, learn-rate `T`,
slip `S` (wrong despite mastery), and guess `G` (right without mastery).
Defaults: `L0=0.20, T=0.30, S=0.10, G=0.20`.

The update has two steps. First, **condition on the observation** with Bayes'
rule. For a *correct* answer:

```
p_post = p·(1−S) / ( p·(1−S) + (1−p)·G )
```

Then apply a **learning transition** (the student may have just learned it):

```
p_next = p_post + (1 − p_post)·T
```

Worked example, one correct answer from the prior `p = 0.20`:

```
p_post = 0.20·0.90 / (0.20·0.90 + 0.80·0.20)
       = 0.18 / 0.34  ≈ 0.529
p_next = 0.529 + (1 − 0.529)·0.30  ≈ 0.671
```

So a single correct answer moves mastery from 0.20 to ≈0.671 — but **not** past
the 0.95 mastery threshold, because guessing is modelled. It takes roughly three
consecutive correct answers to declare mastery, which is the point: one lucky
guess shouldn't graduate a student.

### Forgetting: half-life decay

Between sessions, predicted recall of a KC decays exponentially:

```
recall(t) = 2^(−elapsed / half_life)
```

At `elapsed = half_life` recall is exactly `2^(−1) = 0.5`; at half that interval
it's `2^(−0.5) ≈ 0.707`. A **spaced** correct review (one that waited long
enough) multiplies the half-life by 2, so memories you successfully retrieve
after a gap last longer — while cramming (re-answering immediately) earns no
bonus. The scheduler flags a KC for review once its recall drops to ≈0.85, the
"desirable difficulty" sweet spot.

### Why these matter

`select_next` uses mastery (to gate locked KCs) and recall (to schedule review),
interleaving new learning with due reviews so no single skill dominates a
session. The `eval` module can fit an Additive Factor Model across many students
to flag KCs whose error rate *doesn't* decline with practice — a sign the KC
conflates sub-skills and should be split.

---

## 4. Creating learning content

There are three ways to create content, from highest-level to lowest.

### 4a. Author a curriculum (data, any subject)

A curriculum is a prerequisite graph of knowledge components. You author it as
JSON — no Python required. Put files under `src/mathtutor/data/curricula/`.

Minimal schema (only `id` is required per KC; everything else has defaults):

```json
{
  "subject": "calculus_intro",
  "knowledge_components": [
    { "id": "limits_intro", "name": "Introduction to Limits",
      "prerequisites": [], "verifier_domain": "limits",
      "difficulty_band": 1, "generators": ["gen_limits_intro"] },

    { "id": "continuity", "prerequisites": ["limits_intro"] },

    { "id": "derivative_definition", "prerequisites": ["limits_intro"] }
  ]
}
```

KCs may be listed in **any order** — the loader sorts them so prerequisites come
first, and raises a clear `CurriculumError` on a cycle, an unknown prerequisite,
or a duplicate id. Load a bundled curriculum by name:

```python
from mathtutor.domain.curriculum import load_curriculum, list_curricula

list_curricula()                    # e.g. ['calculus_intro'] — what's bundled
calc = load_curriculum("calculus_intro")   # name, with or without ".json"

print(calc.subject)                 # "calculus_intro"
print(calc.topological_order())     # a valid teaching order
print(calc.ready_kcs(set()))        # KCs a fresh student can start with
```

`load_curriculum` reads through `importlib.resources`, so it works identically
whether the package is run from a source checkout or installed as a wheel —
unlike a hard-coded `src/...` path. For a curriculum stored **outside** the
package (e.g. a file a teacher dropped somewhere), read it directly instead:

```python
from mathtutor.domain.curriculum import Curriculum
calc = Curriculum.from_json_file("/path/to/my_curriculum.json")
```

You can also build from a dict or a JSON string (`Curriculum.from_dict(...)`,
`Curriculum.from_json(...)`) and serialize back out (`.to_dict()`, `.to_json()`).

> **Note on `generators`.** The names here are claims about which generator
> produces problems for a KC. A KC may name a generator you haven't built yet —
> that's an intentional roadmap. But a generator must actually be *registered*
> (see 4b) before the engine can produce problems for that KC.

### 4b. Add a problem generator

A generator is a function that returns a `Problem`. It must be registered with
`@register("name")` and it **self-verifies**: before returning, it asks the CAS
to confirm its own reference answer is correct (and that the problem's
difficulty is in band). If self-verification fails, it raises `GeneratorError` —
so a broken generator can never ship a bad problem.

```python
import random, uuid
from mathtutor.domain.generators import register, Problem, _assert_self_verified
from mathtutor.domain.verifiers.linear_equation import EquationVerifier
from mathtutor.contracts import Target

@register("one_step_multiply")
def gen_one_step_multiply(*, difficulty_band: int, seed: int) -> Problem:
    rng = random.Random(seed)
    x = rng.randint(-9, 9)
    a = rng.choice([i for i in range(-9, 10) if i != 0])
    b = a * x                                  # so a·x = b has integer solution x

    prompt = f"Solve for x: {a}x = {b}"
    reference = str(x)
    target = Target(domain="linear_equation", payload={"answer": reference})

    problem = Problem(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"osm-{seed}-{difficulty_band}")),
        kc_id="linear_one_step",
        domain="linear_equation",
        prompt_text=prompt,
        parsed_target=target,
        reference_answer=reference,
        difficulty_band=difficulty_band,
        meta={"a": a, "x": x},
    )

    # solve-step proxy must fall in the band's range (band1: 1–3, band2: 3–6, band3: 6–10)
    solve_steps = {1: 2, 2: 4, 3: 7}[difficulty_band]
    _assert_self_verified(problem, EquationVerifier(), solve_steps)
    return problem
```

Use it (importing the module is what runs the `@register`):

```python
from mathtutor.domain.generators import generate
p = generate("one_step_multiply", difficulty_band=1, seed=0)
print(p.prompt_text, "->", p.reference_answer)
```

Three generators ship today: `linear_equation`, `quadratic_equation`,
`fraction_addition`. Their `domain` field also tells you which verifier judges
them.

### 4c. A one-off problem by hand

For a quick experiment you can skip generators entirely — pair a verifier with a
`Target`:

```python
from mathtutor.contracts import Target
from mathtutor.domain.verifiers.polynomial import PolynomialVerifier

verifier = PolynomialVerifier()
target = Target(domain="polynomial", payload={"answer": "{2, 3}"})

# the verifier parses, then judges
student = verifier.parse("{2, 3}")
print(verifier.accepts(student, target).correct)   # True
```

The available verifier domains: `EquationVerifier` (linear/equation),
`PolynomialVerifier` (polynomials & quadratics), `FractionVerifier` (reduced
rationals), `InequalityVerifier`, `SystemVerifier`.

---

## 5. Run a full adaptive session

This ties it all together: a curriculum drives KC selection, BKT tracks mastery,
the scheduler interleaves review, the orchestrator judges and coaches each
answer. Save it as `run_session.py` at the project root and run
`python run_session.py`.

```python
"""An interactive adaptive practice session over a small curriculum."""
from __future__ import annotations

from mathtutor.domain.curriculum import Curriculum
from mathtutor.domain.generators import generate
from mathtutor.learner.bkt import BKTLearnerState
from mathtutor.learner.scheduling import RetentionState, update_after_review, select_next
from mathtutor.orchestrator import Orchestrator, Session
from mathtutor.contracts import Verdict

# Map a problem's domain to the verifier that judges it.
from mathtutor.domain.verifiers.linear_equation import EquationVerifier
from mathtutor.domain.verifiers.polynomial import PolynomialVerifier
from mathtutor.domain.verifiers.fraction import FractionVerifier

VERIFIER_FOR_DOMAIN = {
    "linear_equation": EquationVerifier,
    "quadratic_equation": PolynomialVerifier,
    "fraction_addition": FractionVerifier,
}

# A tiny curriculum whose KCs map onto the three registered generators.
PRACTICE = {
    "subject": "practice",
    "knowledge_components": [
        {"id": "fractions",  "generators": ["fraction_addition"]},
        {"id": "linear",     "prerequisites": ["fractions"], "generators": ["linear_equation"]},
        {"id": "quadratics", "prerequisites": ["linear"],    "generators": ["quadratic_equation"]},
    ],
}

def main() -> None:
    curriculum = Curriculum.from_dict(PRACTICE)
    bkt = BKTLearnerState()
    orch = Orchestrator()                      # offline tutor; no API key needed
    mastered: set[str] = set()
    retention: dict[str, RetentionState] = {}
    clock = 0.0
    seed = 0

    print("Adaptive session — type an answer, or 'q' to quit.\n")

    while True:
        chosen = select_next(curriculum, mastered, retention, now_ts=clock, k=1)
        if not chosen:
            print("\nNothing left to practice — curriculum complete!")
            break
        kc_id = chosen[0]

        gen_name = curriculum.get(kc_id).generators[0]
        problem = generate(gen_name, difficulty_band=1, seed=seed)
        seed += 1

        verifier = VERIFIER_FOR_DOMAIN[problem.domain]()
        session = Session(
            user_pseudonym="demo",
            kc_id=kc_id,
            kc_name=kc_id,
            problem_id=problem.id,
            problem_statement=problem.prompt_text,
            target=problem.parsed_target,
            verifier=verifier,
            correct_answer_str=problem.reference_answer,
            bkt_model=bkt,                     # real mastery tracking
        )

        print(f"[{kc_id}]  {problem.prompt_text}")
        answer = input("  your answer: ").strip()
        if answer.lower() in {"q", "quit", ""}:
            break

        result = orch.handle_turn(session, answer)   # BKT update happens inside

        print(f"  -> {result.verdict.value}  (mastery now {bkt.p_mastered(kc_id):.2f})")
        if result.misconception_id:
            print(f"  -> misconception: {result.misconception_id}")
        print(f"  tutor: {result.coaching_message}\n")

        # advance the clock and record this review for spaced repetition
        clock += 3600.0
        prev = retention.get(kc_id, RetentionState(last_seen_ts=clock))
        retention[kc_id] = update_after_review(
            prev, now_ts=clock, success=(result.verdict == Verdict.CORRECT)
        )

        if bkt.mastered(kc_id):
            mastered.add(kc_id)
            print(f"  *** mastered {kc_id}! moving on ***\n")

if __name__ == "__main__":
    main()
```

Answer correctly a few times in a row and you'll watch mastery climb past 0.95,
the KC graduate, and the scheduler unlock the next one in the prerequisite
chain (`fractions → linear → quadratics`).

---

## 6. Telemetry & analytics

Every turn emits a `TelemetryEvent`. For durable, privacy-safe logging, give the
orchestrator a `TelemetrySink` and pseudonymize student ids before they touch
disk:

```python
from mathtutor.eval.telemetry import TelemetrySink, pseudonymize
from mathtutor.orchestrator import Orchestrator, Session

sink = TelemetrySink("events.jsonl")
orch = Orchestrator(telemetry_sink=sink)

session = Session(user_pseudonym=pseudonymize("alice@example.com", salt="keep-this-secret"))
# ... run turns; each appends one JSON line to events.jsonl
```

Read events back and analyze learning:

```python
from mathtutor.eval.telemetry import TelemetrySink
from mathtutor.eval.learning_curves import (
    fit_afm, error_rate_curve, flag_misspecified_kcs, normalized_gain
)

events = TelemetrySink.read_all("events.jsonl")

flag_misspecified_kcs(events)        # KCs whose error rate doesn't decline -> split them
error_rate_curve(events, "linear")   # per-opportunity error rate
fit_afm(events)                      # KC easiness + learning rates across students
normalized_gain(pre=40, post=70)     # Hake's gain == 0.5
```

The sink is append-only and pseudonymous by construction: raw ids are HMAC-hashed
with your secret salt and never written.

---

## 7. Extending the engine

- **New verifier:** implement the `Verifier` protocol (`parse`, `canonical`,
  `accepts`) in `domain/verifiers/`. `accepts` returns a `Judgment`; correctness
  flows from CAS equivalence, never string comparison.
- **New misconception:** add a buggy-rule class to `tutoring/misconceptions.py`
  with `id`, `applies_to`, and `transform`. `diagnose(previous_line,
  student_line)` matches the student's *transformation between two lines*
  against these rules — so misconception detection needs the prior line, which
  the orchestrator threads through `Session.previous_artifact`.
- **Real LLM tutor:** pass any object with a `coach(context) -> str` method as
  `Orchestrator(llm=...)`. If it raises, the offline tutor covers for it; its
  output is still run through claim certification and answer redaction.

---

## 8. Testing

```bash
pytest -q                                  # everything
pytest tests/test_orchestrator.py -v       # one module, verbose
pytest -k curriculum                        # by keyword
```

When you add a generator, the suite's subset guard
(`test_registered_generators_are_referenced_by_curriculum`) checks that every
*registered* generator is referenced by at least one KC. The reverse — a KC
naming a not-yet-built generator — is allowed on purpose.

---

## Troubleshooting

**`PytestConfigWarning: No files were found in testpaths`** — your
`pyproject.toml` has a `testpaths` setting pointing somewhere pytest can't find,
so it falls back to recursive discovery (tests still run). Set it to your actual
tests directory, e.g.:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

**`KeyError: Unknown generator: ...`** — you called `generate("name", ...)` for a
generator whose module hasn't been imported (so its `@register` never ran), or
the name is misspelled. Import the module that defines it first.

**`GeneratorError: Verifier rejected generated answer` / `out of band`** — a
generator's reference answer failed CAS self-verification, or its solve-step
proxy fell outside the difficulty band's range (band1: 1–3, band2: 3–6,
band3: 6–10). Fix the answer or the `solve_steps` value.

**A wrong answer raised the student's mastery.** That's expected with the real
BKT model attached: the learning-transition term fires regardless of
correctness, so a single wrong answer can nudge `p` up slightly even as the
Bayesian step pulls it down. Mastery only crosses the 0.95 threshold after
sustained correct performance.