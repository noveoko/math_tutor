# tests/test_contracts.py

from mathtutor.contracts import (
    Artifact,
    BuggyRule,
    Canonical,
    Judgment,
    KnowledgeComponent,
    Target,
    TelemetryEvent,
    Verdict,
    verdict_from_judgment,
)


def test_construct_dataclasses():
    artifact = Artifact("expression", 42, "42", {})
    target = Target("algebra", {"answer": 42}, "reduced", 1)
    canonical = Canonical(42, {})
    kc = KnowledgeComponent(
        id="kc1",
        name="Linear Equations",
        prerequisites=[],
        verifier_domain="algebra",
        difficulty_band=1,
        generators=["gen_linear"],
    )
    rule = BuggyRule(
        id="bug1",
        description="Sign error",
        applies_to=lambda x: True,
        transform=lambda x: x,
        remediation="Check signs carefully",
    )

    assert artifact.kind == "expression"
    assert target.domain == "algebra"
    assert canonical.key == 42
    assert kc.name == "Linear Equations"
    assert rule.id == "bug1"


def test_verdict_mapping():
    correct = Judgment(True, True, True, True, False, True, 1.0, {})
    partial = Judgment(True, False, False, False, True, True, 1.0, {})
    wrong = Judgment(True, False, False, False, False, True, 1.0, {})
    abstain1 = Judgment(False, False, False, False, False, False, 0.0, {})
    abstain2 = Judgment(True, False, False, False, False, False, 0.4, {})

    assert verdict_from_judgment(correct) == Verdict.CORRECT
    assert verdict_from_judgment(partial) == Verdict.PARTIAL
    assert verdict_from_judgment(wrong) == Verdict.WRONG
    assert verdict_from_judgment(abstain1) == Verdict.ABSTAIN
    assert verdict_from_judgment(abstain2) == Verdict.ABSTAIN


def test_telemetry_roundtrip():
    event = TelemetryEvent(
        event_id="e1",
        session_id="s1",
        user_pseudonym="user",
        ts=123456.0,
        kc_id="kc1",
        action="submit",
        verdict="correct",
    )

    json_blob = event.to_json()
    restored = TelemetryEvent.from_json(json_blob)

    assert restored == event
