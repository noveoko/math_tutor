# tests/test_eval.py

"""Tests for eval/telemetry.py and eval/learning_curves.py.

Coverage targets
----------------
T1  emit → read round-trip preserves every field.
T2  append-only: two separate emits produce two lines; prior lines intact.
T3  read_all on non-existent file returns [].
T4  pseudonymize is deterministic per (raw_id, salt).
T5  pseudonymize never stores the raw id in its output.
T6  different salts produce different pseudonyms.
T7  normalized_gain(40, 70) == 0.5.
T8  normalized_gain(100, *) returns nan (ceiling guard).
T9  clean-decline KC is NOT flagged.
T10 flat KC IS flagged.
T11 fit_afm converges on simple synthetic data and returns expected keys.
T12 error_rate_curve returns correct per-opportunity rates.
T13 flag_misspecified_kcs returns sorted list.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
import uuid

import pytest

from mathtutor.contracts import TelemetryEvent
from mathtutor.eval.telemetry import TelemetrySink, pseudonymize
from mathtutor.eval.learning_curves import (
    error_rate_curve,
    fit_afm,
    flag_misspecified_kcs,
    normalized_gain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SALT = "test-salt-abc123"
_ALT_SALT = "different-salt-xyz"


def _event(
    *,
    user: str = "alice",
    kc: str = "kc-linear",
    opp: int = 0,
    verdict: str = "correct",
    session: str = "s1",
) -> TelemetryEvent:
    """Convenience factory: pseudonymizes user before constructing event."""
    return TelemetryEvent(
        event_id=str(uuid.uuid4()),
        session_id=session,
        user_pseudonym=pseudonymize(user, _SALT),
        ts=time.time(),
        kc_id=kc,
        opportunity_index=opp,
        verdict=verdict,
    )


def _make_declining_events(
    kc: str = "kc-declining",
    n_students: int = 20,
    n_opps: int = 8,
    seed: int = 42,
) -> list[TelemetryEvent]:
    """Synthetic events where error rate clearly declines with opportunity.

    At opportunity *n*, a student answers correctly with probability
    ``0.3 + 0.07 * n`` (capped at 0.95), ensuring a strong decline in
    error rate.
    """
    import random
    rng = random.Random(seed)
    events = []
    for s in range(n_students):
        for opp in range(n_opps):
            p_correct = min(0.95, 0.30 + 0.07 * opp)
            verdict = "correct" if rng.random() < p_correct else "wrong"
            events.append(_event(
                user=f"student-{s}", kc=kc, opp=opp, verdict=verdict
            ))
    return events


def _make_flat_events(
    kc: str = "kc-flat",
    n_students: int = 20,
    n_opps: int = 8,
    seed: int = 99,
) -> list[TelemetryEvent]:
    """Synthetic events where error rate stays ~constant (flat KC)."""
    import random
    rng = random.Random(seed)
    events = []
    for s in range(n_students):
        for opp in range(n_opps):
            verdict = "correct" if rng.random() < 0.50 else "wrong"
            events.append(_event(
                user=f"student-{s}", kc=kc, opp=opp, verdict=verdict
            ))
    return events


# ---------------------------------------------------------------------------
# T1-T3: TelemetrySink round-trip and append semantics
# ---------------------------------------------------------------------------

class TestTelemetrySink:

    def test_emit_read_roundtrip(self, tmp_path):
        """T1: every field survives emit → read_all."""
        path = tmp_path / "events.jsonl"
        sink = TelemetrySink(path)
        ev = TelemetryEvent(
            event_id="evt-001",
            session_id="sess-A",
            user_pseudonym=pseudonymize("bob@example.com", _SALT),
            ts=1_700_000_000.0,
            kc_id="algebra-1",
            problem_id="prob-42",
            opportunity_index=3,
            action="answer",
            verdict="wrong",
            hint_level=1,
            latency_ms=1234,
            p_known_before=0.4,
            p_known_after=0.35,
            policy_id="policy-v2",
        )
        sink.emit(ev)
        recovered = TelemetrySink.read_all(path)
        assert len(recovered) == 1
        assert recovered[0] == ev

    def test_append_only_two_emits(self, tmp_path):
        """T2: second emit appends; first line is unchanged."""
        path = tmp_path / "events.jsonl"
        sink = TelemetrySink(path)

        ev1 = _event(user="alice", kc="kc-A", opp=0, verdict="correct")
        ev2 = _event(user="bob",   kc="kc-B", opp=1, verdict="wrong")

        sink.emit(ev1)
        # Read after first emit
        after_first = TelemetrySink.read_all(path)
        assert len(after_first) == 1

        sink.emit(ev2)
        after_second = TelemetrySink.read_all(path)
        assert len(after_second) == 2

        # First event must be byte-for-byte identical after second write
        assert after_second[0] == ev1
        assert after_second[1] == ev2

    def test_read_all_missing_file(self, tmp_path):
        """T3: read_all returns [] when path does not exist."""
        result = TelemetrySink.read_all(tmp_path / "nonexistent.jsonl")
        assert result == []

    def test_parent_dirs_created(self, tmp_path):
        """Sink auto-creates nested directories."""
        path = tmp_path / "deep" / "nested" / "events.jsonl"
        sink = TelemetrySink(path)
        sink.emit(_event())
        assert path.exists()


# ---------------------------------------------------------------------------
# T4-T6: Pseudonymization
# ---------------------------------------------------------------------------

class TestPseudonymize:

    def test_deterministic(self):
        """T4: same inputs always give same pseudonym."""
        p1 = pseudonymize("alice@example.com", _SALT)
        p2 = pseudonymize("alice@example.com", _SALT)
        assert p1 == p2

    def test_raw_id_not_in_output(self):
        """T5: raw user id does not appear in the pseudonym string."""
        raw_id = "verysecretuser@example.com"
        pseudonym = pseudonymize(raw_id, _SALT)
        assert raw_id not in pseudonym
        # Also check the pseudonym doesn't contain any fragment of the email
        assert "@" not in pseudonym
        assert "verysecretuser" not in pseudonym

    def test_different_salts_differ(self):
        """T6: changing the salt produces a different pseudonym."""
        p1 = pseudonymize("alice@example.com", _SALT)
        p2 = pseudonymize("alice@example.com", _ALT_SALT)
        assert p1 != p2

    def test_different_users_differ(self):
        """Different users under the same salt produce different pseudonyms."""
        p1 = pseudonymize("alice@example.com", _SALT)
        p2 = pseudonymize("bob@example.com", _SALT)
        assert p1 != p2

    def test_pseudonym_is_hex_string(self):
        """Pseudonym is a 64-char lowercase hex digest (SHA-256 output)."""
        p = pseudonymize("user", _SALT)
        assert len(p) == 64
        assert all(c in "0123456789abcdef" for c in p)

    def test_event_pseudonym_not_raw_id(self, tmp_path):
        """T5 (via sink): raw id never written to disk."""
        raw_id = "private-user-12345"
        path = tmp_path / "events.jsonl"
        sink = TelemetrySink(path)
        ev = TelemetryEvent(
            event_id="e1", session_id="s1",
            user_pseudonym=pseudonymize(raw_id, _SALT),
            ts=time.time(),
        )
        sink.emit(ev)
        content = path.read_text(encoding="utf-8")
        assert raw_id not in content


# ---------------------------------------------------------------------------
# T7-T8: normalized_gain
# ---------------------------------------------------------------------------

class TestNormalizedGain:

    def test_known_value(self):
        """T7: normalized_gain(40, 70) == 0.5 (Hake example)."""
        assert normalized_gain(40, 70) == pytest.approx(0.5)

    def test_zero_gain(self):
        """Pre == post => gain == 0."""
        assert normalized_gain(50, 50) == pytest.approx(0.0)

    def test_full_gain(self):
        """Post == 100 => gain == 1.0."""
        assert normalized_gain(0, 100) == pytest.approx(1.0)

    def test_ceiling_pre_100(self):
        """T8: pre == 100 returns nan (ceiling guard)."""
        result = normalized_gain(100, 100)
        assert math.isnan(result)

    def test_negative_gain(self):
        """Post < pre => negative gain (regression)."""
        g = normalized_gain(60, 40)
        assert g < 0


# ---------------------------------------------------------------------------
# T9-T10: flag_misspecified_kcs
# ---------------------------------------------------------------------------

class TestFlagMisspecifiedKCs:

    def test_clean_decline_not_flagged(self):
        """T9: a KC with monotone-declining error rate is NOT flagged."""
        events = _make_declining_events("kc-good")
        flagged = flag_misspecified_kcs(events)
        assert "kc-good" not in flagged

    def test_flat_curve_flagged(self):
        """T10: a KC with flat error rate IS flagged."""
        events = _make_flat_events("kc-bad")
        flagged = flag_misspecified_kcs(events)
        assert "kc-bad" in flagged

    def test_mixed_flags(self):
        """Good KC not flagged; bad KC flagged when both are present."""
        events = (
            _make_declining_events("kc-good", seed=7)
            + _make_flat_events("kc-bad", seed=0)  # seed=0: mean Δ > 0 reliably
        )
        flagged = flag_misspecified_kcs(events)
        assert "kc-bad" in flagged
        assert "kc-good" not in flagged

    def test_result_is_sorted(self):
        """T13: flagged list is lexicographically sorted."""
        events = _make_flat_events("zz-kc") + _make_flat_events("aa-kc")
        flagged = flag_misspecified_kcs(events)
        assert flagged == sorted(flagged)

    def test_no_events_returns_empty(self):
        """No events => no KCs to flag."""
        assert flag_misspecified_kcs([]) == []

    def test_single_opportunity_flagged(self):
        """KC with only 1 opportunity index can't be assessed => flagged."""
        events = [_event(kc="kc-one-opp", opp=0, verdict="correct")]
        flagged = flag_misspecified_kcs(events)
        assert "kc-one-opp" in flagged


# ---------------------------------------------------------------------------
# T11: fit_afm
# ---------------------------------------------------------------------------

class TestFitAFM:

    def test_returns_expected_keys(self):
        """T11a: result dict has all expected keys."""
        events = _make_declining_events("kc-fit", n_students=10)
        result = fit_afm(events)
        for key in ("kc_easiness", "kc_learning_rate", "student_ability",
                    "converged", "n_obs"):
            assert key in result, f"missing key: {key}"

    def test_converged_on_clean_data(self):
        """T11b: optimiser converges on well-behaved synthetic data."""
        events = _make_declining_events("kc-fit", n_students=30, n_opps=10)
        result = fit_afm(events)
        assert result["converged"] is True

    def test_positive_learning_rate(self):
        """T11c: declining-error KC should get a positive learning rate."""
        events = _make_declining_events("kc-lr", n_students=40, n_opps=10)
        result = fit_afm(events)
        lr = result["kc_learning_rate"]["kc-lr"]
        # The data was constructed so practice helps; γ should be positive
        assert lr > 0, f"Expected γ > 0, got {lr}"

    def test_n_obs_correct(self):
        """T11d: n_obs equals number of scored events (correct+wrong)."""
        events = _make_declining_events("kc-n", n_students=5, n_opps=4)
        result = fit_afm(events)
        # All events have verdict correct/wrong
        assert result["n_obs"] == len(events)

    def test_empty_events_not_converged(self):
        """T11e: empty input returns converged=False, empty dicts."""
        result = fit_afm([])
        assert result["converged"] is False
        assert result["kc_easiness"] == {}

    def test_ignores_events_without_kc(self):
        """Events with kc_id=None are silently ignored."""
        ev_no_kc = TelemetryEvent(
            event_id="x", session_id="s", user_pseudonym="p",
            ts=1.0, kc_id=None, verdict="correct"
        )
        events = _make_declining_events("kc-ok", n_students=5) + [ev_no_kc]
        result = fit_afm(events)
        assert None not in result["kc_easiness"]


# ---------------------------------------------------------------------------
# T12: error_rate_curve
# ---------------------------------------------------------------------------

class TestErrorRateCurve:

    def test_correct_rates(self):
        """T12: error rate at opp 0 is 0.0 when all correct; 1.0 when all wrong."""
        events = [
            _event(kc="kc-x", opp=0, verdict="correct"),
            _event(kc="kc-x", opp=0, verdict="correct"),
            _event(kc="kc-x", opp=1, verdict="wrong"),
            _event(kc="kc-x", opp=1, verdict="wrong"),
        ]
        curve = error_rate_curve(events, "kc-x")
        assert len(curve) == 2
        assert curve[0] == pytest.approx(0.0)   # all correct at opp 0
        assert curve[1] == pytest.approx(1.0)   # all wrong at opp 1

    def test_empty_for_unknown_kc(self):
        """No events for requested KC => empty list."""
        events = [_event(kc="kc-other", opp=0, verdict="correct")]
        assert error_rate_curve(events, "kc-unknown") == []

    def test_nan_for_gap(self):
        """Gap in opportunity index produces nan at that position."""
        events = [
            _event(kc="kc-gap", opp=0, verdict="correct"),
            # opp=1 missing
            _event(kc="kc-gap", opp=2, verdict="wrong"),
        ]
        curve = error_rate_curve(events, "kc-gap")
        assert len(curve) == 3
        assert math.isnan(curve[1])

    def test_mixed_verdict(self):
        """50% error rate at a given opportunity."""
        events = [
            _event(kc="kc-mix", opp=0, verdict="correct"),
            _event(kc="kc-mix", opp=0, verdict="wrong"),
        ]
        curve = error_rate_curve(events, "kc-mix")
        assert curve[0] == pytest.approx(0.5)