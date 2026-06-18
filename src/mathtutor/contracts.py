# mathtutor/contracts.py

"""Shared contracts and core types for Verified Math Tutor.

This module intentionally has no dependencies on other project modules.
It defines dataclasses, enums, protocols, and serialization helpers used
throughout the system.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol
import json


class ParseError(Exception):
    """Raised when raw student input cannot be parsed into a math artifact."""


@dataclass(slots=True)
class Artifact:
    """Parsed mathematical artifact."""

    kind: str  # equation|expression|inequality|system|value|set
    expr: object
    raw: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Target:
    """Describes the constraints for a correct answer."""

    domain: str
    payload: dict[str, Any]
    form: str | None = None
    complete_count: int | None = None


@dataclass(slots=True)
class Canonical:
    """Canonical comparable representation."""

    key: object
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Judgment:
    parsed_ok: bool
    value_equivalent: bool
    form_ok: bool
    correct: bool
    partial: bool
    decidable: bool
    confidence: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KnowledgeComponent:
    """A knowledge component in the curriculum DAG."""

    id: str
    name: str
    prerequisites: list[str]
    verifier_domain: str
    difficulty_band: int
    generators: list[str]


@dataclass(slots=True)
class BuggyRule:
    """Represents a common misconception transformation."""

    id: str
    description: str
    applies_to: Callable[[object], bool]
    transform: Callable[[object], object]
    remediation: str


class Verifier(Protocol):
    """Protocol implemented by domain verifiers."""

    domain: str

    def parse(self, raw: str) -> Artifact:
        ...

    def canonical(self, a: Artifact) -> Canonical:
        ...

    def accepts(self, student: Artifact, target: Target) -> Judgment:
        ...


class SupportLevel(Enum):
    WORKED = "worked"
    COMPLETION = "completion"
    INDEPENDENT = "independent"


class Verdict(Enum):
    CORRECT = "correct"
    PARTIAL = "partial"
    WRONG = "wrong"
    ABSTAIN = "abstain"


def verdict_from_judgment(j: Judgment) -> Verdict:
    """Map a Judgment to a coarse verdict."""
    if not j.parsed_ok or j.confidence < 0.5:
        return Verdict.ABSTAIN
    if j.correct:
        return Verdict.CORRECT
    if j.partial:
        return Verdict.PARTIAL
    return Verdict.WRONG


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """Immutable telemetry event."""

    event_id: str
    session_id: str
    user_pseudonym: str
    ts: float

    kc_id: str | None = None
    problem_id: str | None = None
    opportunity_index: int = 0
    action: str = ""

    input_artifact: str | None = None
    verdict: str | None = None
    error_kind: str | None = None
    misconception_id: str | None = None

    support_level: str | None = None
    hint_level: int = 0
    latency_ms: int = 0

    p_known_before: float | None = None
    p_known_after: float | None = None

    policy_id: str | None = None
    affect_signal: str | None = None

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, data: str) -> "TelemetryEvent":
        """Deserialize from JSON."""
        return cls(**json.loads(data))
