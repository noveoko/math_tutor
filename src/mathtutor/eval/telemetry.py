# mathtutor/eval/telemetry.py

"""Append-only telemetry sink and pseudonymization helpers.

Design principles
-----------------
* **Append-only**: ``TelemetrySink.emit`` only ever appends a JSON line.
  No line is ever modified or deleted by this module.
* **Pseudonymous by construction**: raw user IDs never touch the file.
  ``pseudonymize(raw_id, salt)`` converts them to an HMAC-SHA-256 hex
  digest before the event is written.  The salt is a caller-supplied
  secret (e.g. an environment variable); this module never stores it.
* **No PII escaping**: if a caller somehow embeds raw IDs elsewhere in
  the event fields that is their bug; this module only guarantees
  ``user_pseudonym`` is hashed.

File format: one UTF-8 JSON object per line (JSON Lines / NDJSON),
terminated by ``\\n``.  Reading is purely sequential; no index is built.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Iterable

from mathtutor.contracts import TelemetryEvent


# ---------------------------------------------------------------------------
# Pseudonymization
# ---------------------------------------------------------------------------

def pseudonymize(raw_id: str, salt: str) -> str:
    """Return a pseudonym for *raw_id* using HMAC-SHA-256 with *salt*.

    Properties
    ----------
    * **Deterministic**: same ``(raw_id, salt)`` always yields the same hex
      digest, so events from the same user are linkable within a study but
      the raw identity is not recoverable without the salt.
    * **One-way** (under HMAC security): without the salt an adversary
      cannot reverse the pseudonym to the original ID.
    * **Collision-resistant**: 256-bit output; birthday-bound is 2^128.

    Parameters
    ----------
    raw_id:
        The raw user identifier (e.g. an email address or database UUID).
        This value is **never** written to disk by this module.
    salt:
        A secret string known only to the system operator.  Treat it like
        a password: load from an environment variable, not source code.

    Returns
    -------
    str
        64-character lowercase hex digest.
    """
    return hmac.new(
        salt.encode(),
        raw_id.encode(),
        digestmod=hashlib.sha256,
    ).hexdigest()


# ---------------------------------------------------------------------------
# Sink
# ---------------------------------------------------------------------------

class TelemetrySink:
    """Append-only writer for :class:`~mathtutor.contracts.TelemetryEvent`.

    Parameters
    ----------
    path:
        Path to the JSON-Lines log file.  Created (including parent dirs)
        if it does not exist; opened in append mode, never truncated.

    Example
    -------
    >>> import tempfile, os, time
    >>> from mathtutor.contracts import TelemetryEvent
    >>> from mathtutor.eval.telemetry import TelemetrySink, pseudonymize
    >>>
    >>> salt = "dev-only-salt"
    >>> with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as f:
    ...     path = f.name
    >>> sink = TelemetrySink(path)
    >>> event = TelemetryEvent(
    ...     event_id="e1", session_id="s1",
    ...     user_pseudonym=pseudonymize("alice@example.com", salt),
    ...     ts=time.time(), kc_id="linear-equations", opportunity_index=0,
    ...     verdict="correct",
    ... )
    >>> sink.emit(event)
    >>> os.unlink(path)
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def emit(self, event: TelemetryEvent) -> None:
        """Append *event* as a single JSON line to the log file.

        The file is opened, written, and closed on every call so that:

        * Multiple processes can safely append concurrently (O_APPEND is
          atomic for writes ≤ PIPE_BUF on POSIX; JSON Lines are short).
        * A crash between emits never corrupts previously written lines.

        Parameters
        ----------
        event:
            A fully-constructed, pseudonymized :class:`TelemetryEvent`.
            The caller is responsible for setting ``user_pseudonym`` via
            :func:`pseudonymize`; this method does not inspect PII.
        """
        line = event.to_json() + "\n"
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @staticmethod
    def read_all(path: str | os.PathLike[str]) -> list[TelemetryEvent]:
        """Deserialize every event from a JSON-Lines log file.

        Parameters
        ----------
        path:
            Path to a file previously written by :meth:`emit`.

        Returns
        -------
        list[TelemetryEvent]
            Events in the order they were appended (chronological if
            callers emit in order).  Returns ``[]`` if the file is empty
            or does not exist.

        Raises
        ------
        json.JSONDecodeError
            If a line is not valid JSON (indicates file corruption).
        TypeError / KeyError
            If a line's JSON does not match :class:`TelemetryEvent` fields.
        """
        p = Path(path)
        if not p.exists():
            return []
        events: list[TelemetryEvent] = []
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:  # skip blank lines that can appear at EOF
                    events.append(TelemetryEvent.from_json(line))
        return events