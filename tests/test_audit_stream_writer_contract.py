# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — audit stream writer/reader contract tests

"""The audit stream writer must only emit envelopes its reader accepts.

One envelope the reader refuses makes the whole stream unreadable, including
every event before it, and blocks reopening the stream for append. A refused
write must also leave the writer's sequence untouched, or the next valid event
opens a gap that fails integrity verification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_phase_orchestrator.exceptions import AuditError
from scpn_phase_orchestrator.runtime.audit_logger import AuditLogger
from scpn_phase_orchestrator.runtime.audit_stream import (
    EventStreamWriter,
    read_event_stream,
    verify_event_stream_integrity,
)

_BAD_EVENTS = [
    pytest.param({"event": "x" * 129}, None, id="derived-type-too-long"),
    pytest.param({"event": "bad\ntype"}, None, id="derived-type-control-char"),
    pytest.param({"note": 1}, "x" * 129, id="explicit-type-too-long"),
    pytest.param({"note": 1}, "bad\ttype", id="explicit-type-control-char"),
    pytest.param({"note": 1}, "", id="explicit-type-empty"),
    pytest.param([1, 2], None, id="payload-not-mapping"),
    pytest.param({"value": float("nan")}, None, id="payload-non-finite"),
]


@pytest.fixture(autouse=True)
def _unsigned_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPO_AUDIT_KEY", raising=False)
    monkeypatch.delenv("SPO_AUDIT_KEYRING", raising=False)


@pytest.mark.parametrize(("payload", "event_type"), _BAD_EVENTS)
def test_refused_write_keeps_stream_readable_and_chain_intact(
    tmp_path: Path, payload: object, event_type: str | None
) -> None:
    path = tmp_path / "audit.spoa"
    writer = EventStreamWriter(path)
    writer.write({"step": 1})
    with pytest.raises(ValueError):
        writer.write(payload, event_type=event_type)  # type: ignore[arg-type]
    writer.write({"step": 2})
    writer.close()

    events = read_event_stream(path)
    assert [event.sequence for event in events] == [1, 2]
    assert verify_event_stream_integrity(events) == (True, 2)

    reopened = EventStreamWriter(path)
    reopened.write({"step": 3})
    reopened.close()
    assert verify_event_stream_integrity(read_event_stream(path)) == (True, 3)


@pytest.mark.parametrize(("payload", "event_type"), _BAD_EVENTS)
def test_resolve_event_type_refuses_what_write_refuses(
    tmp_path: Path, payload: object, event_type: str | None
) -> None:
    writer = EventStreamWriter(tmp_path / "audit.spoa")
    try:
        with pytest.raises(ValueError):
            writer.resolve_event_type(payload, event_type=event_type)  # type: ignore[arg-type]
    finally:
        writer.close()


def test_resolve_event_type_matches_recorded_type(tmp_path: Path) -> None:
    path = tmp_path / "audit.spoa"
    writer = EventStreamWriter(path)
    cases = [
        ({"header": True}, None),
        ({"event": "regime_change"}, None),
        ({"step": 4}, None),
        ({"other": 1}, None),
        ({"other": 1}, "operator_note"),
        ({"event": "y" * 128}, None),
    ]
    expected = [writer.resolve_event_type(p, event_type=t) for p, t in cases]
    for payload, event_type in cases:
        writer.write(payload, event_type=event_type)
    writer.close()
    recorded = [event.event_type for event in read_event_stream(path)]
    assert recorded == expected
    assert expected[:5] == [
        "header",
        "regime_change",
        "step",
        "record",
        "operator_note",
    ]


def test_logger_refuses_unsealable_event_before_either_sink(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "audit.jsonl"
    stream_path = tmp_path / "audit.spoa"
    with AuditLogger(jsonl_path, event_stream=stream_path) as logger:
        logger.log_event("first", {"n": 1})
        with pytest.raises(AuditError, match="event stream"):
            logger.log_event("e" * 200, {"n": 2})
        logger.log_event("second", {"n": 3})

    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    events = read_event_stream(stream_path)
    assert len(lines) == len(events) == 2
    assert [event.event_type for event in events] == ["first", "second"]
    assert verify_event_stream_integrity(events) == (True, 2)


def test_logger_without_stream_keeps_long_event_names(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "audit.jsonl"
    with AuditLogger(jsonl_path) as logger:
        logger.log_event("e" * 200, {"n": 1})
    assert len(jsonl_path.read_text(encoding="utf-8").strip().splitlines()) == 1
