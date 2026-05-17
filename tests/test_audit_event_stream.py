# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — audit event-stream tests

from __future__ import annotations

import hashlib
import json

import numpy as np
from click.testing import CliRunner

from scpn_phase_orchestrator.audit.logger import AuditLogger
from scpn_phase_orchestrator.audit.replay import ReplayEngine
from scpn_phase_orchestrator.audit.stream import (
    read_event_stream,
    verify_event_stream_integrity,
)
from scpn_phase_orchestrator.cli import main
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _state() -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=0.82, psi=0.5)],
        cross_layer_alignment=np.zeros((1, 1)),
        stability_proxy=0.82,
        regime_id="nominal",
    )


def test_audit_logger_dual_writes_jsonl_and_protobuf_stream(tmp_path) -> None:
    jsonl_path = tmp_path / "audit.jsonl"
    stream_path = tmp_path / "audit.spoa"

    with AuditLogger(jsonl_path, event_stream=stream_path) as logger:
        logger.log_header(n_oscillators=2, dt=0.01, seed=7)
        logger.log_step(
            0,
            _state(),
            [],
            phases=np.array([0.1, 0.2]),
            omegas=np.array([1.0, 1.0]),
            knm=np.array([[0.0, 0.3], [0.3, 0.0]]),
            alpha=np.zeros((2, 2)),
        )

    jsonl_entries = [
        json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    stream_events = read_event_stream(stream_path)

    assert [event.sequence for event in stream_events] == [1, 2]
    assert [event.event_type for event in stream_events] == ["header", "step"]
    assert stream_events[0].previous_hash == "0" * 64
    assert stream_events[1].previous_hash == stream_events[0].event_hash
    assert stream_events[1].payload["step"] == 0
    assert stream_events[1].payload["regime"] == "nominal"
    assert stream_events[1].payload["_hash"] == jsonl_entries[1]["_hash"]

    ok, verified = verify_event_stream_integrity(stream_events)
    assert ok is True
    assert verified == 2


def test_event_stream_integrity_detects_payload_tampering(tmp_path) -> None:
    stream_path = tmp_path / "audit.spoa"
    with AuditLogger(tmp_path / "audit.jsonl", event_stream=stream_path) as logger:
        logger.log_event("operator_note", {"step": 3, "detail": "baseline"})

    events = read_event_stream(stream_path)
    events[0].payload["detail"] = "tampered"

    ok, verified = verify_event_stream_integrity(events)
    assert ok is False
    assert verified == 0


def test_event_stream_unsigned_development_mode_is_explicit(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("SPO_AUDIT_KEY", raising=False)
    monkeypatch.delenv("SPO_AUDIT_KEYRING", raising=False)
    stream_path = tmp_path / "audit.spoa"

    with AuditLogger(tmp_path / "audit.jsonl", event_stream=stream_path) as logger:
        logger.log_event("operator_note", {"step": 3, "detail": "unsigned"})

    events = read_event_stream(stream_path)
    assert events[0].audit_mode == "unsigned-development"
    assert events[0].signature_algorithm == ""
    assert events[0].signature_key_id == ""
    assert events[0].signature == ""


def test_event_stream_integrity_verifies_hmac_signed_envelopes(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SPO_AUDIT_KEY", "stream-signing-key")
    stream_path = tmp_path / "audit.spoa"

    with AuditLogger(tmp_path / "audit.jsonl", event_stream=stream_path) as logger:
        logger.log_event("operator_note", {"step": 3, "detail": "signed"})

    events = read_event_stream(stream_path)
    key_id = hashlib.sha256(b"stream-signing-key").hexdigest()[:16]

    assert events[0].signature_key_id == key_id
    assert events[0].signature_algorithm == "HMAC-SHA256"
    assert len(events[0].signature) == 64
    ok, verified = verify_event_stream_integrity(events)
    assert ok is True
    assert verified == 1


def test_event_stream_integrity_requires_hmac_when_key_is_configured(
    tmp_path, monkeypatch
) -> None:
    stream_path = tmp_path / "audit.spoa"
    with AuditLogger(tmp_path / "audit.jsonl", event_stream=stream_path) as logger:
        logger.log_event("operator_note", {"step": 3, "detail": "unsigned"})

    monkeypatch.setenv("SPO_AUDIT_KEY", "stream-signing-key")
    events = read_event_stream(stream_path)

    ok, verified = verify_event_stream_integrity(events)
    assert ok is False
    assert verified == 0


def test_jsonl_and_stream_append_continue_existing_hash_chains(tmp_path) -> None:
    jsonl_path = tmp_path / "audit.jsonl"
    stream_path = tmp_path / "audit.spoa"

    with AuditLogger(jsonl_path, event_stream=stream_path) as logger:
        logger.log_event("operator_note", {"step": 1, "detail": "first"})
    with AuditLogger(jsonl_path, event_stream=stream_path) as logger:
        logger.log_event("operator_note", {"step": 2, "detail": "second"})

    jsonl_entries = [
        json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    stream_events = read_event_stream(stream_path)

    assert jsonl_entries[0]["_hash"] != jsonl_entries[1]["_hash"]
    jsonl_ok, jsonl_verified = ReplayEngine.verify_integrity(jsonl_entries)
    assert jsonl_ok is True
    assert jsonl_verified == 2
    assert stream_events[1].sequence == 2
    assert stream_events[1].previous_hash == stream_events[0].event_hash

    ok, verified = verify_event_stream_integrity(stream_events)
    assert ok is True
    assert verified == 2


def test_spo_watch_replays_protobuf_stream_from_start(tmp_path) -> None:
    stream_path = tmp_path / "audit.spoa"
    with AuditLogger(tmp_path / "audit.jsonl", event_stream=stream_path) as logger:
        logger.log_header(n_oscillators=2, dt=0.01)
        logger.log_step(
            0,
            _state(),
            [],
            phases=np.array([0.1, 0.2]),
            omegas=np.array([1.0, 1.0]),
            knm=np.array([[0.0, 0.3], [0.3, 0.0]]),
            alpha=np.zeros((2, 2)),
        )

    result = CliRunner().invoke(
        main,
        [
            "watch",
            str(stream_path),
            "--format",
            "protobuf",
            "--from-start",
            "--max-events",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert "#1 header" in result.output
    assert "#2 step step=0 regime=nominal stability=0.8200" in result.output
    assert "stream integrity: OK (2 events)" in result.output
