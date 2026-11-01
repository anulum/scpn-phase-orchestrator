# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — audit event-stream tests

from __future__ import annotations

import cProfile
import hashlib
import importlib.util
import io
import json
import time
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from google.protobuf import descriptor_pb2, descriptor_pool
from grpc_tools import protoc

from scpn_phase_orchestrator.runtime import audit_stream as audit_stream_module
from scpn_phase_orchestrator.runtime.audit_logger import AuditLogger
from scpn_phase_orchestrator.runtime.audit_signing import SIGNATURE_ALGORITHM
from scpn_phase_orchestrator.runtime.audit_stream import (
    STREAM_MAGIC,
    AuditStreamEvent,
    AuditStreamTornTailWarning,
    EventStreamWriter,
    _AuditEnvelope,
    _canonical_json,
    _encode_varint,
    _event_type_for_payload,
    _read_varint,
    _reject_json_constant,
    iter_event_stream,
    read_event_stream,
    tail_event_stream,
    verify_event_stream_integrity,
)
from scpn_phase_orchestrator.runtime.cli import main
from scpn_phase_orchestrator.runtime.replay import ReplayEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


class _StopPollingError(Exception):
    """Sentinel used to break the unbounded ``iter_event_stream`` poll loop."""


def test_presplit_audit_stream_submodule_is_removed() -> None:
    assert importlib.util.find_spec("scpn_phase_orchestrator.audit.stream") is None


def test_audit_envelope_descriptor_is_not_registered_in_default_pool() -> None:
    default_pool = descriptor_pool.Default()

    with pytest.raises(KeyError):
        default_pool.FindMessageTypeByName("spo.audit.AuditEnvelope")


def test_audit_envelope_descriptor_contract_is_stable() -> None:
    descriptor = _AuditEnvelope.DESCRIPTOR
    assert descriptor.full_name == "spo.audit.AuditEnvelope"
    assert [field.name for field in descriptor.fields] == [
        "schema_version",
        "stream_id",
        "sequence",
        "event_type",
        "recorded_at",
        "source",
        "previous_hash",
        "payload_json",
        "payload_sha256",
        "event_hash",
        "signature_algorithm",
        "signature_key_id",
        "signature",
        "audit_mode",
    ]


_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "proto_path",
    [
        _ROOT / "proto/audit.proto",
        _ROOT / "src/scpn_phase_orchestrator/audit/audit.proto",
    ],
)
def test_static_audit_proto_matches_runtime_descriptor(
    proto_path: Path, tmp_path: Path
) -> None:
    runtime_fields = [
        (field.name, field.number, field.type)
        for field in _AuditEnvelope.DESCRIPTOR.fields
    ]

    assert _compiled_audit_proto_fields(proto_path, tmp_path) == runtime_fields


def _compiled_audit_proto_fields(
    proto_path: Path, tmp_path: Path
) -> list[tuple[str, int, int]]:
    """Compile ``proto_path`` and return the AuditEnvelope field contract."""
    import grpc_tools

    descriptor_out = tmp_path / f"{proto_path.parent.name}_audit.desc"
    include_dir = Path(grpc_tools.__file__).parent / "_proto"
    rc = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{proto_path.parent}",
            f"-I{include_dir}",
            f"--descriptor_set_out={descriptor_out}",
            proto_path.name,
        ]
    )
    assert rc == 0

    descriptor_set = descriptor_pb2.FileDescriptorSet()
    descriptor_set.ParseFromString(descriptor_out.read_bytes())
    for file_descriptor in descriptor_set.file:
        for message in file_descriptor.message_type:
            if message.name == "AuditEnvelope":
                return [
                    (field.name, field.number, field.type) for field in message.field
                ]
    raise AssertionError(f"AuditEnvelope not found in compiled {proto_path}")


def _state() -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=0.82, psi=0.5)],
        cross_layer_alignment=np.zeros((1, 1)),
        stability_proxy=0.82,
        regime_id="nominal",
    )


def test_audit_logger_dual_writes_jsonl_and_protobuf_stream(tmp_path: Path) -> None:
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


def test_event_stream_integrity_detects_payload_tampering(tmp_path: Path) -> None:
    stream_path = tmp_path / "audit.spoa"
    with AuditLogger(tmp_path / "audit.jsonl", event_stream=stream_path) as logger:
        logger.log_event("operator_note", {"step": 3, "detail": "baseline"})

    events = read_event_stream(stream_path)
    events[0].payload["detail"] = "tampered"

    ok, verified = verify_event_stream_integrity(events)
    assert ok is False
    assert verified == 0


def test_event_stream_unsigned_development_mode_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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


def test_event_stream_writer_rejects_non_finite_payload_numbers(tmp_path: Path) -> None:
    stream_path = tmp_path / "audit.spoa"
    writer = EventStreamWriter(stream_path)
    try:
        with pytest.raises(ValueError, match="finite JSON"):
            writer.write({"event": "operator_note", "R": float("nan")})
    finally:
        writer.close()


def test_event_stream_integrity_verifies_hmac_signed_envelopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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


def test_event_stream_integrity_rejects_tampered_audit_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPO_AUDIT_KEY", "stream-signing-key")
    stream_path = tmp_path / "audit.spoa"

    with AuditLogger(tmp_path / "audit.jsonl", event_stream=stream_path) as logger:
        logger.log_event("operator_note", {"step": 3, "detail": "signed"})

    events = read_event_stream(stream_path)
    events[0].audit_mode = "unsigned-development"

    ok, verified = verify_event_stream_integrity(events)
    assert ok is False
    assert verified == 0


def test_event_stream_integrity_requires_hmac_when_key_is_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream_path = tmp_path / "audit.spoa"
    with AuditLogger(tmp_path / "audit.jsonl", event_stream=stream_path) as logger:
        logger.log_event("operator_note", {"step": 3, "detail": "unsigned"})

    monkeypatch.setenv("SPO_AUDIT_KEY", "stream-signing-key")
    events = read_event_stream(stream_path)

    ok, verified = verify_event_stream_integrity(events)
    assert ok is False
    assert verified == 0


def test_jsonl_and_stream_append_continue_existing_hash_chains(tmp_path: Path) -> None:
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


def test_spo_watch_replays_protobuf_stream_from_start(tmp_path: Path) -> None:
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


def test_audit_stream_event_rejects_payload_hash_mismatch() -> None:
    payload = {"event": "operator_note", "step": 1}
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    with pytest.raises(ValueError, match="payload_sha256 mismatch"):
        AuditStreamEvent(
            schema_version=1,
            stream_id="spo-audit",
            sequence=1,
            event_type="operator_note",
            recorded_at_unix_ns=1,
            source="runtime",
            previous_hash="0" * 64,
            payload_json=payload_json,
            payload_sha256="f" * 64,
            event_hash="1" * 64,
            signature_algorithm="",
            signature_key_id="",
            signature="",
            audit_mode="unsigned-development",
            payload=payload,
        )


def test_audit_stream_event_rejects_signed_mode_without_signature_metadata() -> None:
    payload = {"event": "operator_note", "step": 1}
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
    with pytest.raises(
        ValueError, match="signed audit_mode requires expected signature_algorithm"
    ):
        AuditStreamEvent(
            schema_version=1,
            stream_id="spo-audit",
            sequence=1,
            event_type="operator_note",
            recorded_at_unix_ns=1,
            source="runtime",
            previous_hash="0" * 64,
            payload_json=payload_json,
            payload_sha256=payload_hash,
            event_hash="1" * 64,
            signature_algorithm="",
            signature_key_id="",
            signature="",
            audit_mode="signed",
            payload=payload,
        )


@pytest.mark.parametrize("event_type", ["bad\ntype", "a" * 129])
def test_audit_stream_event_rejects_invalid_event_type_label(event_type: str) -> None:
    payload = {"event": "operator_note", "step": 1}
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
    with pytest.raises(ValueError, match="event_type"):
        AuditStreamEvent(
            schema_version=1,
            stream_id="spo-audit",
            sequence=1,
            event_type=event_type,
            recorded_at_unix_ns=1,
            source="runtime",
            previous_hash="0" * 64,
            payload_json=payload_json,
            payload_sha256=payload_hash,
            event_hash="1" * 64,
            signature_algorithm="",
            signature_key_id="",
            signature="",
            audit_mode="unsigned-development",
            payload=payload,
        )


@pytest.mark.parametrize("source", ["bad\nsource", "a" * 129])
def test_audit_stream_event_rejects_invalid_source_label(source: str) -> None:
    payload = {"event": "operator_note", "step": 1}
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
    with pytest.raises(ValueError, match="source"):
        AuditStreamEvent(
            schema_version=1,
            stream_id="spo-audit",
            sequence=1,
            event_type="operator_note",
            recorded_at_unix_ns=1,
            source=source,
            previous_hash="0" * 64,
            payload_json=payload_json,
            payload_sha256=payload_hash,
            event_hash="1" * 64,
            signature_algorithm="",
            signature_key_id="",
            signature="",
            audit_mode="unsigned-development",
            payload=payload,
        )


@pytest.mark.parametrize(
    "poll_interval_s", [True, -0.1, float("nan"), float("inf"), "0.2"]
)
def test_iter_event_stream_rejects_invalid_poll_interval(
    poll_interval_s: object,
) -> None:
    iter_any: Callable[..., Iterator[AuditStreamEvent]] = iter_event_stream
    with pytest.raises(ValueError, match="poll_interval_s"):
        iterator = iter_any(
            "nonexistent.spoa",
            poll_interval_s=poll_interval_s,
        )
        next(iterator)


@pytest.mark.parametrize("max_events", [0, -1, True, 1.5, "2"])
def test_tail_event_stream_rejects_invalid_max_events(max_events: object) -> None:
    tail_any: Callable[..., object] = tail_event_stream
    with pytest.raises(ValueError, match="max_events"):
        tail_any(
            "nonexistent.spoa",
            max_events=max_events,
            poll_interval_s=0.0,
        )


def test_iter_event_stream_missing_path_requires_from_start() -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        iterator = iter_event_stream(
            "nonexistent.spoa",
            from_start=False,
            poll_interval_s=0.0,
        )
        next(iterator)


def test_tail_event_stream_missing_path_requires_from_start() -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        tail_event_stream(
            "nonexistent.spoa",
            from_start=False,
            max_events=1,
            poll_interval_s=0.0,
        )


@pytest.mark.parametrize("stream_id", ["", "bad\nid", "a" * 129, True])
def test_event_stream_writer_rejects_invalid_stream_id(
    tmp_path: Path,
    stream_id: object,
) -> None:
    writer_any: Callable[..., EventStreamWriter] = EventStreamWriter
    with pytest.raises(ValueError, match="stream_id"):
        writer_any(
            tmp_path / "audit.spoa",
            stream_id=stream_id,
        )


def _valid_event(**overrides: object) -> AuditStreamEvent:
    payload = {"note": "x"}
    payload_json = _canonical_json(payload)
    fields: dict[str, object] = {
        "schema_version": 1,
        "stream_id": "spo-audit",
        "sequence": 1,
        "event_type": "operator_note",
        "recorded_at_unix_ns": 1,
        "source": "runtime",
        "previous_hash": "0" * 64,
        "payload_json": payload_json,
        "payload_sha256": hashlib.sha256(payload_json.encode()).hexdigest(),
        "event_hash": "1" * 64,
        "signature_algorithm": "",
        "signature_key_id": "",
        "signature": "",
        "audit_mode": "unsigned",
        "payload": payload,
    }
    fields.update(overrides)
    event_any: Callable[..., AuditStreamEvent] = AuditStreamEvent
    return event_any(**fields)


def test_valid_audit_stream_event_constructs() -> None:
    assert _valid_event().schema_version == 1


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"schema_version": 1.5}, "schema_version must be an integer"),
        ({"schema_version": 0}, "schema_version must be positive"),
        ({"stream_id": ""}, "stream_id must be a non-empty string"),
        ({"stream_id": "a\x01b"}, "stream_id must not contain control"),
        ({"sequence": 1.5}, "sequence must be an integer"),
        ({"sequence": 0}, "sequence must be positive"),
        ({"recorded_at_unix_ns": -1}, "recorded_at_unix_ns must be a non-negative"),
        ({"event_type": ""}, "event_type must be a non-empty string"),
        ({"source": ""}, "source must be a non-empty string"),
        ({"previous_hash": "bad"}, "must be a lowercase 64-char hex digest"),
        ({"payload_json": ""}, "payload_json must be a non-empty string"),
        ({"payload": "not-a-dict"}, "payload must be a JSON object mapping"),
        ({"payload_json": '{"a":1}'}, "must match canonical JSON encoding"),
        ({"audit_mode": ""}, "audit_mode must be a non-empty string"),
        ({"signature_algorithm": 1}, "signature_algorithm must be a string"),
        ({"signature_key_id": 1}, "signature_key_id must be a string"),
        ({"signature": 1}, "signature must be a string"),
        (
            {"audit_mode": "signed", "signature_algorithm": "wrong"},
            "expected signature_algorithm",
        ),
        (
            {
                "audit_mode": "signed",
                "signature_algorithm": SIGNATURE_ALGORITHM,
                "signature_key_id": "bad",
                "signature": "f" * 64,
            },
            "16-char lowercase key id",
        ),
        (
            {
                "audit_mode": "signed",
                "signature_algorithm": SIGNATURE_ALGORITHM,
                "signature_key_id": "0" * 16,
                "signature": "bad",
            },
            "64-char lowercase signature",
        ),
        ({"signature": "abc"}, "unsigned audit_mode must not include signature"),
    ],
)
def test_audit_stream_event_rejects_invalid_fields(
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _valid_event(**overrides)


def test_reject_json_constant_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        _reject_json_constant("Infinity")


def test_encode_varint_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="cannot encode negative"):
        _encode_varint(-1)


def test_read_varint_rejects_truncated_input() -> None:
    with pytest.raises(ValueError, match="truncated varint"):
        _read_varint(io.BytesIO(b"\x80"))


def test_read_varint_rejects_oversized_value() -> None:
    with pytest.raises(ValueError, match="varint is too large"):
        _read_varint(io.BytesIO(b"\x80" * 10))


def test_event_type_for_payload_falls_back_to_record() -> None:
    assert _event_type_for_payload({"unrelated": 1}) == "record"


def test_verify_rejects_event_with_unchained_hash() -> None:
    ok, count = verify_event_stream_integrity([_valid_event()])
    assert ok is False
    assert count == 0


def test_verify_rejects_out_of_order_sequence() -> None:
    ok, count = verify_event_stream_integrity([_valid_event(sequence=2)])
    assert ok is False
    assert count == 0


def test_verify_rejects_broken_previous_hash_chain() -> None:
    ok, count = verify_event_stream_integrity([_valid_event(previous_hash="1" * 64)])
    assert ok is False
    assert count == 0


def test_event_stream_writer_rejects_empty_audit_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPO_AUDIT_KEY", "")
    with pytest.raises(ValueError, match="SPO_AUDIT_KEY must not be empty"):
        EventStreamWriter(tmp_path / "audit.spoa")


def test_event_stream_writer_resumes_from_existing_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPO_AUDIT_KEY", raising=False)
    path = tmp_path / "audit.spoa"
    writer = EventStreamWriter(path)
    writer.write({"event": "first"})
    writer.write({"event": "second"})
    writer.close()

    resumed = EventStreamWriter(path)
    resumed.write({"event": "third"})
    resumed.close()

    events = read_event_stream(path)
    assert [event.sequence for event in events] == [1, 2, 3]


def test_read_event_stream_rejects_truncated_envelope(tmp_path: Path) -> None:
    path = tmp_path / "audit.spoa"
    path.write_bytes(STREAM_MAGIC + _encode_varint(64) + b"short")
    with pytest.raises(ValueError, match="truncated protobuf envelope"):
        read_event_stream(path)


def test_read_event_stream_rejects_unknown_stream_magic(tmp_path: Path) -> None:
    path = tmp_path / "audit.spoa"
    path.write_bytes(b"WRONG\n")
    with pytest.raises(ValueError, match="not an SPO audit event stream"):
        read_event_stream(path)


def _two_event_stream_bytes(tmp_path: Path) -> tuple[bytes, bytes]:
    """Return a one-event stream and the bytes of its complete second frame."""
    path = tmp_path / "source.spoa"
    writer = EventStreamWriter(path)
    writer.write({"event": "first"})
    writer.close()
    first = path.read_bytes()
    writer = EventStreamWriter(path)
    writer.write({"event": "second"})
    writer.close()
    return first, path.read_bytes()[len(first) :]


@pytest.mark.parametrize("cut", [1, 2, -1], ids=["varint", "body-start", "body-end"])
def test_partial_tail_reader_returns_the_complete_events(
    tmp_path: Path, cut: int
) -> None:
    """A stream ending inside a frame yields the events before it, strictly not."""
    first, frame = _two_event_stream_bytes(tmp_path)
    path = tmp_path / "live.spoa"
    # A one-byte cut of a multi-byte varint ends inside the length prefix.
    assert frame[0] & 0x80
    path.write_bytes(first + frame[:cut])
    events = read_event_stream(path, allow_partial_tail=True)
    assert [event.event_type for event in events] == ["first"]
    with pytest.raises(ValueError, match="truncated"):
        read_event_stream(path)


@pytest.mark.parametrize("length", [0, 1, len(STREAM_MAGIC) - 1])
def test_partial_tail_reader_waits_for_the_magic_header(
    tmp_path: Path, length: int
) -> None:
    """A header still being written reads as empty; a wrong header still fails."""
    path = tmp_path / "live.spoa"
    path.write_bytes(STREAM_MAGIC[:length])
    assert read_event_stream(path, allow_partial_tail=True) == []
    path.write_bytes(b"WRONG\n")
    with pytest.raises(ValueError, match="not an SPO audit event stream"):
        read_event_stream(path, allow_partial_tail=True)


def test_iter_event_stream_waits_for_a_frame_in_flight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A poll that sees half a frame yields it once the writer finishes it."""
    first, frame = _two_event_stream_bytes(tmp_path)
    path = tmp_path / "live.spoa"
    path.write_bytes(first + frame[:3])
    polls: list[float] = []

    def _finish_frame(seconds: float) -> None:
        polls.append(seconds)
        if len(polls) > 1:
            raise _StopPollingError
        with path.open("ab") as fh:
            fh.write(frame[3:])

    assert vars(audit_stream_module)["time"] is time
    monkeypatch.setattr(time, "sleep", _finish_frame)
    collected: list[AuditStreamEvent] = []
    with pytest.raises(_StopPollingError):
        for event in iter_event_stream(path, from_start=True, poll_interval_s=0.0):
            collected.append(event)
    assert [event.event_type for event in collected] == ["first", "second"]
    assert verify_event_stream_integrity(collected) == (True, 2)


def _poll_stream(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    polls: int,
    torn_tail_polls: int,
    on_poll: Callable[[int], None] | None = None,
) -> tuple[list[str], list[AuditStreamTornTailWarning]]:
    """Run ``iter_event_stream`` for ``polls`` polls; return events and reports."""
    count = [0]

    def _sleep(_seconds: float) -> None:
        count[0] += 1
        if count[0] >= polls:
            raise _StopPollingError
        if on_poll is not None:
            on_poll(count[0])

    assert vars(audit_stream_module)["time"] is time
    monkeypatch.setattr(time, "sleep", _sleep)
    collected: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(_StopPollingError):
            for event in iter_event_stream(
                path,
                from_start=True,
                poll_interval_s=0.0,
                torn_tail_polls=torn_tail_polls,
            ):
                collected.append(event.event_type)
    reports = [
        item.message
        for item in caught
        if isinstance(item.message, AuditStreamTornTailWarning)
    ]
    return collected, reports


def test_iter_event_stream_reports_a_persistent_torn_tail_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frame left incomplete for many polls is reported once, not ignored."""
    first, frame = _two_event_stream_bytes(tmp_path)
    path = tmp_path / "live.spoa"
    path.write_bytes(first + frame[:3])

    events, reports = _poll_stream(path, monkeypatch, polls=10, torn_tail_polls=4)

    assert events == ["first"]
    assert len(reports) == 1
    message = str(reports[0])
    assert f"incomplete frame of 3 bytes at offset {len(first)}" in message
    assert "unchanged for 4 polls" in message


def test_iter_event_stream_does_not_report_a_frame_finished_in_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writer that finishes the frame before the threshold is not reported."""
    first, frame = _two_event_stream_bytes(tmp_path)
    path = tmp_path / "live.spoa"
    path.write_bytes(first + frame[:3])

    def _finish(poll: int) -> None:
        if poll == 2:
            with path.open("ab") as fh:
                fh.write(frame[3:])

    events, reports = _poll_stream(
        path, monkeypatch, polls=8, torn_tail_polls=4, on_poll=_finish
    )

    assert events == ["first", "second"]
    assert reports == []


def test_iter_event_stream_reports_again_when_the_torn_tail_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tail that grows but stays incomplete re-arms the report."""
    first, frame = _two_event_stream_bytes(tmp_path)
    path = tmp_path / "live.spoa"
    path.write_bytes(first + frame[:2])

    def _grow(poll: int) -> None:
        if poll == 4:
            with path.open("ab") as fh:
                fh.write(frame[2:3])

    events, reports = _poll_stream(
        path, monkeypatch, polls=8, torn_tail_polls=3, on_poll=_grow
    )

    assert events == ["first"]
    assert len(reports) == 2
    assert f"incomplete frame of 2 bytes at offset {len(first)}" in str(reports[0])
    assert f"incomplete frame of 3 bytes at offset {len(first)}" in str(reports[1])


def test_iter_event_stream_reports_a_stalled_magic_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A header that never completes is a torn tail at offset zero."""
    path = tmp_path / "live.spoa"
    path.write_bytes(STREAM_MAGIC[:2])

    events, reports = _poll_stream(path, monkeypatch, polls=5, torn_tail_polls=2)

    assert events == []
    assert len(reports) == 1
    assert "incomplete frame of 2 bytes at offset 0" in str(reports[0])


def test_iter_event_stream_waits_for_a_stream_that_does_not_exist_yet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """From the start, a missing file is polled until it appears, not reported."""
    first, _frame = _two_event_stream_bytes(tmp_path)
    path = tmp_path / "late.spoa"

    def _create(poll: int) -> None:
        if poll == 3:
            path.write_bytes(first)

    events, reports = _poll_stream(
        path, monkeypatch, polls=6, torn_tail_polls=1, on_poll=_create
    )

    assert events == ["first"]
    assert reports == []


@pytest.mark.parametrize("torn_tail_polls", [0, -1, True, 1.5])
def test_iter_event_stream_rejects_invalid_torn_tail_polls(
    tmp_path: Path, torn_tail_polls: object
) -> None:
    iter_any: Callable[..., Iterator[AuditStreamEvent]] = iter_event_stream
    with pytest.raises(ValueError, match="torn_tail_polls must be a positive integer"):
        next(
            iter_any(
                tmp_path / "x.spoa", from_start=True, torn_tail_polls=torn_tail_polls
            )
        )


def test_writer_emits_each_frame_in_one_write(tmp_path: Path) -> None:
    """The length prefix and envelope reach the file in a single write call."""
    writer = EventStreamWriter(tmp_path / "audit.spoa")
    profiler = cProfile.Profile()
    profiler.enable()
    for step in range(4):
        writer.write({"step": step})
    profiler.disable()
    writer.close()
    writes = sum(
        entry.callcount
        for entry in profiler.getstats()
        if isinstance(entry.code, str)
        and "method 'write' of '_io.FileIO' objects" in entry.code
    )
    assert writes == 4


def test_tail_event_stream_requires_max_events(tmp_path: Path) -> None:
    path = tmp_path / "audit.spoa"
    EventStreamWriter(path).close()
    with pytest.raises(ValueError, match="max_events is required"):
        tail_event_stream(path)


def test_iter_event_stream_skips_existing_events_when_not_from_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "audit.spoa"
    writer = EventStreamWriter(path)
    writer.write({"event": "existing"})
    writer.close()

    def _stop(_seconds: float) -> None:
        raise _StopPollingError

    assert vars(audit_stream_module)["time"] is time
    monkeypatch.setattr(time, "sleep", _stop)
    collected: list[AuditStreamEvent] = []
    with pytest.raises(_StopPollingError):
        for event in iter_event_stream(path, from_start=False, poll_interval_s=0.0):
            collected.append(event)
    assert collected == []


def test_event_stream_writer_resume_handles_magic_only_stream(tmp_path: Path) -> None:
    path = tmp_path / "audit.spoa"
    EventStreamWriter(path).close()

    resumed = EventStreamWriter(path)
    resumed.write({"event": "first"})
    resumed.close()

    events = read_event_stream(path)
    assert [event.sequence for event in events] == [1]


def test_tail_event_stream_returns_at_max_events(tmp_path: Path) -> None:
    path = tmp_path / "audit.spoa"
    writer = EventStreamWriter(path)
    writer.write({"event": "a"})
    writer.write({"event": "b"})
    writer.close()

    tailed = tail_event_stream(path, from_start=True, max_events=1, poll_interval_s=0.0)
    assert [event.event_type for event in tailed] == ["a"]


def test_tail_event_stream_accumulates_below_max_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "audit.spoa"
    writer = EventStreamWriter(path)
    writer.write({"event": "a"})
    writer.write({"event": "b"})
    writer.close()

    def _stop(_seconds: float) -> None:
        raise _StopPollingError

    assert vars(audit_stream_module)["time"] is time
    monkeypatch.setattr(time, "sleep", _stop)
    with pytest.raises(_StopPollingError):
        tail_event_stream(path, from_start=True, max_events=5, poll_interval_s=0.0)


def test_verify_fails_closed_on_invalid_keyring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPO_AUDIT_KEY", "stream-signing-key")
    monkeypatch.setenv("SPO_AUDIT_KEYRING", "not-json")
    ok, verified = verify_event_stream_integrity([_valid_event()])
    assert ok is False
    assert verified == 0


def _read_signed_event(tmp_path: Path) -> AuditStreamEvent:
    stream_path = tmp_path / "audit.spoa"
    with AuditLogger(tmp_path / "audit.jsonl", event_stream=stream_path) as logger:
        logger.log_event("operator_note", {"step": 1, "detail": "signed"})
    return read_event_stream(stream_path)[0]


def test_verify_rejects_signed_event_with_wrong_signature_algorithm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPO_AUDIT_KEY", "stream-signing-key")
    event = _read_signed_event(tmp_path)
    event.signature_algorithm = "WRONG-ALGORITHM"
    ok, verified = verify_event_stream_integrity([event])
    assert ok is False
    assert verified == 0


def test_verify_rejects_signed_event_with_short_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPO_AUDIT_KEY", "stream-signing-key")
    event = _read_signed_event(tmp_path)
    event.signature = "deadbeef"
    ok, verified = verify_event_stream_integrity([event])
    assert ok is False
    assert verified == 0


def test_verify_rejects_signed_event_with_unknown_key_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPO_AUDIT_KEY", "stream-signing-key")
    event = _read_signed_event(tmp_path)
    event.signature_key_id = "0" * 16
    ok, verified = verify_event_stream_integrity([event])
    assert ok is False
    assert verified == 0


def test_verify_rejects_signed_event_when_keyring_id_mismatches_material(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPO_AUDIT_KEY", "stream-signing-key")
    event = _read_signed_event(tmp_path)
    monkeypatch.setattr(
        audit_stream_module,
        "audit_verification_keys",
        lambda: {event.signature_key_id: "different-key-material"},
    )

    ok, verified = verify_event_stream_integrity([event])

    assert ok is False
    assert verified == 0
