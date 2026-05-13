# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — protobuf audit event stream

"""Event-sourced audit stream backed by length-delimited protobuf envelopes."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, TypeAlias, cast

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.message import Message

Payload: TypeAlias = dict[str, Any]

SCHEMA_VERSION = 1
STREAM_MAGIC = b"SPOA1\n"
ZERO_HASH = "0" * 64

__all__ = [
    "AuditStreamEvent",
    "EventStreamWriter",
    "iter_event_stream",
    "read_event_stream",
    "tail_event_stream",
    "verify_event_stream_integrity",
]


@dataclass
class AuditStreamEvent:
    """Decoded audit event envelope with parsed JSON payload."""

    schema_version: int
    stream_id: str
    sequence: int
    event_type: str
    recorded_at_unix_ns: int
    source: str
    previous_hash: str
    payload_json: str
    payload_sha256: str
    event_hash: str
    payload: Payload


def _audit_envelope_class() -> type[Message]:
    _ = _timestamp_pb2.DESCRIPTOR
    pool = descriptor_pool.Default()
    try:
        descriptor = pool.FindMessageTypeByName("spo.audit.AuditEnvelope")
    except KeyError:
        file_proto = descriptor_pb2.FileDescriptorProto()
        file_proto.name = "audit.proto"
        file_proto.package = "spo.audit"
        file_proto.syntax = "proto3"
        file_proto.dependency.append("google/protobuf/timestamp.proto")
        message = file_proto.message_type.add()
        message.name = "AuditEnvelope"

        def add_field(
            name: str,
            number: int,
            field_type: int,
            type_name: str | None = None,
        ) -> None:
            field = message.field.add()
            field_any = cast("Any", field)
            field_any.name = name
            field_any.number = number
            field_any.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
            field_any.type = field_type
            if type_name is not None:
                field_any.type_name = type_name

        add_field("schema_version", 1, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
        add_field("stream_id", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add_field("sequence", 3, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64)
        add_field("event_type", 4, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add_field(
            "recorded_at",
            5,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            ".google.protobuf.Timestamp",
        )
        add_field("source", 6, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add_field("previous_hash", 7, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add_field("payload_json", 8, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add_field("payload_sha256", 9, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        add_field("event_hash", 10, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
        pool.Add(file_proto)
        descriptor = pool.FindMessageTypeByName("spo.audit.AuditEnvelope")
    return message_factory.GetMessageClass(descriptor)


_AuditEnvelope = _audit_envelope_class()


def _canonical_json(payload: Payload) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint cannot encode negative values")
    chunks = bytearray()
    while value >= 0x80:
        chunks.append((value & 0x7F) | 0x80)
        value >>= 7
    chunks.append(value)
    return bytes(chunks)


def _read_varint(fh: BinaryIO) -> int | None:
    shift = 0
    result = 0
    while True:
        raw = fh.read(1)
        if raw == b"":
            if shift == 0:
                return None
            raise ValueError("truncated varint in audit event stream")
        byte = raw[0]
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result
        shift += 7
        if shift > 63:
            raise ValueError("audit event stream varint is too large")


def _event_hash(
    *,
    stream_id: str,
    sequence: int,
    event_type: str,
    recorded_at_unix_ns: int,
    source: str,
    previous_hash: str,
    payload_sha256: str,
    schema_version: int,
) -> str:
    material = _canonical_json(
        {
            "event_type": event_type,
            "payload_sha256": payload_sha256,
            "previous_hash": previous_hash,
            "recorded_at_unix_ns": recorded_at_unix_ns,
            "schema_version": schema_version,
            "sequence": sequence,
            "source": source,
            "stream_id": stream_id,
        }
    )
    return hashlib.sha256(material.encode()).hexdigest()


def _event_type_for_payload(payload: Payload) -> str:
    if payload.get("header") is True:
        return "header"
    if "event" in payload:
        return str(payload["event"])
    if "step" in payload:
        return "step"
    return "record"


def _message_to_event(message: Message) -> AuditStreamEvent:
    envelope = cast("Any", message)
    payload_json = str(envelope.payload_json)
    payload = json.loads(payload_json)
    recorded_at = envelope.recorded_at
    recorded_at_unix_ns = int(recorded_at.seconds) * 1_000_000_000 + int(
        recorded_at.nanos
    )
    return AuditStreamEvent(
        schema_version=int(envelope.schema_version),
        stream_id=str(envelope.stream_id),
        sequence=int(envelope.sequence),
        event_type=str(envelope.event_type),
        recorded_at_unix_ns=recorded_at_unix_ns,
        source=str(envelope.source),
        previous_hash=str(envelope.previous_hash),
        payload_json=payload_json,
        payload_sha256=str(envelope.payload_sha256),
        event_hash=str(envelope.event_hash),
        payload=payload,
    )


class EventStreamWriter:
    """Append length-delimited protobuf audit events to a stream file."""

    def __init__(self, path: str | Path, *, stream_id: str = "spo-audit") -> None:
        self._path = Path(path)
        self._stream_id = stream_id
        self._sequence = 0
        self._previous_hash = ZERO_HASH
        is_new = not self._path.exists() or self._path.stat().st_size == 0
        self._fh = self._path.open("ab", buffering=0)
        if is_new:
            self._fh.write(STREAM_MAGIC)
        else:
            events = read_event_stream(self._path)
            if events:
                last = events[-1]
                self._sequence = last.sequence
                self._previous_hash = last.event_hash

    def write(self, payload: Payload, *, event_type: str | None = None) -> None:
        canonical_payload = _canonical_json(payload)
        payload_sha256 = hashlib.sha256(canonical_payload.encode()).hexdigest()
        self._sequence += 1
        now_ns = time.time_ns()
        resolved_type = event_type or _event_type_for_payload(payload)
        event_hash = _event_hash(
            stream_id=self._stream_id,
            sequence=self._sequence,
            event_type=resolved_type,
            recorded_at_unix_ns=now_ns,
            source="spo",
            previous_hash=self._previous_hash,
            payload_sha256=payload_sha256,
            schema_version=SCHEMA_VERSION,
        )
        message = cast("Any", _AuditEnvelope())
        message.schema_version = SCHEMA_VERSION
        message.stream_id = self._stream_id
        message.sequence = self._sequence
        message.event_type = resolved_type
        recorded_at = message.recorded_at
        recorded_at.seconds = now_ns // 1_000_000_000
        recorded_at.nanos = now_ns % 1_000_000_000
        message.source = "spo"
        message.previous_hash = self._previous_hash
        message.payload_json = canonical_payload
        message.payload_sha256 = payload_sha256
        message.event_hash = event_hash
        raw = message.SerializeToString(deterministic=True)
        self._fh.write(_encode_varint(len(raw)))
        self._fh.write(raw)
        self._previous_hash = event_hash

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()


def _read_events_from_handle(fh: BinaryIO) -> list[AuditStreamEvent]:
    magic = fh.read(len(STREAM_MAGIC))
    if magic != STREAM_MAGIC:
        raise ValueError("not an SPO audit event stream")
    events: list[AuditStreamEvent] = []
    while True:
        size = _read_varint(fh)
        if size is None:
            return events
        raw = fh.read(size)
        if len(raw) != size:
            raise ValueError("truncated protobuf envelope in audit event stream")
        message = _AuditEnvelope()
        message.ParseFromString(raw)
        events.append(_message_to_event(message))


def read_event_stream(path: str | Path) -> list[AuditStreamEvent]:
    """Read all protobuf events from an SPO audit stream."""

    with Path(path).open("rb") as fh:
        return _read_events_from_handle(fh)


def iter_event_stream(
    path: str | Path,
    *,
    from_start: bool = False,
    poll_interval_s: float = 0.2,
) -> Iterator[AuditStreamEvent]:
    """Yield existing and newly appended stream events in order."""

    path_obj = Path(path)
    offset = 0
    if not from_start and path_obj.exists():
        offset = len(read_event_stream(path_obj))
    while True:
        current = read_event_stream(path_obj) if path_obj.exists() else []
        yield from current[offset:]
        offset = len(current)
        time.sleep(poll_interval_s)


def tail_event_stream(
    path: str | Path,
    *,
    from_start: bool = False,
    max_events: int | None = None,
    poll_interval_s: float = 0.2,
) -> list[AuditStreamEvent]:
    """Tail a stream file until ``max_events`` decoded events are available.

    Use :func:`iter_event_stream` for unbounded live streaming.
    """

    if max_events is None:
        raise ValueError("max_events is required for bounded tail_event_stream")
    events: list[AuditStreamEvent] = []
    for event in iter_event_stream(
        path,
        from_start=from_start,
        poll_interval_s=poll_interval_s,
    ):
        events.append(event)
        if len(events) >= max_events:
            return events
    return events


def verify_event_stream_integrity(
    events: list[AuditStreamEvent],
) -> tuple[bool, int]:
    """Verify payload digests, sequence continuity, and event hash chaining."""

    previous_hash = ZERO_HASH
    expected_sequence = 1
    verified = 0
    for event in events:
        canonical_payload = _canonical_json(event.payload)
        payload_sha256 = hashlib.sha256(canonical_payload.encode()).hexdigest()
        expected_hash = _event_hash(
            stream_id=event.stream_id,
            sequence=event.sequence,
            event_type=event.event_type,
            recorded_at_unix_ns=event.recorded_at_unix_ns,
            source=event.source,
            previous_hash=previous_hash,
            payload_sha256=payload_sha256,
            schema_version=event.schema_version,
        )
        if event.sequence != expected_sequence:
            return False, verified
        if event.previous_hash != previous_hash:
            return False, verified
        if event.payload_sha256 != payload_sha256:
            return False, verified
        if event.event_hash != expected_hash:
            return False, verified
        previous_hash = event.event_hash
        expected_sequence += 1
        verified += 1
    return True, verified
