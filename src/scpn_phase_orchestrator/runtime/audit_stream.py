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
import hmac
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, TypeAlias, cast

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.message import Message

from scpn_phase_orchestrator.runtime.audit_signing import (
    SIGNATURE_ALGORITHM,
    audit_verification_keys,
    key_id_for_secret,
)

Payload: TypeAlias = dict[str, Any]

SCHEMA_VERSION = 1
STREAM_MAGIC = b"SPOA1\n"
ZERO_HASH = "0" * 64
_AUDIT_LABEL_MAX_LEN = 128

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
    signature_algorithm: str
    signature_key_id: str
    signature: str
    audit_mode: str
    payload: Payload

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise ValueError("schema_version must be an integer")
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if not isinstance(self.stream_id, str) or not self.stream_id:
            raise ValueError("stream_id must be a non-empty string")
        if any(ord(char) < 32 for char in self.stream_id):
            raise ValueError("stream_id must not contain control characters")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise ValueError("sequence must be an integer")
        if self.sequence <= 0:
            raise ValueError("sequence must be positive")
        if (
            isinstance(self.recorded_at_unix_ns, bool)
            or not isinstance(self.recorded_at_unix_ns, int)
            or self.recorded_at_unix_ns < 0
        ):
            raise ValueError("recorded_at_unix_ns must be a non-negative integer")
        if not isinstance(self.event_type, str) or not self.event_type:
            raise ValueError("event_type must be a non-empty string")
        if len(self.event_type) > _AUDIT_LABEL_MAX_LEN:
            raise ValueError(
                f"event_type must be at most {_AUDIT_LABEL_MAX_LEN} characters"
            )
        if any(ord(char) < 32 for char in self.event_type):
            raise ValueError("event_type must not contain control characters")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty string")
        if len(self.source) > _AUDIT_LABEL_MAX_LEN:
            raise ValueError(
                f"source must be at most {_AUDIT_LABEL_MAX_LEN} characters"
            )
        if any(ord(char) < 32 for char in self.source):
            raise ValueError("source must not contain control characters")
        for field_name in ("previous_hash", "payload_sha256", "event_hash"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)
            ):
                raise ValueError(f"{field_name} must be a lowercase 64-char hex digest")
        if not isinstance(self.payload_json, str) or not self.payload_json:
            raise ValueError("payload_json must be a non-empty string")
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be a JSON object mapping")
        if _canonical_json(self.payload) != self.payload_json:
            raise ValueError(
                "payload_json must match canonical JSON encoding of payload"
            )
        computed_payload_hash = hashlib.sha256(self.payload_json.encode()).hexdigest()
        if computed_payload_hash != self.payload_sha256:
            raise ValueError("payload_sha256 mismatch for payload_json")
        if not isinstance(self.audit_mode, str) or not self.audit_mode:
            raise ValueError("audit_mode must be a non-empty string")
        if not isinstance(self.signature_algorithm, str):
            raise ValueError("signature_algorithm must be a string")
        if not isinstance(self.signature_key_id, str):
            raise ValueError("signature_key_id must be a string")
        if not isinstance(self.signature, str):
            raise ValueError("signature must be a string")
        is_signed = self.audit_mode in {"signed", "hmac-signed"}
        if is_signed:
            if self.signature_algorithm != SIGNATURE_ALGORITHM:
                raise ValueError(
                    "signed audit_mode requires expected signature_algorithm"
                )
            if len(self.signature_key_id) != 16 or any(
                ch not in "0123456789abcdef" for ch in self.signature_key_id
            ):
                raise ValueError("signed audit_mode requires 16-char lowercase key id")
            if len(self.signature) != 64 or any(
                ch not in "0123456789abcdef" for ch in self.signature
            ):
                raise ValueError(
                    "signed audit_mode requires 64-char lowercase signature"
                )
        else:
            if self.signature_algorithm or self.signature_key_id or self.signature:
                raise ValueError(
                    "unsigned audit_mode must not include signature metadata"
                )


def _build_audit_envelope_file_proto() -> descriptor_pb2.FileDescriptorProto:
    """Build the protobuf file descriptor for the audit envelope."""
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
        """Add a field descriptor to the protobuf message being built."""
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
    add_field(
        "signature_algorithm", 11, descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    )
    add_field("signature_key_id", 12, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    add_field("signature", 13, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    add_field("audit_mode", 14, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    return file_proto


_AUDIT_ENVELOPE_FILE_PROTO = _build_audit_envelope_file_proto()
_AUDIT_ENVELOPE_FILE_PROTO_BYTES = _AUDIT_ENVELOPE_FILE_PROTO.SerializeToString()


def _audit_envelope_class() -> type[Message]:
    """Return the dynamically-built audit envelope protobuf class."""
    _ = _timestamp_pb2.DESCRIPTOR
    pool = descriptor_pool.DescriptorPool()
    try:
        pool.AddSerializedFile(_timestamp_pb2.DESCRIPTOR.serialized_pb)  # type: ignore[no-untyped-call]  # protobuf DescriptorPool API is untyped
        pool.AddSerializedFile(_AUDIT_ENVELOPE_FILE_PROTO_BYTES)  # type: ignore[no-untyped-call]  # protobuf DescriptorPool API is untyped
        descriptor = pool.FindMessageTypeByName("spo.audit.AuditEnvelope")  # type: ignore[no-untyped-call]  # protobuf DescriptorPool API is untyped
    except Exception as exc:  # pragma: no cover - defensive fail-closed path
        raise RuntimeError(
            "failed to initialise audit envelope protobuf schema"
        ) from exc
    return message_factory.GetMessageClass(descriptor)


_AuditEnvelope = _audit_envelope_class()


def _canonical_json(payload: Payload) -> str:
    """Return the canonical JSON encoding of a payload."""
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except ValueError as exc:
        raise ValueError("payload must contain only finite JSON numbers") from exc


def _reject_json_constant(value: str) -> None:
    """Raise if the JSON value is a forbidden constant."""
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _encode_varint(value: int) -> bytes:
    """Encode an integer as a protobuf varint."""
    if value < 0:
        raise ValueError("varint cannot encode negative values")
    chunks = bytearray()
    while value >= 0x80:
        chunks.append((value & 0x7F) | 0x80)
        value >>= 7
    chunks.append(value)
    return bytes(chunks)


def _read_varint(fh: BinaryIO) -> int | None:
    """Read a protobuf varint from the byte stream."""
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
    """Return the SHA-256 hash of an audit event."""
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


def _signature_material(
    *,
    audit_mode: str,
    stream_id: str,
    sequence: int,
    event_type: str,
    recorded_at_unix_ns: int,
    source: str,
    previous_hash: str,
    payload_sha256: str,
    event_hash: str,
    schema_version: int,
    key_id: str,
) -> str:
    """Return the canonical bytes signed for an audit event."""
    return _canonical_json(
        {
            "audit_mode": audit_mode,
            "event_hash": event_hash,
            "event_type": event_type,
            "key_id": key_id,
            "payload_sha256": payload_sha256,
            "previous_hash": previous_hash,
            "recorded_at_unix_ns": recorded_at_unix_ns,
            "schema_version": schema_version,
            "sequence": sequence,
            "source": source,
            "stream_id": stream_id,
        }
    )


def _event_type_for_payload(payload: Payload) -> str:
    """Return the audit event type for a payload."""
    if payload.get("header") is True:
        return "header"
    if "event" in payload:
        return str(payload["event"])
    if "step" in payload:
        return "step"
    return "record"


def _message_to_event(message: Message) -> AuditStreamEvent:
    """Convert a protobuf message into an audit event."""
    envelope = cast("Any", message)
    payload_json = str(envelope.payload_json)
    payload = json.loads(payload_json, parse_constant=_reject_json_constant)
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
        signature_algorithm=str(envelope.signature_algorithm),
        signature_key_id=str(envelope.signature_key_id),
        signature=str(envelope.signature),
        audit_mode=str(envelope.audit_mode),
        payload=payload,
    )


class EventStreamWriter:
    """Append length-delimited protobuf audit events to a stream file."""

    def __init__(self, path: str | Path, *, stream_id: str = "spo-audit") -> None:
        self._path = Path(path)
        self._stream_id = _validate_stream_id(stream_id)
        self._sequence = 0
        self._previous_hash = ZERO_HASH
        self._audit_key = os.environ.get("SPO_AUDIT_KEY")
        if self._audit_key == "":
            raise ValueError("SPO_AUDIT_KEY must not be empty")
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
        """Append one payload as a hashed and optionally signed audit event.

        Parameters
        ----------
        payload : Payload
            The event or wire payload.
        event_type : str | None
            Named event type, or ``None``.
        """
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
        signature = ""
        signature_key_id = ""
        signature_algorithm = ""
        audit_mode = "unsigned-development"
        if self._audit_key is not None:
            audit_mode = "hmac-signed"
            signature_algorithm = SIGNATURE_ALGORITHM
            signature_key_id = key_id_for_secret(self._audit_key)
            signature = hmac.new(
                self._audit_key.encode(),
                _signature_material(
                    audit_mode=audit_mode,
                    stream_id=self._stream_id,
                    sequence=self._sequence,
                    event_type=resolved_type,
                    recorded_at_unix_ns=now_ns,
                    source="spo",
                    previous_hash=self._previous_hash,
                    payload_sha256=payload_sha256,
                    event_hash=event_hash,
                    schema_version=SCHEMA_VERSION,
                    key_id=signature_key_id,
                ).encode(),
                hashlib.sha256,
            ).hexdigest()
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
        message.signature_algorithm = signature_algorithm
        message.signature_key_id = signature_key_id
        message.signature = signature
        message.audit_mode = audit_mode
        raw = message.SerializeToString(deterministic=True)
        self._fh.write(_encode_varint(len(raw)))
        self._fh.write(raw)
        self._previous_hash = event_hash

    @property
    def path(self) -> Path:
        """Return the stream file path written by this writer.

        Returns
        -------
        Path
            The protobuf audit stream path.
        """
        return self._path

    def flush(self) -> None:
        """Flush buffered audit bytes without closing the stream handle."""
        self._fh.flush()

    def close(self) -> None:
        """Flush buffered audit bytes and close the underlying stream handle."""
        self._fh.flush()
        self._fh.close()


def _read_events_from_handle(fh: BinaryIO) -> list[AuditStreamEvent]:
    """Read audit events from an open file handle."""
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
    """Read all protobuf events from an SPO audit stream.

    Parameters
    ----------
    path : str | Path
        Filesystem path to the target file.

    Returns
    -------
    list[AuditStreamEvent]
        The decoded audit stream events.
    """
    with Path(path).open("rb") as fh:
        return _read_events_from_handle(fh)


def _validate_poll_interval_s(value: object) -> float:
    """Return the poll interval (s) as a validated positive value, else raise."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("poll_interval_s must be a finite non-negative real")
    parsed = float(value)
    if not parsed >= 0.0 or not parsed < float("inf"):
        raise ValueError("poll_interval_s must be a finite non-negative real")
    return parsed


def _validate_stream_id(value: object) -> str:
    """Return the validated audit stream id, else raise."""
    if not isinstance(value, str) or not value:
        raise ValueError("stream_id must be a non-empty string")
    if len(value) > 128:
        raise ValueError("stream_id must be at most 128 characters")
    if any(ord(char) < 32 for char in value):
        raise ValueError("stream_id must not contain control characters")
    return value


def iter_event_stream(
    path: str | Path,
    *,
    from_start: bool = False,
    poll_interval_s: float = 0.2,
) -> Iterator[AuditStreamEvent]:
    """Yield existing and newly appended stream events in order.

    Parameters
    ----------
    path : str | Path
        Filesystem path to the target file.
    from_start : bool
        Whether to replay from the start of the stream.
    poll_interval_s : float
        Poll interval in seconds.

    Returns
    -------
    Iterator[AuditStreamEvent]
        An iterator over existing and newly appended stream events.

    Raises
    ------
    FileNotFoundError
        If the stream file does not exist.
    """
    poll_interval = _validate_poll_interval_s(poll_interval_s)
    path_obj = Path(path)
    if not from_start and not path_obj.exists():
        raise FileNotFoundError(f"audit event stream path does not exist: {path_obj}")
    offset = 0
    if not from_start and path_obj.exists():
        offset = len(read_event_stream(path_obj))
    while True:
        current = read_event_stream(path_obj) if path_obj.exists() else []
        yield from current[offset:]
        offset = len(current)
        time.sleep(poll_interval)


def tail_event_stream(
    path: str | Path,
    *,
    from_start: bool = False,
    max_events: int | None = None,
    poll_interval_s: float = 0.2,
) -> list[AuditStreamEvent]:
    """Tail a stream file until ``max_events`` decoded events are available.

    Use :func:`iter_event_stream` for unbounded live streaming.

    Parameters
    ----------
    path : str | Path
        Filesystem path to the target file.
    from_start : bool
        Whether to replay from the start of the stream.
    max_events : int | None
        Maximum number of events to read, or ``None``.
    poll_interval_s : float
        Poll interval in seconds.

    Returns
    -------
    list[AuditStreamEvent]
        The decoded events, up to ``max_events``.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    if max_events is None:
        raise ValueError("max_events is required for bounded tail_event_stream")
    if (
        isinstance(max_events, bool)
        or not isinstance(max_events, int)
        or max_events <= 0
    ):
        raise ValueError("max_events must be a positive integer")
    poll_interval = _validate_poll_interval_s(poll_interval_s)
    events: list[AuditStreamEvent] = []
    iterator = iter_event_stream(
        path,
        from_start=from_start,
        poll_interval_s=poll_interval,
    )
    while len(events) < max_events:
        events.append(next(iterator))
    return events


def verify_event_stream_integrity(
    events: list[AuditStreamEvent],
) -> tuple[bool, int]:
    """Verify payload digests, sequence continuity, and event hash chaining.

    Parameters
    ----------
    events : list[AuditStreamEvent]
        The decoded audit stream events.

    Returns
    -------
    tuple[bool, int]
        A ``(ok, count)`` pair: integrity flag and verified event count.
    """
    try:
        audit_keys = audit_verification_keys()
    except ValueError:
        return False, 0
    require_signature = bool(audit_keys)
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
        if require_signature and not _verify_event_signature(
            event,
            audit_keys,
            expected_hash=expected_hash,
            expected_previous_hash=previous_hash,
        ):
            return False, verified
        previous_hash = event.event_hash
        expected_sequence += 1
        verified += 1
    return True, verified


def _verify_event_signature(
    event: AuditStreamEvent,
    audit_keys: dict[str, str],
    *,
    expected_hash: str,
    expected_previous_hash: str,
) -> bool:
    """Return whether an audit event's signature is valid."""
    if event.audit_mode != "hmac-signed":
        return False
    if event.signature_algorithm != SIGNATURE_ALGORITHM:
        return False
    if len(event.signature) != 64:
        return False
    audit_key = audit_keys.get(event.signature_key_id)
    if audit_key is None:
        return False
    if event.signature_key_id != key_id_for_secret(audit_key):
        return False
    expected_signature = hmac.new(
        audit_key.encode(),
        _signature_material(
            audit_mode=event.audit_mode,
            stream_id=event.stream_id,
            sequence=event.sequence,
            event_type=event.event_type,
            recorded_at_unix_ns=event.recorded_at_unix_ns,
            source=event.source,
            previous_hash=expected_previous_hash,
            payload_sha256=event.payload_sha256,
            event_hash=expected_hash,
            schema_version=event.schema_version,
            key_id=event.signature_key_id,
        ).encode(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(event.signature, expected_signature)
