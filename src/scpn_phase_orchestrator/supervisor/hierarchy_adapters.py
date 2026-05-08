# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hierarchy adapter boundaries

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from scpn_phase_orchestrator.supervisor.hierarchy import (
    HierarchySyncEnvelope,
    HierarchySyncLedger,
    HierarchyTransportRuntime,
)

__all__ = [
    "HierarchyAdapterResult",
    "handle_hierarchy_frame",
    "handle_hierarchy_rest_payload",
    "replay_hierarchy_jsonl",
]

_REST_PAYLOAD_KEYS = frozenset({"envelope", "envelopes"})
_FRAME_KEYS = frozenset({"kind", "type", "payload"})
_SINGLE_FRAME_KIND = "hierarchy_sync"
_BATCH_FRAME_KIND = "hierarchy_sync_batch"


@dataclass(frozen=True)
class HierarchyAdapterResult:
    """Audit-safe result returned by decoded hierarchy adapter boundaries."""

    boundary: str
    ledger: HierarchySyncLedger
    watermarks: Mapping[str, int]
    frame_kind: str | None = None
    status: str = "accepted"

    @property
    def accepted_count(self) -> int:
        """Return the number of envelopes accepted by the runtime."""
        return len(self.ledger.accepted)

    @property
    def rejected_count(self) -> int:
        """Return the number of envelopes rejected by the runtime."""
        return len(self.ledger.rejected)

    def to_audit_record(self) -> dict[str, object]:
        """Return a deterministic JSON-safe adapter audit payload."""
        plan = self.ledger.plan
        record: dict[str, object] = {
            "boundary": self.boundary,
            "status": self.status,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "watermarks": {
                source: int(self.watermarks[source])
                for source in sorted(self.watermarks)
            },
            "parent_plan": {
                "R": float(plan.parent_R),
                "psi": float(plan.parent_psi),
                "regime_id": plan.parent_state.regime_id,
                "layer_count": len(plan.parent_state.layers),
            },
            "ledger": self.ledger.to_audit_record(),
        }
        if self.frame_kind is not None:
            record["frame_kind"] = self.frame_kind
        return record


def replay_hierarchy_jsonl(
    lines: Iterable[str | Mapping[str, object] | HierarchySyncEnvelope],
    *,
    runtime: HierarchyTransportRuntime | None = None,
) -> HierarchyAdapterResult:
    """Replay decoded or JSONL hierarchy records through a socket-free runtime."""
    active_runtime = runtime or HierarchyTransportRuntime()
    records = tuple(
        _load_jsonl_record(line, index) for index, line in enumerate(lines, 1)
    )
    ledger = active_runtime.ingest(records)
    return _adapter_result("jsonl_replay", ledger, active_runtime)


def handle_hierarchy_rest_payload(
    payload: Mapping[str, object],
    *,
    headers: Mapping[str, object],
    runtime: HierarchyTransportRuntime | None = None,
) -> HierarchyAdapterResult:
    """Handle a decoded REST request payload without owning an HTTP server."""
    _require_json_content_type(headers)
    records = _records_from_payload(payload, location="REST payload")
    active_runtime = runtime or HierarchyTransportRuntime()
    ledger = active_runtime.ingest(records)
    return _adapter_result("rest_boundary", ledger, active_runtime)


def handle_hierarchy_frame(
    frame: Mapping[str, object],
    *,
    runtime: HierarchyTransportRuntime | None = None,
) -> HierarchyAdapterResult:
    """Handle a decoded WebSocket-style frame without owning a socket."""
    if not isinstance(frame, Mapping):
        raise ValueError("frame must be a decoded mapping")
    _reject_unknown_keys(frame, allowed=_FRAME_KEYS, location="frame")
    frame_kind = _frame_kind(frame)
    payload = frame.get("payload")
    if payload is None:
        raise ValueError("payload must be provided")
    records: tuple[Mapping[str, object] | HierarchySyncEnvelope, ...]
    if frame_kind == _SINGLE_FRAME_KIND:
        records = (_require_record(payload, "frame payload"),)
    elif frame_kind == _BATCH_FRAME_KIND:
        records = _records_from_frame_payload(payload)
    else:
        raise ValueError("frame kind must be hierarchy_sync or hierarchy_sync_batch")
    active_runtime = runtime or HierarchyTransportRuntime()
    ledger = active_runtime.ingest(records)
    return _adapter_result(
        "websocket_frame",
        ledger,
        active_runtime,
        frame_kind=frame_kind,
    )


def _adapter_result(
    boundary: str,
    ledger: HierarchySyncLedger,
    runtime: HierarchyTransportRuntime,
    *,
    frame_kind: str | None = None,
) -> HierarchyAdapterResult:
    return HierarchyAdapterResult(
        boundary=boundary,
        ledger=ledger,
        watermarks=runtime.previous_sequences,
        frame_kind=frame_kind,
    )


def _load_jsonl_record(
    line: str | Mapping[str, object] | HierarchySyncEnvelope,
    index: int,
) -> Mapping[str, object] | HierarchySyncEnvelope:
    if isinstance(line, HierarchySyncEnvelope):
        return line
    if isinstance(line, Mapping):
        return line
    if not isinstance(line, str):
        raise ValueError(f"JSONL line {index} must be a string or decoded mapping")
    text = line.strip()
    if not text:
        raise ValueError(f"JSONL line {index} must not be blank")
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSONL line {index} must be valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"JSONL line {index} must decode to a mapping")
    return decoded


def _require_json_content_type(headers: Mapping[str, object]) -> None:
    if not isinstance(headers, Mapping):
        raise ValueError("headers must be a decoded mapping")
    content_type = None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == "content-type":
            content_type = value
            break
    if not isinstance(content_type, str):
        raise ValueError("content-type must be application/json")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        raise ValueError("content-type must be application/json")


def _records_from_payload(
    payload: Mapping[str, object],
    *,
    location: str,
) -> tuple[Mapping[str, object] | HierarchySyncEnvelope, ...]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{location} must be a decoded mapping")
    _reject_unknown_keys(payload, allowed=_REST_PAYLOAD_KEYS, location=location)
    has_single = "envelope" in payload
    has_batch = "envelopes" in payload
    if has_single == has_batch:
        raise ValueError(
            f"{location} must contain exactly one of envelope or envelopes"
        )
    if has_single:
        return (_require_record(payload["envelope"], f"{location}.envelope"),)
    return _require_record_sequence(payload["envelopes"], f"{location}.envelopes")


def _records_from_frame_payload(
    payload: object,
) -> tuple[Mapping[str, object] | HierarchySyncEnvelope, ...]:
    if isinstance(payload, Mapping) and "envelopes" in payload:
        _reject_unknown_keys(
            payload,
            allowed=frozenset({"envelopes"}),
            location="frame payload",
        )
        return _require_record_sequence(payload["envelopes"], "frame payload.envelopes")
    return _require_record_sequence(payload, "frame payload")


def _require_record(
    value: object,
    location: str,
) -> Mapping[str, object] | HierarchySyncEnvelope:
    if isinstance(value, HierarchySyncEnvelope):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a decoded envelope mapping")
    return value


def _require_record_sequence(
    value: object,
    location: str,
) -> tuple[Mapping[str, object] | HierarchySyncEnvelope, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{location} must be a sequence of decoded envelope mappings")
    records = tuple(
        _require_record(item, f"{location}[{index}]")
        for index, item in enumerate(value)
    )
    if not records:
        raise ValueError(f"{location} must contain at least one envelope")
    return records


def _frame_kind(frame: Mapping[str, object]) -> str:
    if "kind" in frame and "type" in frame:
        raise ValueError("frame must not contain both kind and type")
    value = frame.get("kind", frame.get("type"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("frame kind must be hierarchy_sync or hierarchy_sync_batch")
    return value.strip()


def _reject_unknown_keys(
    record: Mapping[str, object],
    *,
    allowed: frozenset[str],
    location: str,
) -> None:
    unknown = set(record).difference(allowed)
    if unknown:
        keys = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(f"{location} contains unknown keys: {keys}")
