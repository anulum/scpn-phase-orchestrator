# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Digital-twin sync envelope validation and JSONL

"""Sync envelope construction, transport validation, and JSONL persistence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ._shared import _require_non_empty
from .contract import DigitalTwinBindingContract, DigitalTwinSyncCapability


@dataclass(frozen=True)
class DigitalTwinSyncEnvelope:
    """Transport-neutral live-sync payload envelope for digital twins."""

    contract_hash: str
    capability: str
    direction: str
    sequence: int
    payload: dict[str, object]

    def __post_init__(self) -> None:
        _require_non_empty(self.contract_hash, "contract_hash")
        _require_non_empty(self.capability, "capability")
        _require_non_empty(self.direction, "direction")
        if self.sequence < 0:
            raise ValueError("sequence must be >= 0")

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe sync envelope.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe audit mapping of the DigitalTwinSyncEnvelope
            fields.
        """
        return {
            "contract_hash": self.contract_hash,
            "capability": self.capability,
            "direction": self.direction,
            "sequence": self.sequence,
            "payload": dict(self.payload),
        }

    def to_json(self) -> str:
        """Serialise the envelope with deterministic key ordering.

        Returns
        -------
        str
            The envelope serialised as a JSON string with deterministically sorted keys.
        """
        return json.dumps(self.to_audit_record(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class DigitalTwinTransportValidation:
    """Validation result for one digital-twin sync envelope."""

    accepted: bool
    reason: str
    envelope: DigitalTwinSyncEnvelope

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe validation record.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe audit mapping of the DigitalTwinTransportValidation
            fields.
        """
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "envelope": self.envelope.to_audit_record(),
        }


@dataclass(frozen=True)
class DigitalTwinSyncJsonlReport:
    """JSONL file-adapter replay report for digital-twin sync envelopes."""

    path: str
    written: int
    accepted: tuple[DigitalTwinTransportValidation, ...]
    rejected: tuple[dict[str, object], ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe file-adapter report.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe audit mapping of the DigitalTwinSyncJsonlReport
            fields.
        """
        return {
            "path": self.path,
            "written": self.written,
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
            "accepted": [validation.to_audit_record() for validation in self.accepted],
            "rejected": list(self.rejected),
        }


def build_digital_twin_sync_envelope(
    contract: DigitalTwinBindingContract,
    *,
    capability: str,
    direction: str,
    sequence: int,
    payload: dict[str, object],
) -> DigitalTwinSyncEnvelope:
    """Build a transport-neutral sync payload envelope for a contract.

    This helper does not send data. It creates the deterministic envelope that
    REST, gRPC, Kafka, file, or hardware adapters can validate before handing a
    payload to the runtime.

    Parameters
    ----------
    contract : DigitalTwinBindingContract
        The contract the envelope conforms to.
    capability : str
        The sync capability the envelope exercises.
    direction : str
        Sync direction (e.g. ``inbound``/``outbound``).
    sequence : int
        Monotonic envelope sequence number.
    payload : dict[str, object]
        The deterministic payload to wrap.

    Returns
    -------
    DigitalTwinSyncEnvelope
        A validated, transport-neutral sync envelope.
    """
    return DigitalTwinSyncEnvelope(
        contract_hash=contract.contract_hash,
        capability=capability,
        direction=direction,
        sequence=sequence,
        payload=payload,
    )


def write_digital_twin_sync_jsonl(
    path: str | Path,
    envelopes: Sequence[DigitalTwinSyncEnvelope],
) -> DigitalTwinSyncJsonlReport:
    """Write sync envelopes to deterministic JSONL for offline replay.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination JSONL file path.
    envelopes : Sequence[DigitalTwinSyncEnvelope]
        The envelopes to serialise, in order.

    Returns
    -------
    DigitalTwinSyncJsonlReport
        A report of the written file and envelope count.
    """
    target = Path(path)
    lines = [envelope.to_json() for envelope in envelopes]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return DigitalTwinSyncJsonlReport(
        path=str(target),
        written=len(lines),
        accepted=(),
        rejected=(),
    )


def read_digital_twin_sync_jsonl(
    contract: DigitalTwinBindingContract,
    path: str | Path,
) -> DigitalTwinSyncJsonlReport:
    """Read JSONL sync envelopes and validate them against a contract.

    Parameters
    ----------
    contract : DigitalTwinBindingContract
        The contract to validate envelopes against.
    path : str or pathlib.Path
        JSONL file to read.

    Returns
    -------
    DigitalTwinSyncJsonlReport
        A report of read, accepted, and rejected envelopes.
    """
    source = Path(path)
    accepted: list[DigitalTwinTransportValidation] = []
    rejected: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            raw_record = json.loads(raw_line)
        except json.JSONDecodeError:
            rejected.append(_jsonl_rejection(line_number, "malformed_json"))
            continue
        envelope = _envelope_from_record(raw_record)
        if envelope is None:
            rejected.append(_jsonl_rejection(line_number, "invalid_envelope"))
            continue
        validation = validate_digital_twin_sync_envelope(contract, envelope)
        if validation.accepted:
            accepted.append(validation)
        else:
            rejected.append(_jsonl_rejection(line_number, validation.reason))
    return DigitalTwinSyncJsonlReport(
        path=str(source),
        written=0,
        accepted=tuple(accepted),
        rejected=tuple(rejected),
    )


def validate_digital_twin_sync_envelope(
    contract: DigitalTwinBindingContract,
    envelope: DigitalTwinSyncEnvelope,
) -> DigitalTwinTransportValidation:
    """Validate a digital-twin sync envelope against a binding contract.

    Parameters
    ----------
    contract : DigitalTwinBindingContract
        The binding contract to validate against.
    envelope : DigitalTwinSyncEnvelope
        The sync envelope to validate.

    Returns
    -------
    DigitalTwinTransportValidation
        The validation result (accepted, or rejected with reasons).
    """
    if envelope.contract_hash != contract.contract_hash:
        return _transport_validation(False, "contract_hash_mismatch", envelope)
    capability = _find_capability(contract, envelope.capability)
    if capability is None:
        return _transport_validation(False, "capability_not_declared", envelope)
    if not _direction_allowed(
        declared=capability.direction,
        observed=envelope.direction,
    ):
        return _transport_validation(False, "direction_not_allowed", envelope)
    if not envelope.payload:
        return _transport_validation(False, "payload_empty", envelope)
    return _transport_validation(True, "accepted", envelope)


def _find_capability(
    contract: DigitalTwinBindingContract,
    name: str,
) -> DigitalTwinSyncCapability | None:
    """Return the capability matching the name, else raise."""
    for capability in contract.sync_capabilities:
        if capability.name == name:
            return capability
    return None


def _envelope_from_record(record: object) -> DigitalTwinSyncEnvelope | None:
    """Build a digital-twin envelope from a record."""
    if not isinstance(record, dict):
        return None
    contract_hash = record.get("contract_hash")
    capability = record.get("capability")
    direction = record.get("direction")
    sequence = record.get("sequence")
    payload = record.get("payload")
    if not isinstance(contract_hash, str):
        return None
    if not isinstance(capability, str):
        return None
    if not isinstance(direction, str):
        return None
    if not isinstance(sequence, int) or isinstance(sequence, bool):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return DigitalTwinSyncEnvelope(
            contract_hash=contract_hash,
            capability=capability,
            direction=direction,
            sequence=sequence,
            payload=dict(payload),
        )
    except ValueError:
        return None


def _jsonl_rejection(line_number: int, reason: str) -> dict[str, object]:
    """Build a JSONL rejection record."""
    return {
        "line_number": line_number,
        "reason": reason,
    }


def _direction_allowed(*, declared: str, observed: str) -> bool:
    """Return whether a sync direction is allowed."""
    return declared == "bidirectional" or declared == observed


def _transport_validation(
    accepted: bool,
    reason: str,
    envelope: DigitalTwinSyncEnvelope,
) -> DigitalTwinTransportValidation:
    """Validate the transport configuration, else raise."""
    return DigitalTwinTransportValidation(
        accepted=accepted,
        reason=reason,
        envelope=envelope,
    )
