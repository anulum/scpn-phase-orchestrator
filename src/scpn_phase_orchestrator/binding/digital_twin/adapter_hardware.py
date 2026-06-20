# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Digital-twin hardware sync adapter

"""Hardware digital-twin sync adapter validating decoded device payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ._shared import _has_authorization, _require_non_empty
from .contract import (
    _DEFAULT_SYNC_CAPABILITIES,
    DigitalTwinAdapterCompatibility,
    DigitalTwinBindingContract,
    build_digital_twin_adapter_manifest,
)
from .envelope import (
    DigitalTwinSyncEnvelope,
    _envelope_from_record,
    validate_digital_twin_sync_envelope,
)


@dataclass(frozen=True)
class DigitalTwinSyncHardwareResponse:
    """No-I/O response for a hardware digital-twin sync boundary."""

    accepted: bool
    reason: str
    hardware_write_permitted: bool
    frame: dict[str, object]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe hardware adapter response.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe audit mapping of the
            DigitalTwinSyncHardwareResponse fields.
        """
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "hardware_write_permitted": self.hardware_write_permitted,
            "frame": dict(self.frame),
        }


@dataclass
class DigitalTwinSyncHardwareAdapter:
    """No-I/O hardware boundary for digital-twin sync payloads.

    The adapter validates decoded frames from a hardware integration layer. It
    never opens device files, writes registers, toggles GPIO, or applies
    actuation; accepted envelopes are only queued for caller-controlled review.
    """

    contract: DigitalTwinBindingContract
    compatibility: DigitalTwinAdapterCompatibility
    device_ids: tuple[str, ...]
    _queue: list[DigitalTwinSyncEnvelope]

    @classmethod
    def for_contract(
        cls,
        contract: DigitalTwinBindingContract,
        *,
        device_ids: Sequence[str],
        name: str = "hardware-sync",
        sync_capabilities: Sequence[str] = _DEFAULT_SYNC_CAPABILITIES,
        requires_auth: bool = True,
        supports_replay: bool = True,
    ) -> DigitalTwinSyncHardwareAdapter:
        """Create a no-I/O hardware boundary for a digital-twin contract.

        Parameters
        ----------
        contract : DigitalTwinBindingContract
            The digital-twin binding contract the adapter serves.
        device_ids : Sequence[str]
            Identifiers of the hardware devices the boundary serves.
        name : str, optional
            Human-readable adapter name.
        sync_capabilities : Sequence[str], optional
            Sync capabilities the adapter advertises.
        requires_auth : bool, optional
            Whether the adapter boundary requires authentication.
        supports_replay : bool, optional
            Whether the adapter supports replay of past envelopes.

        Returns
        -------
        DigitalTwinSyncHardwareAdapter
            A new no-I/O hardware boundary bound to the contract.

        Raises
        ------
        ValueError
            If ``device_ids`` is empty.
        """
        if not device_ids:
            raise ValueError("hardware device_ids must not be empty")
        checked_device_ids = tuple(device_ids)
        for device_id in checked_device_ids:
            _require_non_empty(device_id, "hardware device_id")
        compatibility = build_digital_twin_adapter_manifest(
            contract,
            name=name,
            transport="hardware",
            sync_capabilities=sync_capabilities,
            supports_replay=supports_replay,
            requires_auth=requires_auth,
            notes="no-I/O hardware boundary",
        )
        return cls(
            contract=contract,
            compatibility=compatibility,
            device_ids=checked_device_ids,
            _queue=[],
        )

    def handle_frame(
        self,
        frame: Mapping[str, object],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> DigitalTwinSyncHardwareResponse:
        """Validate one decoded hardware frame and queue accepted envelopes.

        Parameters
        ----------
        frame : Mapping[str, object]
            The decoded hardware hardware frame.
        headers : Mapping[str, str] or None, optional
            Optional transport headers (e.g. auth tokens).

        Returns
        -------
        DigitalTwinSyncHardwareResponse
            The hardware response; accepted envelopes are queued for :meth:`drain`.
        """
        device_id = frame.get("device_id")
        if not isinstance(device_id, str) or device_id not in self.device_ids:
            return _hardware_response(
                False,
                "device_not_registered",
                {"device_id": device_id, "registered_devices": list(self.device_ids)},
            )
        if frame.get("safety_interlock") is not True:
            return _hardware_response(
                False,
                "safety_interlock_required",
                {"device_id": device_id},
            )
        if not self.compatibility.compatible:
            return _hardware_response(
                False,
                "adapter_incompatible",
                {
                    "device_id": device_id,
                    "reasons": list(self.compatibility.reasons),
                    "contract_hash": self.contract.contract_hash,
                },
            )
        if self.compatibility.manifest.requires_auth and not _has_authorization(
            headers,
        ):
            return _hardware_response(
                False,
                "auth_required",
                {"device_id": device_id, "contract_hash": self.contract.contract_hash},
            )
        value = frame.get("value")
        if not isinstance(value, Mapping):
            return _hardware_response(
                False,
                "invalid_frame_value",
                {"device_id": device_id, "contract_hash": self.contract.contract_hash},
            )
        envelope = _envelope_from_record(dict(value))
        if envelope is None:
            return _hardware_response(
                False,
                "invalid_envelope",
                {"device_id": device_id, "contract_hash": self.contract.contract_hash},
            )
        validation = validate_digital_twin_sync_envelope(self.contract, envelope)
        if not validation.accepted:
            return _hardware_response(
                False,
                validation.reason,
                {
                    "device_id": device_id,
                    "capability": envelope.capability,
                    "sequence": envelope.sequence,
                    "contract_hash": self.contract.contract_hash,
                },
            )
        self._queue.append(envelope)
        return _hardware_response(
            True,
            "accepted",
            {
                "device_id": device_id,
                "capability": envelope.capability,
                "sequence": envelope.sequence,
                "contract_hash": self.contract.contract_hash,
            },
        )

    def drain(self) -> tuple[DigitalTwinSyncEnvelope, ...]:
        """Return accepted hardware envelopes in arrival order and clear the queue.

        Returns
        -------
        tuple[DigitalTwinSyncEnvelope, ...]
            The queued sync envelopes in submission order; the internal queue is left
            empty.
        """
        drained = tuple(self._queue)
        self._queue.clear()
        return drained

    def to_audit_record(self) -> dict[str, object]:
        """Return hardware adapter state without exposing payload contents.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe state of the DigitalTwinSyncHardwareAdapter (queue
            counters and status); no network surface or payload contents are exposed.
        """
        return {
            "contract_hash": self.contract.contract_hash,
            "manifest": self.compatibility.manifest.to_audit_record(),
            "compatible": self.compatibility.compatible,
            "device_ids": list(self.device_ids),
            "queued_count": len(self._queue),
            "queued_sequences": [envelope.sequence for envelope in self._queue],
            "hardware_write_permitted": False,
        }


def _hardware_response(
    accepted: bool,
    reason: str,
    frame: dict[str, object],
) -> DigitalTwinSyncHardwareResponse:
    return DigitalTwinSyncHardwareResponse(
        accepted=accepted,
        reason=reason,
        hardware_write_permitted=False,
        frame=frame,
    )
