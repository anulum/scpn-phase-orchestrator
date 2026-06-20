# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Digital-twin in-memory sync adapter

"""In-memory digital-twin sync adapter for tests and local replay."""

from __future__ import annotations

from dataclasses import dataclass

from .contract import DigitalTwinBindingContract
from .envelope import (
    DigitalTwinSyncEnvelope,
    DigitalTwinTransportValidation,
    validate_digital_twin_sync_envelope,
)


@dataclass
class DigitalTwinSyncMemoryAdapter:
    """In-memory reference adapter for validated digital-twin sync payloads."""

    contract: DigitalTwinBindingContract
    _queue: list[DigitalTwinSyncEnvelope]

    @classmethod
    def for_contract(
        cls,
        contract: DigitalTwinBindingContract,
    ) -> DigitalTwinSyncMemoryAdapter:
        """Create an empty adapter for a digital-twin binding contract.

        Parameters
        ----------
        contract : DigitalTwinBindingContract
            The digital-twin binding contract the adapter serves.

        Returns
        -------
        DigitalTwinSyncMemoryAdapter
            An empty in-memory adapter bound to the contract.
        """
        return cls(contract=contract, _queue=[])

    def submit(
        self,
        envelope: DigitalTwinSyncEnvelope,
    ) -> DigitalTwinTransportValidation:
        """Validate and queue one envelope when accepted.

        Parameters
        ----------
        envelope : DigitalTwinSyncEnvelope
            The sync envelope to validate and queue.

        Returns
        -------
        DigitalTwinTransportValidation
            The validation result; the envelope is queued only when accepted.
        """
        validation = validate_digital_twin_sync_envelope(self.contract, envelope)
        if validation.accepted:
            self._queue.append(envelope)
        return validation

    def drain(self) -> tuple[DigitalTwinSyncEnvelope, ...]:
        """Return queued envelopes in submission order and clear the queue.

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
        """Return adapter state without exposing any network surface.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe state of the DigitalTwinSyncMemoryAdapter (queue
            counters and status); no network surface or payload contents are exposed.
        """
        return {
            "contract_hash": self.contract.contract_hash,
            "queued_count": len(self._queue),
            "queued_sequences": [envelope.sequence for envelope in self._queue],
        }
