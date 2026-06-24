# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Digital-twin REST sync adapter

"""REST digital-twin sync adapter validating decoded request/response payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ._shared import _has_authorization
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
class DigitalTwinSyncRestResponse:
    """HTTP-style response for a REST digital-twin sync boundary."""

    status_code: int
    accepted: bool
    reason: str
    body: dict[str, object]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe REST adapter response.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe audit mapping of the DigitalTwinSyncRestResponse
            fields.
        """
        return {
            "status_code": self.status_code,
            "accepted": self.accepted,
            "reason": self.reason,
            "body": dict(self.body),
        }


@dataclass
class DigitalTwinSyncRestAdapter:
    """Dependency-free REST boundary for digital-twin sync payloads.

    The adapter deliberately does not open sockets. Web frameworks can call
    :meth:`handle_post` from a route handler after parsing request JSON and
    headers; the adapter then enforces manifest compatibility, authentication
    posture, envelope shape, and contract validation before queuing payloads.
    """

    contract: DigitalTwinBindingContract
    compatibility: DigitalTwinAdapterCompatibility
    _queue: list[DigitalTwinSyncEnvelope]

    @classmethod
    def for_contract(
        cls,
        contract: DigitalTwinBindingContract,
        *,
        name: str = "rest-sync",
        sync_capabilities: Sequence[str] = _DEFAULT_SYNC_CAPABILITIES,
        requires_auth: bool = True,
        supports_replay: bool = False,
    ) -> DigitalTwinSyncRestAdapter:
        """Create a REST adapter boundary for a digital-twin contract.

        Parameters
        ----------
        contract : DigitalTwinBindingContract
            The digital-twin binding contract the adapter serves.
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
        DigitalTwinSyncRestAdapter
            A new REST adapter boundary bound to the contract.
        """
        compatibility = build_digital_twin_adapter_manifest(
            contract,
            name=name,
            transport="rest",
            sync_capabilities=sync_capabilities,
            supports_replay=supports_replay,
            requires_auth=requires_auth,
            notes="dependency-free REST boundary",
        )
        return cls(contract=contract, compatibility=compatibility, _queue=[])

    def handle_post(
        self,
        body: Mapping[str, object],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> DigitalTwinSyncRestResponse:
        """Validate one HTTP POST body and queue accepted sync envelopes.

        Parameters
        ----------
        body : Mapping[str, object]
            The decoded REST HTTP POST body.
        headers : Mapping[str, str] or None, optional
            Optional transport headers (e.g. auth tokens).

        Returns
        -------
        DigitalTwinSyncRestResponse
            The REST response; accepted envelopes are queued for :meth:`drain`.
        """
        if not self.compatibility.compatible:
            return _rest_response(
                503,
                False,
                "adapter_incompatible",
                {
                    "reasons": list(self.compatibility.reasons),
                    "contract_hash": self.contract.contract_hash,
                },
            )
        if self.compatibility.manifest.requires_auth and not _has_authorization(
            headers,
        ):
            return _rest_response(
                401,
                False,
                "auth_required",
                {"contract_hash": self.contract.contract_hash},
            )
        envelope = _envelope_from_record(dict(body))
        if envelope is None:
            return _rest_response(
                400,
                False,
                "invalid_envelope",
                {"contract_hash": self.contract.contract_hash},
            )
        validation = validate_digital_twin_sync_envelope(self.contract, envelope)
        if not validation.accepted:
            return _rest_response(
                422,
                False,
                validation.reason,
                {
                    "capability": envelope.capability,
                    "sequence": envelope.sequence,
                    "contract_hash": self.contract.contract_hash,
                },
            )
        self._queue.append(envelope)
        return _rest_response(
            202,
            True,
            "accepted",
            {
                "capability": envelope.capability,
                "sequence": envelope.sequence,
                "contract_hash": self.contract.contract_hash,
            },
        )

    def drain(self) -> tuple[DigitalTwinSyncEnvelope, ...]:
        """Return accepted REST envelopes in arrival order and clear the queue.

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
        """Return REST adapter state without exposing payload contents.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe state of the DigitalTwinSyncRestAdapter (queue
            counters and status); no network surface or payload contents are exposed.
        """
        return {
            "contract_hash": self.contract.contract_hash,
            "manifest": self.compatibility.manifest.to_audit_record(),
            "compatible": self.compatibility.compatible,
            "queued_count": len(self._queue),
            "queued_sequences": [envelope.sequence for envelope in self._queue],
        }


def _rest_response(
    status_code: int,
    accepted: bool,
    reason: str,
    body: dict[str, object],
) -> DigitalTwinSyncRestResponse:
    """Build the REST response for a digital-twin sync."""
    return DigitalTwinSyncRestResponse(
        status_code=status_code,
        accepted=accepted,
        reason=reason,
        body=body,
    )
