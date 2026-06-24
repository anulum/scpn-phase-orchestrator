# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Digital-twin gRPC sync adapter

"""gRPC digital-twin sync adapter validating decoded request/response payloads."""

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
class DigitalTwinSyncGrpcResponse:
    """gRPC-style response for a digital-twin sync boundary."""

    status_code: str
    accepted: bool
    reason: str
    message: dict[str, object]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe gRPC adapter response.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe audit mapping of the DigitalTwinSyncGrpcResponse
            fields.
        """
        return {
            "status_code": self.status_code,
            "accepted": self.accepted,
            "reason": self.reason,
            "message": dict(self.message),
        }


@dataclass
class DigitalTwinSyncGrpcAdapter:
    """Dependency-free gRPC boundary for digital-twin sync payloads.

    The adapter does not start a gRPC server or import generated protobuf
    classes. A real servicer can pass decoded protobuf fields into
    :meth:`handle_unary`; this boundary then applies the same contract checks
    as other transports before queuing accepted envelopes.
    """

    contract: DigitalTwinBindingContract
    compatibility: DigitalTwinAdapterCompatibility
    _queue: list[DigitalTwinSyncEnvelope]

    @classmethod
    def for_contract(
        cls,
        contract: DigitalTwinBindingContract,
        *,
        name: str = "grpc-sync",
        sync_capabilities: Sequence[str] = _DEFAULT_SYNC_CAPABILITIES,
        requires_auth: bool = True,
        supports_replay: bool = False,
    ) -> DigitalTwinSyncGrpcAdapter:
        """Create a gRPC adapter boundary for a digital-twin contract.

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
        DigitalTwinSyncGrpcAdapter
            A new gRPC adapter boundary bound to the contract.
        """
        compatibility = build_digital_twin_adapter_manifest(
            contract,
            name=name,
            transport="grpc",
            sync_capabilities=sync_capabilities,
            supports_replay=supports_replay,
            requires_auth=requires_auth,
            notes="dependency-free gRPC boundary",
        )
        return cls(contract=contract, compatibility=compatibility, _queue=[])

    def handle_unary(
        self,
        request: Mapping[str, object],
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> DigitalTwinSyncGrpcResponse:
        """Validate one unary gRPC request and queue accepted envelopes.

        Parameters
        ----------
        request : Mapping[str, object]
            The decoded unary gRPC request body.
        metadata : Mapping[str, str] or None, optional
            Optional request metadata (e.g. auth tokens).

        Returns
        -------
        DigitalTwinSyncGrpcResponse
            The response; accepted envelopes are queued for :meth:`drain`.
        """
        if not self.compatibility.compatible:
            return _grpc_response(
                "FAILED_PRECONDITION",
                False,
                "adapter_incompatible",
                {
                    "reasons": list(self.compatibility.reasons),
                    "contract_hash": self.contract.contract_hash,
                },
            )
        if self.compatibility.manifest.requires_auth and not _has_authorization(
            metadata,
        ):
            return _grpc_response(
                "UNAUTHENTICATED",
                False,
                "auth_required",
                {"contract_hash": self.contract.contract_hash},
            )
        envelope = _envelope_from_record(dict(request))
        if envelope is None:
            return _grpc_response(
                "INVALID_ARGUMENT",
                False,
                "invalid_envelope",
                {"contract_hash": self.contract.contract_hash},
            )
        validation = validate_digital_twin_sync_envelope(self.contract, envelope)
        if not validation.accepted:
            return _grpc_response(
                "FAILED_PRECONDITION",
                False,
                validation.reason,
                {
                    "capability": envelope.capability,
                    "sequence": envelope.sequence,
                    "contract_hash": self.contract.contract_hash,
                },
            )
        self._queue.append(envelope)
        return _grpc_response(
            "OK",
            True,
            "accepted",
            {
                "capability": envelope.capability,
                "sequence": envelope.sequence,
                "contract_hash": self.contract.contract_hash,
            },
        )

    def drain(self) -> tuple[DigitalTwinSyncEnvelope, ...]:
        """Return accepted gRPC envelopes in arrival order and clear the queue.

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
        """Return gRPC adapter state without exposing payload contents.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe state of the DigitalTwinSyncGrpcAdapter (queue
            counters and status); no network surface or payload contents are exposed.
        """
        return {
            "contract_hash": self.contract.contract_hash,
            "manifest": self.compatibility.manifest.to_audit_record(),
            "compatible": self.compatibility.compatible,
            "queued_count": len(self._queue),
            "queued_sequences": [envelope.sequence for envelope in self._queue],
        }


def _grpc_response(
    status_code: str,
    accepted: bool,
    reason: str,
    message: dict[str, object],
) -> DigitalTwinSyncGrpcResponse:
    """Build the gRPC response for a digital-twin sync."""
    return DigitalTwinSyncGrpcResponse(
        status_code=status_code,
        accepted=accepted,
        reason=reason,
        message=message,
    )
