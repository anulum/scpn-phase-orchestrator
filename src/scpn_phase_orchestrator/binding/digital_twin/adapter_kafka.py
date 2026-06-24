# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Digital-twin Kafka sync adapter

"""Kafka digital-twin sync adapter validating decoded message payloads."""

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
class DigitalTwinSyncKafkaResponse:
    """Broker-style response for a Kafka digital-twin sync boundary."""

    accepted: bool
    reason: str
    retryable: bool
    message: dict[str, object]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe Kafka adapter response.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe audit mapping of the DigitalTwinSyncKafkaResponse
            fields.
        """
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "retryable": self.retryable,
            "message": dict(self.message),
        }


@dataclass
class DigitalTwinSyncKafkaAdapter:
    """Dependency-free Kafka boundary for digital-twin sync payloads.

    The adapter expects a broker consumer to pass a decoded message dictionary.
    It does not import Kafka clients, open sockets, or commit offsets. Accepted
    envelopes are queued for caller-controlled runtime handoff.
    """

    contract: DigitalTwinBindingContract
    compatibility: DigitalTwinAdapterCompatibility
    topic: str
    _queue: list[DigitalTwinSyncEnvelope]

    @classmethod
    def for_contract(
        cls,
        contract: DigitalTwinBindingContract,
        *,
        topic: str = "spo.digital_twin.sync",
        name: str = "kafka-sync",
        sync_capabilities: Sequence[str] = _DEFAULT_SYNC_CAPABILITIES,
        requires_auth: bool = True,
        supports_replay: bool = True,
    ) -> DigitalTwinSyncKafkaAdapter:
        """Create a Kafka message-boundary adapter for a digital-twin contract.

        Parameters
        ----------
        contract : DigitalTwinBindingContract
            The digital-twin binding contract the adapter serves.
        topic : str, optional
            Kafka topic the adapter binds to.
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
        DigitalTwinSyncKafkaAdapter
            A new Kafka message-boundary adapter bound to the contract.
        """
        _require_non_empty(topic, "kafka topic")
        compatibility = build_digital_twin_adapter_manifest(
            contract,
            name=name,
            transport="kafka",
            sync_capabilities=sync_capabilities,
            supports_replay=supports_replay,
            requires_auth=requires_auth,
            notes="dependency-free Kafka boundary",
        )
        return cls(
            contract=contract,
            compatibility=compatibility,
            topic=topic,
            _queue=[],
        )

    def handle_message(
        self,
        message: Mapping[str, object],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> DigitalTwinSyncKafkaResponse:
        """Validate one decoded Kafka message and queue accepted envelopes.

        Parameters
        ----------
        message : Mapping[str, object]
            The decoded Kafka message body.
        headers : Mapping[str, str] or None, optional
            Optional transport headers (e.g. auth tokens).

        Returns
        -------
        DigitalTwinSyncKafkaResponse
            The Kafka response; accepted envelopes are queued for :meth:`drain`.
        """
        message_topic = message.get("topic", self.topic)
        if not isinstance(message_topic, str) or message_topic != self.topic:
            return _kafka_response(
                False,
                "topic_mismatch",
                False,
                {"expected_topic": self.topic, "observed_topic": message_topic},
            )
        if not self.compatibility.compatible:
            return _kafka_response(
                False,
                "adapter_incompatible",
                True,
                {
                    "reasons": list(self.compatibility.reasons),
                    "contract_hash": self.contract.contract_hash,
                },
            )
        if self.compatibility.manifest.requires_auth and not _has_authorization(
            headers,
        ):
            return _kafka_response(
                False,
                "auth_required",
                True,
                {"contract_hash": self.contract.contract_hash},
            )
        value = message.get("value")
        if not isinstance(value, Mapping):
            return _kafka_response(
                False,
                "invalid_message_value",
                False,
                {"contract_hash": self.contract.contract_hash},
            )
        envelope = _envelope_from_record(dict(value))
        if envelope is None:
            return _kafka_response(
                False,
                "invalid_envelope",
                False,
                {"contract_hash": self.contract.contract_hash},
            )
        validation = validate_digital_twin_sync_envelope(self.contract, envelope)
        if not validation.accepted:
            return _kafka_response(
                False,
                validation.reason,
                False,
                {
                    "capability": envelope.capability,
                    "sequence": envelope.sequence,
                    "contract_hash": self.contract.contract_hash,
                },
            )
        self._queue.append(envelope)
        return _kafka_response(
            True,
            "accepted",
            False,
            {
                "topic": self.topic,
                "capability": envelope.capability,
                "sequence": envelope.sequence,
                "contract_hash": self.contract.contract_hash,
            },
        )

    def drain(self) -> tuple[DigitalTwinSyncEnvelope, ...]:
        """Return accepted Kafka envelopes in arrival order and clear the queue.

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
        """Return Kafka adapter state without exposing payload contents.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe state of the DigitalTwinSyncKafkaAdapter (queue
            counters and status); no network surface or payload contents are exposed.
        """
        return {
            "contract_hash": self.contract.contract_hash,
            "manifest": self.compatibility.manifest.to_audit_record(),
            "compatible": self.compatibility.compatible,
            "topic": self.topic,
            "queued_count": len(self._queue),
            "queued_sequences": [envelope.sequence for envelope in self._queue],
        }


def _kafka_response(
    accepted: bool,
    reason: str,
    retryable: bool,
    message: dict[str, object],
) -> DigitalTwinSyncKafkaResponse:
    """Build the Kafka response for a digital-twin sync."""
    return DigitalTwinSyncKafkaResponse(
        accepted=accepted,
        reason=reason,
        retryable=retryable,
        message=message,
    )
