# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hierarchy sync envelope transport and ledger

"""Sync envelope build, load, ingest, transport runtime, and conflict ledger."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .boundary import (
    _AUDIT_SCOPE_REDUCED_SUMMARIES,
    _DEFAULT_HIERARCHY_SYNC_PROTOCOL,
    _REGIME_NOMINAL,
    ChildSupervisorSummary,
    HierarchySyncEnvelope,
    _canonical_hierarchy_sync_envelope,
    _dummy_summary,
    _duplicate_sequence_conflict_sources,
    _is_forbidden_hierarchy_key,
    _normalise_previous_sequences,
    _rejection,
    _rejection_sort_key,
    _require_float,
    _require_integer,
    _require_non_empty,
    _require_unit_interval,
    _validate_envelope_reduced_only,
    _validate_metadata_value,
    _validate_plan_inputs,
)
from .plan import HierarchicalOrchestrationPlan, build_hierarchical_orchestration_plan

_HIERARCHY_SYNC_ENVELOPE_KEYS = frozenset(
    {
        "monotonic_time_s",
        "protocol_version",
        "sequence",
        "source_node",
        "summary",
    }
)


_HIERARCHY_SYNC_SUMMARY_KEYS = frozenset(
    {
        "R",
        "channel",
        "confidence",
        "metadata",
        "name",
        "psi",
        "regime",
        "weighted_R",
    }
)


@dataclass(frozen=True)
class HierarchySyncLedger:
    """Parent-side ingestion result for sync envelopes."""

    accepted: tuple[HierarchySyncEnvelope, ...]
    rejected: tuple[dict[str, object], ...]
    plan: HierarchicalOrchestrationPlan

    def to_audit_record(self) -> dict[str, object]:
        """Return a serialisable sync-ingestion audit payload.

        Returns
        -------
        dict[str, object]
            Return a serialisable sync-ingestion audit payload.
        """
        return {
            "accepted": [envelope.to_audit_record() for envelope in self.accepted],
            "rejected": list(self.rejected),
            "plan": self.plan.to_audit_record(),
        }


class HierarchyTransportRuntime:
    """Socket-free runtime state for hierarchy transport adapters."""

    def __init__(
        self,
        *,
        previous_sequences: Mapping[str, int] | None = None,
        hierarchy: str = "edge_cloud_summary_sync",
        degraded_threshold: float = 0.65,
        critical_threshold: float = 0.35,
        min_confidence: float = 0.5,
        protocol_version: str = _DEFAULT_HIERARCHY_SYNC_PROTOCOL,
    ) -> None:
        _require_non_empty(hierarchy, "hierarchy")
        _require_non_empty(protocol_version, "protocol_version")
        _require_unit_interval(degraded_threshold, "degraded_threshold")
        _require_unit_interval(critical_threshold, "critical_threshold")
        _require_unit_interval(min_confidence, "min_confidence")
        if critical_threshold > degraded_threshold:
            raise ValueError("critical_threshold must be <= degraded_threshold")
        self._previous_sequences = _normalise_previous_sequences(previous_sequences)
        self._hierarchy = hierarchy
        self._degraded_threshold = degraded_threshold
        self._critical_threshold = critical_threshold
        self._min_confidence = min_confidence
        self._protocol_version = protocol_version

    @property
    def previous_sequences(self) -> dict[str, int]:
        """Return the accepted per-source sequence watermarks.

        Returns
        -------
        dict[str, int]
            Return the accepted per-source sequence watermarks.
        """
        return dict(self._previous_sequences)

    def ingest(
        self,
        records: Sequence[HierarchySyncEnvelope | Mapping[str, object] | str],
    ) -> HierarchySyncLedger:
        """Parse a transport batch, ingest it, and advance accepted watermarks.

        Parameters
        ----------
        records : Sequence[HierarchySyncEnvelope | Mapping[str, object] | str]
            The transport records to ingest.

        Returns
        -------
        HierarchySyncLedger
            The sync ledger with advanced watermarks.
        """
        envelopes = tuple(load_hierarchy_sync_envelope(record) for record in records)
        ledger = ingest_hierarchy_sync_envelopes(
            envelopes,
            previous_sequences=self._previous_sequences,
            hierarchy=self._hierarchy,
            degraded_threshold=self._degraded_threshold,
            critical_threshold=self._critical_threshold,
            min_confidence=self._min_confidence,
            protocol_version=self._protocol_version,
        )
        for envelope in ledger.accepted:
            self._previous_sequences[envelope.source_node] = envelope.sequence
        return ledger

    def ingest_batch(
        self,
        records: Sequence[HierarchySyncEnvelope | Mapping[str, object] | str],
    ) -> HierarchySyncLedger:
        """Alias for adapter batch ingestion.

        Parameters
        ----------
        records : Sequence[HierarchySyncEnvelope | Mapping[str, object] | str]
            The transport records to ingest.

        Returns
        -------
        HierarchySyncLedger
            The sync ledger for the ingested batch.
        """
        return self.ingest(records)

    def to_audit_record(self) -> dict[str, object]:
        """Return socket-free runtime state for audit logging.

        Returns
        -------
        dict[str, object]
            Return socket-free runtime state for audit logging.
        """
        return {
            "hierarchy": self._hierarchy,
            "protocol_version": self._protocol_version,
            "previous_sequences": {
                source: self._previous_sequences[source]
                for source in sorted(self._previous_sequences)
            },
            "audit_scope": _AUDIT_SCOPE_REDUCED_SUMMARIES,
        }


def build_hierarchy_sync_envelope(
    summary: ChildSupervisorSummary,
    *,
    source_node: str,
    sequence: int,
    protocol_version: str = _DEFAULT_HIERARCHY_SYNC_PROTOCOL,
    monotonic_time_s: float | None = None,
) -> HierarchySyncEnvelope:
    """Build a deterministic edge/cloud hierarchy sync envelope.

    The envelope is transport-neutral: callers may write it to JSONL, send it
    over a message bus, or hand it to tests without this module opening sockets
    or performing live deployment work.

    Parameters
    ----------
    summary : ChildSupervisorSummary
        The child supervisor summary.
    source_node : str
        Identifier of the source node.
    sequence : int
        Monotonic envelope sequence number.
    protocol_version : str
        Hierarchy sync protocol version.
    monotonic_time_s : float | None
        Monotonic timestamp in seconds, or ``None``.

    Returns
    -------
    HierarchySyncEnvelope
        The deterministic hierarchy sync envelope.
    """
    return HierarchySyncEnvelope(
        protocol_version=protocol_version,
        source_node=source_node,
        sequence=sequence,
        summary=summary,
        monotonic_time_s=monotonic_time_s,
    )


def load_hierarchy_sync_envelope(
    record: HierarchySyncEnvelope | Mapping[str, object] | str,
) -> HierarchySyncEnvelope:
    """Parse a JSON string or decoded mapping into a strict sync envelope.

    Parameters
    ----------
    record : HierarchySyncEnvelope | Mapping[str, object] | str
        A sync envelope, decoded mapping, or JSON string.

    Returns
    -------
    HierarchySyncEnvelope
        The parsed strict sync envelope.

    Raises
    ------
    ValueError
        If the record cannot be parsed into a strict envelope.
    """
    if isinstance(record, HierarchySyncEnvelope):
        _validate_envelope_reduced_only(record)
        return _canonical_hierarchy_sync_envelope(record)
    payload = _load_mapping_record(record)
    _reject_raw_hierarchy_keys(payload, "hierarchy sync envelope")
    _reject_unknown_keys(
        payload,
        allowed=_HIERARCHY_SYNC_ENVELOPE_KEYS,
        location="hierarchy sync envelope",
    )
    summary_record = payload.get("summary")
    if not isinstance(summary_record, Mapping):
        raise ValueError("summary must be a decoded mapping")
    _reject_raw_hierarchy_keys(summary_record, "hierarchy sync summary")
    _reject_unknown_keys(
        summary_record,
        allowed=_HIERARCHY_SYNC_SUMMARY_KEYS,
        location="hierarchy sync summary",
    )

    sequence = _require_integer(payload.get("sequence"), "sequence")
    monotonic_time_s = payload.get("monotonic_time_s")
    if monotonic_time_s is not None:
        monotonic_time_s = _require_float(monotonic_time_s, "monotonic_time_s")
        if monotonic_time_s < 0.0:
            raise ValueError("monotonic_time_s must be finite and non-negative")

    envelope = HierarchySyncEnvelope(
        protocol_version=_require_text_field(payload, "protocol_version"),
        source_node=_require_text_field(payload, "source_node"),
        sequence=sequence,
        summary=_load_child_summary(summary_record),
        monotonic_time_s=monotonic_time_s,
    )
    _validate_envelope_reduced_only(envelope)
    return envelope


def ingest_hierarchy_sync_envelopes(
    envelopes: Sequence[HierarchySyncEnvelope],
    *,
    previous_sequences: Mapping[str, int] | None = None,
    hierarchy: str = "edge_cloud_summary_sync",
    degraded_threshold: float = 0.65,
    critical_threshold: float = 0.35,
    min_confidence: float = 0.5,
    protocol_version: str = _DEFAULT_HIERARCHY_SYNC_PROTOCOL,
) -> HierarchySyncLedger:
    """Validate envelopes and build a parent plan from accepted summaries.

    Parent nodes reject stale or duplicate sequence numbers per source node and
    reject protocol-version mismatches. Accepted envelopes are sorted by source
    node and sequence before parent-state composition, making JSONL replay and
    cloud ingestion deterministic.

    Parameters
    ----------
    envelopes : Sequence[HierarchySyncEnvelope]
        The ordered transport envelopes.
    previous_sequences : Mapping[str, int] | None
        Accepted per-source sequence watermarks, or ``None``.
    hierarchy : str
        Hierarchy label.
    degraded_threshold : float
        Coherence threshold below which a child is degraded.
    critical_threshold : float
        Coherence threshold below which a child is critical.
    min_confidence : float
        Minimum child summary confidence to include.
    protocol_version : str
        Hierarchy sync protocol version.

    Returns
    -------
    HierarchySyncLedger
        The sync ledger built from accepted summaries.

    Raises
    ------
    ValueError
        If an envelope fails validation.
    """
    canonical_envelopes = tuple(
        _canonical_hierarchy_sync_envelope(envelope) for envelope in envelopes
    )
    for envelope in canonical_envelopes:
        _validate_envelope_reduced_only(envelope)
    _validate_plan_inputs(
        children=[envelope.summary for envelope in canonical_envelopes]
        or [_dummy_summary()],
        hierarchy=hierarchy,
        degraded_threshold=degraded_threshold,
        critical_threshold=critical_threshold,
        min_confidence=min_confidence,
    )
    expected_sequences = _normalise_previous_sequences(previous_sequences)
    accepted: list[HierarchySyncEnvelope] = []
    rejected: list[dict[str, object]] = []
    valid_protocol: list[HierarchySyncEnvelope] = []
    for envelope in sorted(
        canonical_envelopes,
        key=lambda item: (item.source_node, item.sequence),
    ):
        if envelope.protocol_version != protocol_version:
            rejected.append(_rejection(envelope, "protocol_version_mismatch"))
            continue
        valid_protocol.append(envelope)

    duplicate_sequence_conflict_sources = _duplicate_sequence_conflict_sources(
        valid_protocol
    )
    latest_by_source: dict[str, HierarchySyncEnvelope] = {}
    for envelope in valid_protocol:
        if envelope.source_node in duplicate_sequence_conflict_sources:
            continue
        latest = latest_by_source.get(envelope.source_node)
        if latest is None or envelope.sequence > latest.sequence:
            latest_by_source[envelope.source_node] = envelope

    for envelope in valid_protocol:
        if envelope.source_node in duplicate_sequence_conflict_sources:
            rejected.append(_rejection(envelope, "duplicate_sequence_conflict"))
            continue
        latest = latest_by_source[envelope.source_node]
        previous_sequence = expected_sequences.get(envelope.source_node, -1)
        if envelope.sequence <= previous_sequence or envelope is not latest:
            rejected.append(_rejection(envelope, "stale_or_duplicate_sequence"))
            continue
        expected_sequences[envelope.source_node] = envelope.sequence
        accepted.append(envelope)

    accepted_tuple = tuple(
        sorted(accepted, key=lambda item: (item.source_node, item.sequence))
    )
    if not accepted_tuple:
        raise ValueError("at least one hierarchy sync envelope must be accepted")
    plan = build_hierarchical_orchestration_plan(
        [envelope.summary for envelope in accepted_tuple],
        hierarchy=hierarchy,
        degraded_threshold=degraded_threshold,
        critical_threshold=critical_threshold,
        min_confidence=min_confidence,
    )
    return HierarchySyncLedger(
        accepted=accepted_tuple,
        rejected=tuple(
            sorted(
                rejected,
                key=_rejection_sort_key,
            )
        ),
        plan=plan,
    )


def _load_mapping_record(
    record: Mapping[str, object] | str,
) -> Mapping[str, object]:
    if isinstance(record, str):
        try:
            decoded = json.loads(
                record,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except json.JSONDecodeError as exc:
            raise ValueError("hierarchy sync envelope must be valid JSON") from exc
        except ValueError as exc:
            raise ValueError(
                "hierarchy sync envelope JSON must be canonical finite JSON"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise ValueError("hierarchy sync envelope JSON must decode to a mapping")
        return decoded
    if not isinstance(record, Mapping):
        raise ValueError("hierarchy sync envelope must be a mapping or JSON string")
    return record


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {token}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    record: dict[str, object] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate JSON object key is not allowed: {key}")
        record[key] = value
    return record


def _load_child_summary(record: Mapping[str, object]) -> ChildSupervisorSummary:
    _reject_unknown_keys(
        record,
        allowed=_HIERARCHY_SYNC_SUMMARY_KEYS,
        location="hierarchy sync summary",
    )
    metadata = record.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a decoded mapping")
    _validate_metadata_value(metadata, "summary.metadata")
    return ChildSupervisorSummary(
        name=_require_text_field(record, "name"),
        channel=_require_text_field(record, "channel"),
        R=_require_float(record.get("R"), "R"),
        psi=_require_float(record.get("psi"), "psi"),
        regime=_require_text_field(record, "regime", default=_REGIME_NOMINAL),
        confidence=_require_float(record.get("confidence", 1.0), "confidence"),
        metadata=dict(metadata),
    )


def _require_text_field(
    record: Mapping[str, object],
    key: str,
    *,
    default: str | None = None,
) -> str:
    value = record.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _reject_raw_hierarchy_keys(
    record: Mapping[str, object],
    location: str,
) -> None:
    forbidden = {
        key
        for key in record
        if isinstance(key, str) and _is_forbidden_hierarchy_key(key)
    }
    if forbidden:
        keys = ", ".join(sorted(forbidden))
        raise ValueError(f"{location} contains raw child evidence: {keys}")


def _reject_unknown_keys(
    record: Mapping[str, object],
    *,
    allowed: frozenset[str],
    location: str,
) -> None:
    unknown: list[str] = []
    for key in record:
        if not isinstance(key, str):
            raise ValueError(f"{location} keys must be strings")
        if key not in allowed:
            unknown.append(key)
    if unknown:
        keys = ", ".join(sorted(unknown))
        raise ValueError(f"{location} contains unknown fields: {keys}")
