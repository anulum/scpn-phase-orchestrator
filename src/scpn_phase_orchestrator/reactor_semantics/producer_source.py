# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Registry-bound historical source review

"""Versioned source custody for allocated historical review adapters only.

The ownership snapshot grants no direct simulated, physical or telemetry roles.
A successful review preserves that boundary and does not qualify observations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, cast

from .producer_binding import (
    HistoricalProducerBinding,
    review_historical_producer_binding,
)
from .producer_registry import (
    reactor_producer_identity,
    reactor_producer_registry_bytes,
    reactor_producer_registry_from_bytes,
)
from .vocabulary import require_exact_keys, require_sha256

PRODUCER_SOURCE_SCHEMA: Final = (
    "scpn-phase-orchestrator.producer-bound-source-review.v1"
)
PRODUCER_SOURCE_VERSION: Final = "1.0.0"
MAX_PRODUCER_SOURCE_BYTES: Final = 8 * 1024 * 1024


@dataclass(frozen=True)
class ProducerSourcePolicy:
    """Caller-required source identity and immutable ownership snapshot.

    Attributes
    ----------
    configuration : str
        Exact canonical configuration, without alias resolution.
    producer_project : str
        Historical source owner, independently of the device project.
    source_role : str
        Only ``verified_review_adapter`` is allocated by the first snapshot.
        Physical, simulated and control-telemetry roles remain undeclared.
    source_schema : str
        Exact upstream producer schema carried by the historical handoff.
    handoff_schema : str
        Exact existing SPO handoff wire identifier, not the producer schema.
    producer_registry_version : str
        Explicit ownership snapshot version; no latest-version default.
    producer_registry_digest : str
        SHA-256 of the complete packaged ownership snapshot bytes.
    """

    configuration: str
    producer_project: str
    source_role: str
    source_schema: str
    handoff_schema: str
    producer_registry_version: str
    producer_registry_digest: str


@dataclass(frozen=True)
class ProducerSourceReview:
    """Validated ownership review retaining original historical source bytes.

    Attributes
    ----------
    source_bytes : bytes
        Byte-identical canonical historical handoff.
    binding : HistoricalProducerBinding
        Checked source lineage and separate historical configuration registry.
    policy : ProducerSourcePolicy
        Exact caller identity expectations validated during decoding.

    Notes
    -----
    The binding is review-only and non-actionable. This object carries neither
    an observation qualification nor a CONTROL decision or protection receipt.
    """

    source_bytes: bytes
    binding: HistoricalProducerBinding
    policy: ProducerSourcePolicy


def _metadata(policy: ProducerSourcePolicy) -> dict[str, object]:
    """Resolve exact caller identity against the pinned historical adapter."""
    registry = reactor_producer_registry_from_bytes(
        reactor_producer_registry_bytes(
            version=policy.producer_registry_version,
            digest=policy.producer_registry_digest,
        ),
        version=policy.producer_registry_version,
        digest=policy.producer_registry_digest,
    )
    identity = reactor_producer_identity(
        policy.configuration,
        version=policy.producer_registry_version,
        digest=policy.producer_registry_digest,
    )
    adapter = identity["historical_review_adapter"]
    if not isinstance(adapter, dict) or (
        policy.producer_project,
        policy.source_role,
        policy.source_schema,
        policy.handoff_schema,
    ) != (
        adapter["producer_project"],
        adapter["ingress_state"],
        adapter["source_schema"],
        adapter["handoff_schema"],
    ):
        raise ValueError("source tuple has no allocated historical review adapter")
    return {
        "schema": PRODUCER_SOURCE_SCHEMA,
        "schema_version": PRODUCER_SOURCE_VERSION,
        "configuration": policy.configuration,
        "device_project": identity["device_project"],
        "producer_project": policy.producer_project,
        "source_role": policy.source_role,
        "source_schema": policy.source_schema,
        "handoff_schema": policy.handoff_schema,
        "producer_registry_version": policy.producer_registry_version,
        "producer_registry_digest": policy.producer_registry_digest,
        "reactor_registry_version": registry["reactor_registry_version"],
        "reactor_registry_digest": registry["reactor_registry_digest"],
        "semantic_profile_registry_version": registry[
            "semantic_profile_registry_version"
        ],
        "semantic_profile_registry_digest": registry[
            "semantic_profile_registry_digest"
        ],
        "adapter_api": adapter["adapter_api"],
        "semantic_profile": adapter["semantic_profile"],
        "semantic_profile_version": adapter["semantic_profile_version"],
        "authority": "review_only",
        "actionable": False,
    }


def _encode(record: object) -> bytes:
    """Encode source-review records with canonical ASCII-escaped JSON."""
    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate object keys while decoding source-review JSON."""
    record: dict[str, object] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError("duplicate producer source key")
        record[key] = value
    return record


def producer_source_from_bytes(
    data: bytes, *, expected_sha256: str, policy: ProducerSourcePolicy
) -> ProducerSourceReview:
    """Decode canonical source-review bytes under an exact caller policy.

    Parameters
    ----------
    data : bytes
        UTF-8 JSON with sorted keys, compact separators and ASCII escaping,
        no newline. The outer record is flat; the historical source is a string.
        Maximum complete carrier size is 8 MiB. Recursive carriers are refused
        by the explicitly selected historical decoder.
    expected_sha256 : str
        SHA-256 required by the caller over the complete outer bytes.
    policy : ProducerSourcePolicy
        Expected configuration, source role/schema and ownership registry pins.

    Returns
    -------
    ProducerSourceReview
        Original source bytes and checked, non-actionable historical lineage.

    Raises
    ------
    ValueError
        For an undeclared tuple, malformed/noncanonical carrier, mismatched pin,
        duplicate key, nested object, false authority or invalid historical source.
        Ownership snapshot pins and historical source registry pins are separate.
    """
    if not isinstance(data, bytes) or not data or len(data) > MAX_PRODUCER_SOURCE_BYTES:
        raise ValueError("producer source requires bounded bytes")
    require_sha256(expected_sha256, field="expected_sha256")
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("producer source digest mismatch")
    metadata = _metadata(policy)
    try:
        raw = json.loads(data.decode("utf-8"), object_pairs_hook=_unique)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("invalid producer source JSON") from exc
    record = require_exact_keys(
        raw,
        required=frozenset(metadata) | {"source_json", "source_sha256"},
        field="producer source",
    )
    if record["actionable"] is not False or any(
        type(record[key]) is not type(value) or record[key] != value
        for key, value in metadata.items()
    ):
        raise ValueError("producer source ownership or wire binding mismatch")
    if not isinstance(record["source_json"], str):
        raise ValueError("producer source requires a historical JSON string")
    source_digest = require_sha256(record["source_sha256"], field="source_sha256")
    if _encode(record) != data:
        raise ValueError("producer source bytes are not canonical")
    try:
        source = record["source_json"].encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("invalid historical source encoding") from exc
    binding = review_historical_producer_binding(
        source,
        expected_sha256=source_digest,
        configuration=policy.configuration,
        handoff_schema=policy.handoff_schema,
        producer_registry_version=policy.producer_registry_version,
        producer_registry_digest=policy.producer_registry_digest,
    )
    return ProducerSourceReview(source, binding, policy)


def producer_source_to_bytes(
    source_bytes: bytes, *, expected_source_sha256: str, policy: ProducerSourcePolicy
) -> bytes:
    """Wrap historical source bytes without changing their registry or encoding.

    Parameters
    ----------
    source_bytes : bytes
        Complete canonical handoff accepted by the selected historical decoder.
    expected_source_sha256 : str
        Caller-required hash of those original bytes.
    policy : ProducerSourcePolicy
        Exact expected ownership tuple and immutable registry release.

    Returns
    -------
    bytes
        Canonical source-review carrier validated through the public decoder.

    Raises
    ------
    ValueError
        If source encoding, size, digest or allocated historical identity fails.
        The writer cannot create a direct-source allocation.
    """
    if (
        not isinstance(source_bytes, bytes)
        or len(source_bytes) > MAX_PRODUCER_SOURCE_BYTES
    ):
        raise ValueError("producer source requires bounded bytes")
    try:
        text = source_bytes.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("invalid historical source encoding") from exc
    record = {
        **_metadata(policy),
        "source_json": text,
        "source_sha256": expected_source_sha256,
    }
    data = _encode(record)
    producer_source_from_bytes(
        data, expected_sha256=hashlib.sha256(data).hexdigest(), policy=policy
    )
    return data


def producer_source_schema(
    *, producer_registry_version: str, producer_registry_digest: str
) -> dict[str, object]:
    """Generate the finite outer ownership grammar from the pinned registry.

    Parameters
    ----------
    producer_registry_version : str
        Explicit immutable ownership snapshot version.
    producer_registry_digest : str
        Exact snapshot digest.

    Returns
    -------
    dict[str, object]
        Draft 2020-12 schema for the allocated historical tuples. Direct roles
        are rejected. Byte canonicality, hashes and inner semantic lineage remain
        runtime obligations; schema acceptance does not qualify physical evidence.

    Raises
    ------
    ValueError
        If the selected registry version/digest pair is unavailable.
    """
    data = reactor_producer_registry_bytes(
        version=producer_registry_version, digest=producer_registry_digest
    )
    registry = reactor_producer_registry_from_bytes(
        data, version=producer_registry_version, digest=producer_registry_digest
    )
    alternatives: list[dict[str, object]] = []
    for identity in cast("list[dict[str, object]]", registry["entries"]):
        adapter = identity["historical_review_adapter"]
        if not isinstance(adapter, dict):
            continue
        metadata = _metadata(
            ProducerSourcePolicy(
                cast(str, identity["configuration"]),
                adapter["producer_project"],
                adapter["ingress_state"],
                adapter["source_schema"],
                adapter["handoff_schema"],
                producer_registry_version,
                producer_registry_digest,
            )
        )
        properties = {key: {"const": value} for key, value in metadata.items()}
        properties.update(
            {
                "source_json": {"type": "string"},
                "source_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            }
        )
        alternatives.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(properties),
                "properties": properties,
            }
        )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Registry-bound historical source review",
        "oneOf": alternatives,
    }
