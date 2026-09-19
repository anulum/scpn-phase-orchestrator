# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Producer-bound historical assessment carrier

"""Canonical review carrier retaining independently sealed source and assessment."""

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
    reactor_producer_registry_bytes,
    reactor_producer_registry_from_bytes,
)
from .regime_assessment import ReactorRegimeAssessment, regime_assessment_from_bytes
from .vocabulary import require_exact_keys, require_sha256

PRODUCER_BOUND_ASSESSMENT_SCHEMA: Final = (
    "scpn-phase-orchestrator.producer-bound-historical-assessment.v1"
)
PRODUCER_BOUND_ASSESSMENT_VERSION: Final = "1.0.0"
MAX_PRODUCER_BOUND_ASSESSMENT_BYTES: Final = 8 * 1024 * 1024
_TEXT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "configuration",
        "producer_registry_version",
        "producer_registry_digest",
        "source_handoff_schema",
        "source_handoff_json",
        "source_handoff_sha256",
        "assessment_json",
        "assessment_sha256",
        "authority",
    }
)


@dataclass(frozen=True)
class ProducerBoundHistoricalAssessment:
    """Decoded review carrier with byte-preserved historical lineage.

    Attributes
    ----------
    source_bytes : bytes
        Original canonical handoff, unchanged by producer registry selection.
    assessment_bytes : bytes
        Original canonical assessment with its own registry and ontology pins.
    binding : HistoricalProducerBinding
        Verified historical source ownership, not a direct source allocation.
    assessment : ReactorRegimeAssessment
        Review-only assessment whose source identity crosslinks were checked.

    Notes
    -----
    Successful decoding does not qualify observations, clocks, freshness or
    calibration for CONTROL. This type carries no control decision or intent.
    """

    source_bytes: bytes
    assessment_bytes: bytes
    binding: HistoricalProducerBinding
    assessment: ReactorRegimeAssessment


def producer_bound_assessment_from_bytes(
    data: bytes,
    *,
    expected_sha256: str,
    configuration: str,
    producer_registry_version: str,
    producer_registry_digest: str,
) -> ProducerBoundHistoricalAssessment:
    """Decode a pinned carrier and check source/assessment identity crosslinks.

    Parameters
    ----------
    data : bytes
        Canonical UTF-8 JSON: sorted keys, compact separators, ASCII escaping,
        no trailing newline. Outer object is flat; inner documents are strings.
        The entire carrier is bounded to 8 MiB before hashing or parsing.
    expected_sha256 : str
        Caller-required SHA-256 of the complete outer bytes.
    configuration : str
        Exact expected canonical configuration; aliases are not substituted.
    producer_registry_version : str
        Explicit caller-required ownership snapshot version.
    producer_registry_digest : str
        Explicit caller-required ownership snapshot digest.

    Returns
    -------
    ProducerBoundHistoricalAssessment
        Original source/assessment bytes and decoded review-only identities.
        Source and assessment registry scopes are validated independently.

    Raises
    ------
    ValueError
        If encoding, size, nesting, duplicate keys, pins, schemas, authority or
        source crosslinks disagree. Existing legacy decoders validate each inner
        contract; unsupported new carriers cannot be nested recursively.
    """
    if (
        not isinstance(data, bytes)
        or not data
        or len(data) > MAX_PRODUCER_BOUND_ASSESSMENT_BYTES
    ):
        raise ValueError("producer-bound assessment requires bounded bytes")
    require_sha256(expected_sha256, field="expected_sha256")
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("producer-bound assessment digest mismatch")
    _require_flat_document(data)
    try:
        raw = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_keys)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid producer-bound assessment JSON") from exc
    record = require_exact_keys(
        raw, required=_TEXT_FIELDS | {"actionable"}, field="producer-bound assessment"
    )
    if any(not isinstance(record[key], str) for key in _TEXT_FIELDS):
        raise ValueError("producer-bound assessment text fields require strings")
    if (
        record["schema"] != PRODUCER_BOUND_ASSESSMENT_SCHEMA
        or record["schema_version"] != PRODUCER_BOUND_ASSESSMENT_VERSION
    ):
        raise ValueError("unsupported producer-bound assessment wire")
    if record["authority"] != "review_only" or record["actionable"] is not False:
        raise ValueError("producer-bound assessment must remain review-only")
    if (
        record["configuration"],
        record["producer_registry_version"],
        record["producer_registry_digest"],
    ) != (configuration, producer_registry_version, producer_registry_digest):
        raise ValueError("producer-bound assessment caller policy mismatch")
    if _canonical_bytes(record) != data:
        raise ValueError("producer-bound assessment bytes are not canonical")
    try:
        source = cast(str, record["source_handoff_json"]).encode("utf-8")
        assessment_bytes = cast(str, record["assessment_json"]).encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("invalid inner document encoding") from exc
    assessment_digest = require_sha256(
        record["assessment_sha256"], field="assessment_sha256"
    )
    if hashlib.sha256(assessment_bytes).hexdigest() != assessment_digest:
        raise ValueError("inner assessment digest mismatch")
    binding = review_historical_producer_binding(
        source,
        expected_sha256=cast(str, record["source_handoff_sha256"]),
        configuration=configuration,
        handoff_schema=cast(str, record["source_handoff_schema"]),
        producer_registry_version=producer_registry_version,
        producer_registry_digest=producer_registry_digest,
    )
    assessment = regime_assessment_from_bytes(assessment_bytes)
    crosslinks = {
        "configuration": binding.configuration,
        "reactor_context_id": binding.reactor_context_id,
        "event_id": binding.event_id,
        "source_project": binding.source_project,
        "source_revision": binding.source_revision,
        "source_handoff_schema": binding.handoff_schema,
        "source_handoff_sha256": binding.handoff_sha256,
        "source_semantic_ids": binding.source_semantic_ids,
    }
    for name, expected in crosslinks.items():
        if getattr(assessment, name) != expected:
            raise ValueError(f"assessment source crosslink mismatch: {name}")
    return ProducerBoundHistoricalAssessment(
        source, assessment_bytes, binding, assessment
    )


def producer_bound_assessment_to_bytes(
    source_bytes: bytes,
    assessment_bytes: bytes,
    *,
    source_handoff_schema: str,
    expected_source_sha256: str,
    expected_assessment_sha256: str,
    configuration: str,
    producer_registry_version: str,
    producer_registry_digest: str,
) -> bytes:
    """Build a validated carrier without reserialising either inner document.

    Parameters
    ----------
    source_bytes : bytes
        Existing canonical FUSION/MIF handoff bytes.
    assessment_bytes : bytes
        Existing canonical assessment bytes referring to this complete source.
    source_handoff_schema : str
        Explicit historical handoff schema selector.
    expected_source_sha256 : str
        Caller-required digest of the complete source handoff bytes.
    expected_assessment_sha256 : str
        Caller-required digest of the complete assessment bytes.
    configuration : str
        Caller-required exact canonical configuration.
    producer_registry_version : str
        Explicit ownership snapshot version.
    producer_registry_digest : str
        Explicit ownership snapshot digest.

    Returns
    -------
    bytes
        Canonical outer carrier, validated through the public decoder.

    Raises
    ------
    ValueError
        If either inner document or their combined carrier is invalid/oversized,
        or the independently supplied custody and ownership pins disagree.
    """
    for document in (source_bytes, assessment_bytes):
        if (
            not isinstance(document, bytes)
            or len(document) > MAX_PRODUCER_BOUND_ASSESSMENT_BYTES
        ):
            raise ValueError("inner document requires bounded bytes")
    try:
        source_text, assessment_text = (
            source_bytes.decode("utf-8"),
            assessment_bytes.decode("utf-8"),
        )
    except UnicodeError as exc:
        raise ValueError("invalid inner document encoding") from exc
    data = _canonical_bytes(
        {
            "schema": PRODUCER_BOUND_ASSESSMENT_SCHEMA,
            "schema_version": PRODUCER_BOUND_ASSESSMENT_VERSION,
            "configuration": configuration,
            "producer_registry_version": producer_registry_version,
            "producer_registry_digest": producer_registry_digest,
            "source_handoff_schema": source_handoff_schema,
            "source_handoff_json": source_text,
            "source_handoff_sha256": expected_source_sha256,
            "assessment_json": assessment_text,
            "assessment_sha256": expected_assessment_sha256,
            "authority": "review_only",
            "actionable": False,
        }
    )
    producer_bound_assessment_from_bytes(
        data,
        expected_sha256=hashlib.sha256(data).hexdigest(),
        configuration=configuration,
        producer_registry_version=producer_registry_version,
        producer_registry_digest=producer_registry_digest,
    )
    return data


def producer_bound_assessment_schema(
    *, producer_registry_version: str, producer_registry_digest: str
) -> dict[str, object]:
    """Generate outer structural constraints from one exact producer snapshot.

    Parameters
    ----------
    producer_registry_version : str
        Explicit required ownership snapshot version.
    producer_registry_digest : str
        Explicit required ownership snapshot digest.

    Returns
    -------
    dict[str, object]
        Draft 2020-12 schema for outer configuration/historical-schema pairs.
        Embedded canonical bytes, hashes and source/assessment crosslinks are
        runtime obligations, not claims made by this structural schema.

    Raises
    ------
    ValueError
        If the required registry release is unknown or corrupt.
    """
    data = reactor_producer_registry_bytes(
        version=producer_registry_version, digest=producer_registry_digest
    )
    registry = reactor_producer_registry_from_bytes(
        data, version=producer_registry_version, digest=producer_registry_digest
    )
    alternatives: list[dict[str, object]] = []
    for entry in cast("list[dict[str, object]]", registry["entries"]):
        adapter = entry["historical_review_adapter"]
        if isinstance(adapter, dict):
            alternatives.append(
                {
                    "properties": {
                        "configuration": {"const": entry["configuration"]},
                        "source_handoff_schema": {"const": adapter["handoff_schema"]},
                    }
                }
            )
    properties: dict[str, object] = {name: {"type": "string"} for name in _TEXT_FIELDS}
    properties.update(
        {
            "schema": {"const": PRODUCER_BOUND_ASSESSMENT_SCHEMA},
            "schema_version": {"const": PRODUCER_BOUND_ASSESSMENT_VERSION},
            "producer_registry_version": {"const": producer_registry_version},
            "producer_registry_digest": {"const": producer_registry_digest},
            "authority": {"const": "review_only"},
            "actionable": {"const": False},
        }
    )
    for name in ("source_handoff_sha256", "assessment_sha256"):
        properties[name] = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Producer-bound historical assessment carrier",
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_TEXT_FIELDS | {"actionable"}),
        "properties": properties,
        "oneOf": alternatives,
    }


def _canonical_bytes(record: object) -> bytes:
    """Encode flat carrier records with canonical ASCII-escaped JSON."""
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate keys before constructing a carrier record."""
    record: dict[str, object] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate carrier key: {key}")
        record[key] = value
    return record


def _require_flat_document(data: bytes) -> None:
    """Reject containers nested outside JSON strings before parsing."""
    quoted = escaped = False
    depth = 0
    for byte in data:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (123, 91):
            depth += 1
            if depth > 1 or byte == 91:
                raise ValueError("producer-bound assessment outer nesting is forbidden")
        elif byte in (125, 93):
            depth -= 1
