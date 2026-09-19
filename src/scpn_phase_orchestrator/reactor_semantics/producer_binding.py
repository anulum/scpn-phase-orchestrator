# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Historical producer binding review

"""Bind existing canonical handoffs to ownership without granting ingress."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal, cast

from .handoff import (
    HANDOFF_SCHEMA,
    MAX_HANDOFF_JSON_BYTES,
    ReactorSemanticHandoff,
    handoff_from_bytes,
)
from .mif_merge_compression import (
    MAX_MIF_MERGE_COMPRESSION_HANDOFF_BYTES,
    MIFMergeCompressionHandoff,
    mif_merge_compression_handoff_from_bytes,
)
from .producer_registry import reactor_producer_identity
from .vocabulary import require_sha256


@dataclass(frozen=True)
class HistoricalProducerBinding:
    """Read-only ownership match, not evidence qualification or admission.

    The source registry pin is retained separately from the producer registry.
    This local result is not a serialised source or assessment wire contract.
    """

    configuration: str
    device_project: str
    source_project: str
    source_revision: str
    source_schema: str
    handoff_schema: str
    handoff_sha256: str
    source_registry_version: str
    source_registry_digest: str
    reactor_context_id: str
    event_id: str
    source_semantic_ids: tuple[str, ...]
    producer_registry_version: str
    producer_registry_digest: str
    authority: Literal["review_only"] = field(default="review_only", init=False)
    actionable: Literal[False] = field(default=False, init=False)


def review_historical_producer_binding(
    data: bytes,
    *,
    expected_sha256: str,
    configuration: str,
    handoff_schema: str,
    producer_registry_version: str,
    producer_registry_digest: str,
) -> HistoricalProducerBinding:
    """Validate canonical legacy bytes against an explicitly pinned owner map.

    Parameters
    ----------
    data : bytes
        Complete handoff bytes, within the selected legacy decoder's bound.
    expected_sha256 : str
        Caller-required SHA-256 of the entire handoff, not its source payload.
    configuration : str
        Exact canonical configuration expected by the caller.
    handoff_schema : str
        Explicit existing FUSION or MIF handoff schema. No schema guessing.
    producer_registry_version : str
        Caller-required producer ownership snapshot version.
    producer_registry_digest : str
        Caller-required ownership snapshot digest.

    Returns
    -------
    HistoricalProducerBinding
        Matching ownership and unchanged source identity and registry pins.
        Neither a direct source allocation nor assessment/CONTROL admission.

    Raises
    ------
    ValueError
        If bytes, custody pins, configuration or historical adapter disagree.
        Legacy decoders also enforce their existing full contract validation.

    Notes
    -----
    The exact immutable registry admits only the FUSION and MIF handoff schemas
    and string device identities. Schema equality is checked before dispatch;
    the remaining dispatch case is therefore MIF. These properties follow from
    the pinned registry bytes, not from caller-provided dictionaries.
    """
    identity = reactor_producer_identity(
        configuration,
        version=producer_registry_version,
        digest=producer_registry_digest,
    )
    adapter = identity["historical_review_adapter"]
    if not isinstance(adapter, dict):
        raise ValueError("configuration has no historical review adapter")
    if handoff_schema != adapter["handoff_schema"]:
        raise ValueError("handoff schema does not match historical review adapter")
    limit = (
        MAX_HANDOFF_JSON_BYTES
        if handoff_schema == HANDOFF_SCHEMA
        else MAX_MIF_MERGE_COMPRESSION_HANDOFF_BYTES
    )
    if not isinstance(data, bytes) or len(data) > limit:
        raise ValueError("historical handoff requires bounded bytes")
    require_sha256(expected_sha256, field="expected_sha256")
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("historical handoff digest mismatch")
    handoff: ReactorSemanticHandoff | MIFMergeCompressionHandoff
    if handoff_schema == HANDOFF_SCHEMA:
        handoff = handoff_from_bytes(data)
    else:
        handoff = mif_merge_compression_handoff_from_bytes(data)
    if (
        handoff.context.configuration != configuration
        or handoff.source_project != adapter["producer_project"]
        or handoff.source_schema != adapter["source_schema"]
    ):
        raise ValueError("handoff source does not match historical review adapter")
    device_project = cast(str, identity["device_project"])
    return HistoricalProducerBinding(
        configuration=configuration,
        device_project=device_project,
        source_project=handoff.source_project,
        source_revision=handoff.source_revision,
        source_schema=handoff.source_schema,
        handoff_schema=handoff.schema,
        handoff_sha256=expected_sha256,
        source_registry_version=handoff.context.registry_version,
        source_registry_digest=handoff.context.registry_digest,
        reactor_context_id=handoff.context.context_id,
        event_id=handoff.event_id,
        source_semantic_ids=tuple(sorted(item.phase_id for item in handoff.semantics)),
        producer_registry_version=producer_registry_version,
        producer_registry_digest=producer_registry_digest,
    )
