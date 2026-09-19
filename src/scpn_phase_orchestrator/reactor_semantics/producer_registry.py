# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Immutable producer ownership registry

"""Exact released ownership snapshots, distinct from source admission.

The first snapshot allocates no direct source roles. Historical review-adapter
metadata explains existing codecs; it does not grant a new ingress capability.
Decoding recognises only exact packaged bytes, with no latest-version fallback.
"""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from typing import Final, cast

REACTOR_PRODUCER_REGISTRY_SCHEMA: Final = (
    "scpn-phase-orchestrator.reactor-producer-registry.v1"
)
REACTOR_PRODUCER_REGISTRY_VERSION: Final = "1.0.0"
REACTOR_PRODUCER_REGISTRY_SHA256: Final = (
    "5e5b0a7e1b1bccb13123087213d4cdd3c506a7eaf20f01dc6e842c81f8882faa"
)
MAX_REACTOR_PRODUCER_REGISTRY_BYTES: Final = 65_536


def reactor_producer_registry_bytes(*, version: str, digest: str) -> bytes:
    """Load one exact packaged producer registry without registry substitution.

    Parameters
    ----------
    version : str
        Explicit registry version; no implicit current release is selected.
    digest : str
        SHA-256 of the complete canonical UTF-8 document, including its newline.

    Returns
    -------
    bytes
        Immutable canonical snapshot bytes.

    Raises
    ------
    ValueError
        If the release pair is unknown or packaged bytes fail their custody pin.
    """
    if (version, digest) != (
        REACTOR_PRODUCER_REGISTRY_VERSION,
        REACTOR_PRODUCER_REGISTRY_SHA256,
    ):
        raise ValueError("unrecognised producer registry release")
    data = (
        files("scpn_phase_orchestrator.reactor_semantics")
        .joinpath("data/producer_registry_v1.json")
        .read_bytes()
    )
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("packaged producer registry digest mismatch")
    return data


def reactor_producer_registry_from_bytes(
    data: bytes, *, version: str, digest: str
) -> dict[str, object]:
    """Decode an exact allowlisted ownership snapshot into a detached record.

    Parameters
    ----------
    data : bytes
        Complete canonical UTF-8 JSON bytes, bounded to 65,536 bytes. Equality
        to the pinned snapshot rejects duplicate keys, non-finite values,
        alternative encodings, reordered entries and additional nesting before
        any untrusted JSON parsing takes place.
    version : str
        Caller-required registry version.
    digest : str
        Caller-required SHA-256 of the complete document.

    Returns
    -------
    dict[str, object]
        Fresh record. Mutations cannot alter the packaged registry or later reads.

    Raises
    ------
    ValueError
        If the input type, bound, release pair or exact bytes are invalid.
    """
    if not isinstance(data, bytes) or len(data) > MAX_REACTOR_PRODUCER_REGISTRY_BYTES:
        raise ValueError("producer registry requires bounded bytes")
    expected = reactor_producer_registry_bytes(version=version, digest=digest)
    if data != expected:
        raise ValueError("producer registry bytes do not match the required release")
    return cast("dict[str, object]", json.loads(expected))


def reactor_producer_identity(
    configuration: str, *, version: str, digest: str
) -> dict[str, object]:
    """Query a canonical configuration's ownership without admitting a source.

    Parameters
    ----------
    configuration : str
        Exact canonical configuration identifier. Aliases are not substituted.
    version : str
        Explicit producer-registry version.
    digest : str
        Exact required producer-registry document digest.

    Returns
    -------
    dict[str, object]
        Detached ownership, undeclared direct-allocation state and historical
        adapter metadata. An identity match grants no evidence or action rights.

    Raises
    ------
    ValueError
        If the release or configuration is not recognised.
    """
    data = reactor_producer_registry_bytes(version=version, digest=digest)
    record = reactor_producer_registry_from_bytes(data, version=version, digest=digest)
    for entry in cast("list[dict[str, object]]", record["entries"]):
        if entry["configuration"] == configuration:
            return entry
    raise ValueError("configuration is not in the required producer registry")


def reactor_producer_registry_schema(*, version: str, digest: str) -> dict[str, object]:
    """Generate structural JSON Schema for one exact finite registry snapshot.

    Parameters
    ----------
    version : str
        Explicit registry version to describe.
    digest : str
        Exact required registry document digest.

    Returns
    -------
    dict[str, object]
        Draft 2020-12 schema generated from the same record as runtime queries.
        This validates registry content, not canonical bytes or reactor evidence.

    Raises
    ------
    ValueError
        If the required release is unknown or its packaged bytes are corrupt.
    """
    data = reactor_producer_registry_bytes(version=version, digest=digest)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Exact reactor producer ownership registry",
        "const": reactor_producer_registry_from_bytes(
            data, version=version, digest=digest
        ),
    }
