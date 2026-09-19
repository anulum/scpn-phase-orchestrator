# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Historical producer binding tests

"""Exercise original MIF evidence through public decoding and ownership review."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scpn_phase_orchestrator.reactor_semantics import (
    DEFAULT_REACTOR_REGISTRY,
    REACTOR_REGISTRY_V1_0_0,
    ReactorConfigurationRegistry,
    coupled_transport_handoff_from_fusion_bytes,
    handoff_from_bytes,
    handoff_to_bytes,
    mif_merge_compression_handoff_from_mif_bytes,
    mif_merge_compression_handoff_to_bytes,
)
from scpn_phase_orchestrator.reactor_semantics.mif_merge_compression import (
    MAX_MIF_MERGE_COMPRESSION_HANDOFF_BYTES,
    MIF_MERGE_COMPRESSION_HANDOFF_SCHEMA,
)
from scpn_phase_orchestrator.reactor_semantics.producer_binding import (
    review_historical_producer_binding,
)
from scpn_phase_orchestrator.reactor_semantics.producer_registry import (
    REACTOR_PRODUCER_REGISTRY_SHA256,
    REACTOR_PRODUCER_REGISTRY_VERSION,
    reactor_producer_identity,
)

POLICY = {
    "configuration": "frc_compression_mif",
    "handoff_schema": MIF_MERGE_COMPRESSION_HANDOFF_SCHEMA,
    "producer_registry_version": REACTOR_PRODUCER_REGISTRY_VERSION,
    "producer_registry_digest": REACTOR_PRODUCER_REGISTRY_SHA256,
}


def _handoff_bytes(registry: ReactorConfigurationRegistry) -> bytes:
    """Decode original MIF evidence using the selected historical registry."""
    source = Path(
        "tests/fixtures/mif_merge_compression/mif_merge_compression_observation_v1.json"
    ).read_bytes()
    handoff = mif_merge_compression_handoff_from_mif_bytes(source, registry=registry)
    return mif_merge_compression_handoff_to_bytes(handoff, registry=registry)


@pytest.mark.parametrize(
    "registry", [REACTOR_REGISTRY_V1_0_0, DEFAULT_REACTOR_REGISTRY]
)
def test_original_mif_binding_preserves_source_registry(
    registry: ReactorConfigurationRegistry,
) -> None:
    """Preserve the original registry and byte seal through ownership binding."""
    data = _handoff_bytes(registry)
    digest = hashlib.sha256(data).hexdigest()
    if registry == REACTOR_REGISTRY_V1_0_0:
        assert (
            digest == "c0f03b7c49346c39342598275556e8ac28c93138ba14f6e21d6739400e0edeb2"
        )
    binding = review_historical_producer_binding(data, expected_sha256=digest, **POLICY)
    assert binding.source_project == binding.device_project == "SCPN-MIF-CORE"
    assert binding.source_registry_version == registry.version
    assert binding.source_registry_digest == registry.digest
    assert binding.handoff_sha256 == digest
    assert binding.authority == "review_only"
    assert binding.actionable is False


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("configuration", "conventional_tokamak", "schema does not match"),
        ("configuration", "tokamak", "configuration is not"),
        ("handoff_schema", "unknown.v1", "schema does not match"),
        ("producer_registry_version", "9.0.0", "unrecognised"),
        ("producer_registry_digest", "0" * 64, "unrecognised"),
    ],
)
def test_caller_binding_mismatch_refuses(field: str, value: str, message: str) -> None:
    """Reject caller identity pins that disagree with the historical adapter."""
    data = _handoff_bytes(REACTOR_REGISTRY_V1_0_0)
    policy = {**POLICY, field: value}
    with pytest.raises(ValueError, match=message):
        review_historical_producer_binding(
            data, expected_sha256=hashlib.sha256(data).hexdigest(), **policy
        )


def test_custody_and_resealed_noncanonical_bytes_refuse() -> None:
    """Reject changed digests and malformed resealed historical handoffs."""
    data = _handoff_bytes(REACTOR_REGISTRY_V1_0_0)
    with pytest.raises(ValueError, match="digest mismatch"):
        review_historical_producer_binding(data, expected_sha256="0" * 64, **POLICY)
    with pytest.raises(ValueError):
        review_historical_producer_binding(data, expected_sha256="invalid", **POLICY)
    for candidate in (data + b"\n", b"{}", b'{"x":NaN}', b'{"x":1,"x":2}'):
        with pytest.raises(ValueError):
            review_historical_producer_binding(
                candidate,
                expected_sha256=hashlib.sha256(candidate).hexdigest(),
                **POLICY,
            )


def test_unallocated_configurations_do_not_inherit_legacy_adapter() -> None:
    """Refuse historical adapter inheritance for every unallocated configuration."""
    for configuration in DEFAULT_REACTOR_REGISTRY.configurations:
        identity = reactor_producer_identity(
            configuration,
            version=REACTOR_PRODUCER_REGISTRY_VERSION,
            digest=REACTOR_PRODUCER_REGISTRY_SHA256,
        )
        if identity["historical_review_adapter"] is not None:
            continue
        with pytest.raises(ValueError, match="no historical review adapter"):
            review_historical_producer_binding(
                b"{}",
                expected_sha256=hashlib.sha256(b"{}").hexdigest(),
                **{**POLICY, "configuration": configuration},
            )


@pytest.mark.parametrize(
    "data",
    [None, b" " * (MAX_MIF_MERGE_COMPRESSION_HANDOFF_BYTES + 1)],
    ids=["nonbytes", "oversized"],
)
def test_invalid_handoff_byte_boundary_is_refused(data: bytes | None) -> None:
    """Reject nonbytes and oversized inputs before hashing or legacy decoding."""
    with pytest.raises(ValueError, match="requires bounded bytes"):
        review_historical_producer_binding(data, expected_sha256="0" * 64, **POLICY)


def test_valid_legacy_schema_without_adapter_allocation_is_refused() -> None:
    """Reject a legacy-valid FUSION schema absent from the ownership snapshot."""
    raw = Path(
        "tests/fixtures/producer_source/torax_runtime_review_envelope_v1.json"
    ).read_bytes()
    handoff = coupled_transport_handoff_from_fusion_bytes(
        raw,
        expected_sha256="b594e2f8b72056426d628b638f6a849ef39e75daddc827305002b109365596c4",
        registry=REACTOR_REGISTRY_V1_0_0,
    )
    source = json.loads(handoff.source_envelope_json)
    source["schema"] = "scpn-fusion-core.unallocated-review.v1"
    changed = replace(
        handoff,
        source_schema=source["schema"],
        source_envelope_json=json.dumps(
            source,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    data = handoff_to_bytes(changed, registry=REACTOR_REGISTRY_V1_0_0)
    assert handoff_from_bytes(data).source_schema == source["schema"]
    with pytest.raises(ValueError, match="source does not match"):
        review_historical_producer_binding(
            data,
            expected_sha256=hashlib.sha256(data).hexdigest(),
            **{
                **POLICY,
                "configuration": "conventional_tokamak",
                "handoff_schema": handoff.schema,
            },
        )
