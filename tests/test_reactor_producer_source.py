# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Producer source public ingress tests

"""Real historical ingress, ownership refusals and generated wire constraints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from itertools import product
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from scpn_phase_orchestrator.reactor_semantics import (
    DEFAULT_REACTOR_REGISTRY,
    REACTOR_REGISTRY_V1_0_0,
    coupled_transport_handoff_from_fusion_bytes,
    handoff_from_bytes,
    handoff_to_bytes,
    mif_merge_compression_handoff_from_bytes,
    mif_merge_compression_handoff_from_mif_bytes,
    mif_merge_compression_handoff_to_bytes,
    regime_assessment_from_bytes,
)
from scpn_phase_orchestrator.reactor_semantics.producer_registry import (
    REACTOR_PRODUCER_REGISTRY_SHA256,
    REACTOR_PRODUCER_REGISTRY_VERSION,
    reactor_producer_registry_bytes,
)
from scpn_phase_orchestrator.reactor_semantics.producer_source import (
    MAX_PRODUCER_SOURCE_BYTES,
    ProducerSourcePolicy,
    producer_source_from_bytes,
    producer_source_schema,
    producer_source_to_bytes,
)

POLICY = ProducerSourcePolicy(
    "frc_compression_mif",
    "SCPN-MIF-CORE",
    "verified_review_adapter",
    "scpn-mif-core.merge-compression-observation.v1",
    "scpn-phase-orchestrator.mif-merge-compression-handoff.v1",
    REACTOR_PRODUCER_REGISTRY_VERSION,
    REACTOR_PRODUCER_REGISTRY_SHA256,
)
RELEASE = {
    "producer_registry_version": POLICY.producer_registry_version,
    "producer_registry_digest": POLICY.producer_registry_digest,
}


def _encode(record: object) -> bytes:
    """Encode test records with canonical sorted keys and compact separators."""
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


@pytest.fixture(scope="module")
def source_bytes() -> bytes:
    """Project the immutable MIF fixture through its public historical adapter."""
    raw = Path(
        "tests/fixtures/mif_merge_compression/mif_merge_compression_observation_v1.json"
    ).read_bytes()
    handoff = mif_merge_compression_handoff_from_mif_bytes(
        raw, registry=REACTOR_REGISTRY_V1_0_0
    )
    return mif_merge_compression_handoff_to_bytes(
        handoff, registry=REACTOR_REGISTRY_V1_0_0
    )


@pytest.fixture(scope="module")
def carrier(source_bytes: bytes) -> bytes:
    """Wrap the historical source with exact caller policy and source digest."""
    return producer_source_to_bytes(
        source_bytes,
        expected_source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        policy=POLICY,
    )


def test_historical_source_roundtrip_preserves_distinct_registry_scopes(
    source_bytes: bytes, carrier: bytes
) -> None:
    """Retain historical registry pins independently of current ownership pins."""
    result = producer_source_from_bytes(
        carrier, expected_sha256=hashlib.sha256(carrier).hexdigest(), policy=POLICY
    )
    assert result.source_bytes == source_bytes
    assert result.binding.source_registry_version == "1.0.0"
    assert json.loads(carrier)["reactor_registry_version"] == "1.1.0"
    assert result.binding.authority == "review_only"
    assert result.binding.actionable is False
    assert (
        producer_source_to_bytes(
            result.source_bytes,
            expected_source_sha256=result.binding.handoff_sha256,
            policy=result.policy,
        )
        == carrier
    )
    for decoder in (
        handoff_from_bytes,
        mif_merge_compression_handoff_from_bytes,
        regime_assessment_from_bytes,
    ):
        with pytest.raises(ValueError):
            decoder(carrier)


@pytest.mark.parametrize("route", ["mif", "fusion"])
def test_generated_schema_matches_registry_and_public_ingress_for_owner_role_matrix(
    carrier: bytes,
    route: str,
) -> None:
    """Compare structural schema and byte ingress across both historical routes."""
    selected = POLICY
    if route == "fusion":
        raw = Path(
            "tests/fixtures/producer_source/torax_runtime_review_envelope_v1.json"
        ).read_bytes()
        digest = "b594e2f8b72056426d628b638f6a849ef39e75daddc827305002b109365596c4"
        assert hashlib.sha256(raw).hexdigest() == digest
        source = handoff_to_bytes(
            coupled_transport_handoff_from_fusion_bytes(
                raw, expected_sha256=digest, registry=REACTOR_REGISTRY_V1_0_0
            ),
            registry=REACTOR_REGISTRY_V1_0_0,
        )
        selected = replace(
            POLICY,
            configuration="conventional_tokamak",
            producer_project="SCPN-FUSION-CORE",
            source_schema="scpn-fusion-core.torax-runtime-review-envelope.v1",
            handoff_schema="scpn-phase-orchestrator.reactor-semantic-handoff.v1",
        )
        carrier = producer_source_to_bytes(
            source,
            expected_source_sha256=hashlib.sha256(source).hexdigest(),
            policy=selected,
        )
    schema = producer_source_schema(**RELEASE)
    Draft202012Validator.check_schema(schema)
    assert (
        json.loads(Path("docs/specs/reactor_producer_source.schema.json").read_bytes())
        == schema
    )
    validator = Draft202012Validator(schema)
    registry = json.loads(
        reactor_producer_registry_bytes(
            version=POLICY.producer_registry_version,
            digest=POLICY.producer_registry_digest,
        )
    )
    producers = {entry["device_project"] for entry in registry["entries"]} | {
        "SCPN-FUSION-CORE",
        "SCPN-CONTROL",
        "SCPN-PHASE-ORCHESTRATOR",
        "UNREGISTERED",
    }
    original = json.loads(carrier)
    roles = (
        "verified_review_adapter",
        "physical_observable",
        "simulated_observable",
        "control_telemetry",
        "diagnostic_design_declaration",
        "semantic_derivation",
    )
    schemas = (
        POLICY.source_schema,
        "scpn-fusion-core.torax-runtime-review-envelope.v1",
        "unknown.v1",
    )
    for configuration, producer, role, source_schema in product(
        DEFAULT_REACTOR_REGISTRY.configurations, sorted(producers), roles, schemas
    ):
        record = {
            **original,
            "configuration": configuration,
            "producer_project": producer,
            "source_role": role,
            "source_schema": source_schema,
        }
        expected = (configuration, producer, role, source_schema) == (
            selected.configuration,
            selected.producer_project,
            selected.source_role,
            selected.source_schema,
        )
        assert validator.is_valid(record) is expected
        data = _encode(record)
        policy = replace(
            selected,
            configuration=configuration,
            producer_project=producer,
            source_role=role,
            source_schema=source_schema,
        )
        if expected:
            result = producer_source_from_bytes(
                data,
                expected_sha256=hashlib.sha256(data).hexdigest(),
                policy=policy,
            )
            assert result.source_bytes.decode() == original["source_json"]
        else:
            with pytest.raises(ValueError):
                producer_source_from_bytes(
                    data,
                    expected_sha256=hashlib.sha256(data).hexdigest(),
                    policy=policy,
                )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unknown.v2"),
        ("schema_version", "9.0.0"),
        ("device_project", "SCPN-CONTROL"),
        ("producer_registry_digest", "0" * 64),
        ("reactor_registry_digest", "0" * 64),
        ("semantic_profile_registry_digest", "0" * 64),
        ("reactor_registry_version", "9.0.0"),
        ("semantic_profile_registry_version", "9.0.0"),
        ("adapter_api", "unknown.adapter"),
        ("semantic_profile_version", "9.0.0"),
        ("source_schema", "unknown.v1"),
        ("handoff_schema", "unknown.v1"),
        ("authority", "execution"),
        ("actionable", True),
        ("actionable", 0),
        ("source_json", {}),
        ("source_json", "\ud800"),
        ("source_sha256", "0" * 64),
    ],
)
def test_resealed_metadata_and_custody_mutations_are_refused(
    carrier: bytes, field: str, value: object
) -> None:
    """Refuse changed metadata or inner bytes despite a matching outer hash."""
    record = json.loads(carrier)
    record[field] = value
    data = _encode(record)
    with pytest.raises(ValueError):
        producer_source_from_bytes(
            data, expected_sha256=hashlib.sha256(data).hexdigest(), policy=POLICY
        )


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "oversized",
        "duplicate",
        "nonfinite",
        "nested",
        "utf8",
        "noncanonical",
        "non-object",
        "missing",
        "extra",
    ],
)
def test_malformed_and_noncanonical_carriers_are_refused(
    carrier: bytes, kind: str
) -> None:
    """Reject malformed JSON, invalid nesting and noncanonical wire bytes."""
    record = json.loads(carrier)
    candidates = {
        "empty": b"",
        "oversized": b" " * (MAX_PRODUCER_SOURCE_BYTES + 1),
        "duplicate": b'{"schema":"a","schema":"b"}',
        "nonfinite": b'{"source_json":NaN}',
        "nested": b"[" * 2000,
        "utf8": b"\xff",
        "noncanonical": carrier + b"\n",
        "non-object": b"[]",
        "missing": b"{}",
        "extra": _encode({**record, "unexpected": "x"}),
    }
    data = candidates[kind]
    with pytest.raises(ValueError):
        producer_source_from_bytes(
            data, expected_sha256=hashlib.sha256(data).hexdigest(), policy=POLICY
        )


def test_writer_refuses_bad_custody_encoding_and_undeclared_policy(
    source_bytes: bytes, carrier: bytes
) -> None:
    """Reject invalid source bytes and caller policies before writing a carrier."""
    for raw in (b"\xff", b" " * (MAX_PRODUCER_SOURCE_BYTES + 1), b"{}"):
        with pytest.raises(ValueError):
            producer_source_to_bytes(
                raw,
                expected_source_sha256=hashlib.sha256(raw).hexdigest(),
                policy=POLICY,
            )
    for policy in (
        replace(POLICY, source_role="physical_observable"),
        replace(POLICY, producer_registry_version="9.0.0"),
        replace(POLICY, producer_registry_digest="0" * 64),
    ):
        with pytest.raises(ValueError):
            producer_source_to_bytes(
                source_bytes,
                expected_source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                policy=policy,
            )
    with pytest.raises(ValueError, match="digest mismatch"):
        producer_source_from_bytes(carrier, expected_sha256="0" * 64, policy=POLICY)
    with pytest.raises(ValueError):
        producer_source_from_bytes(carrier, expected_sha256="invalid", policy=POLICY)
