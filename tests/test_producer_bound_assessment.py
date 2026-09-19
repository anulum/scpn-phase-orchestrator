# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Producer-bound assessment public wire tests

"""Check original source custody and independently re-sealed crosslinks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from scpn_phase_orchestrator.reactor_semantics import (
    DEFAULT_REACTOR_REGISTRY,
    REACTOR_REGISTRY_V1_0_0,
    ReactorConfigurationRegistry,
    build_abstaining_regime_assessment,
    handoff_from_bytes,
    mif_merge_compression_handoff_from_bytes,
    mif_merge_compression_handoff_from_mif_bytes,
    mif_merge_compression_handoff_to_bytes,
    regime_assessment_from_bytes,
    regime_assessment_to_bytes,
)
from scpn_phase_orchestrator.reactor_semantics.producer_bound_assessment import (
    MAX_PRODUCER_BOUND_ASSESSMENT_BYTES,
    producer_bound_assessment_from_bytes,
    producer_bound_assessment_schema,
    producer_bound_assessment_to_bytes,
)
from scpn_phase_orchestrator.reactor_semantics.producer_registry import (
    REACTOR_PRODUCER_REGISTRY_SHA256,
    REACTOR_PRODUCER_REGISTRY_VERSION,
)

REGISTRY_POLICY = {
    "producer_registry_version": REACTOR_PRODUCER_REGISTRY_VERSION,
    "producer_registry_digest": REACTOR_PRODUCER_REGISTRY_SHA256,
}
POLICY = {**REGISTRY_POLICY, "configuration": "frc_compression_mif"}


def _encode(record: object) -> bytes:
    """Encode test records with canonical sorted keys and compact separators."""
    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _carrier(registry: ReactorConfigurationRegistry) -> bytes:
    """Build a historical source and assessment pair through public codecs."""
    source = Path(
        "tests/fixtures/mif_merge_compression/mif_merge_compression_observation_v1.json"
    ).read_bytes()
    handoff = mif_merge_compression_handoff_from_mif_bytes(source, registry=registry)
    handoff_bytes = mif_merge_compression_handoff_to_bytes(handoff, registry=registry)
    assessment = build_abstaining_regime_assessment(
        handoff, producer_revision="a" * 40, producer_artifact_sha256="b" * 64
    )
    assessment_bytes = regime_assessment_to_bytes(assessment)
    return producer_bound_assessment_to_bytes(
        handoff_bytes,
        assessment_bytes,
        source_handoff_schema=handoff.schema,
        expected_source_sha256=hashlib.sha256(handoff_bytes).hexdigest(),
        expected_assessment_sha256=hashlib.sha256(assessment_bytes).hexdigest(),
        **POLICY,
    )


@pytest.mark.parametrize(
    "registry", [REACTOR_REGISTRY_V1_0_0, DEFAULT_REACTOR_REGISTRY]
)
def test_public_roundtrip_preserves_independent_registry_scopes(
    registry: ReactorConfigurationRegistry,
) -> None:
    """Preserve independently selected source and assessment registry scopes."""
    data = _carrier(registry)
    if registry == REACTOR_REGISTRY_V1_0_0:
        assert hashlib.sha256(data).hexdigest() == (
            "49ba55a7a40ed9e861b854a6b7aaaaf69e560cededd9143c7967dc2b5843651a"
        )
    decoded = producer_bound_assessment_from_bytes(
        data, expected_sha256=hashlib.sha256(data).hexdigest(), **POLICY
    )
    assert decoded.binding.source_registry_digest == registry.digest
    assert decoded.assessment.reactor_registry_digest == DEFAULT_REACTOR_REGISTRY.digest
    assert (
        decoded.assessment.source_handoff_sha256
        == hashlib.sha256(decoded.source_bytes).hexdigest()
    )
    assert decoded.binding.actionable is decoded.assessment.actionable is False
    assert (
        producer_bound_assessment_to_bytes(
            decoded.source_bytes,
            decoded.assessment_bytes,
            source_handoff_schema=decoded.binding.handoff_schema,
            expected_source_sha256=decoded.binding.handoff_sha256,
            expected_assessment_sha256=hashlib.sha256(
                decoded.assessment_bytes
            ).hexdigest(),
            **POLICY,
        )
        == data
    )
    Draft202012Validator(producer_bound_assessment_schema(**REGISTRY_POLICY)).validate(
        json.loads(data)
    )
    for legacy in (
        handoff_from_bytes,
        mif_merge_compression_handoff_from_bytes,
        regime_assessment_from_bytes,
    ):
        with pytest.raises(ValueError):
            legacy(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("event_id", "mif.other.event"),
        ("reactor_context_id", "mif.other.context"),
        ("source_revision", "c" * 40),
        ("source_project", "SCPN-FUSION-CORE"),
        (
            "source_handoff_schema",
            "scpn-phase-orchestrator.reactor-semantic-handoff.v1",
        ),
        ("source_handoff_sha256", "d" * 64),
        ("source_semantic_ids", ["mif.other.semantic"]),
    ],
)
def test_resealed_valid_assessment_cannot_change_source_crosslinks(
    field: str, value: object
) -> None:
    """Reject independently valid assessments with changed source lineage."""
    outer = json.loads(_carrier(REACTOR_REGISTRY_V1_0_0))
    inner = json.loads(outer["assessment_json"])
    inner["payload"][field] = value
    inner["payload_sha256"] = hashlib.sha256(_encode(inner["payload"])).hexdigest()
    changed = _encode(inner)
    regime_assessment_from_bytes(changed)
    outer["assessment_json"] = changed.decode()
    outer["assessment_sha256"] = hashlib.sha256(changed).hexdigest()
    data = _encode(outer)
    with pytest.raises(ValueError, match=f"crosslink mismatch: {field}"):
        producer_bound_assessment_from_bytes(
            data, expected_sha256=hashlib.sha256(data).hexdigest(), **POLICY
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "2.0.0"),
        ("schema", "unknown.v1"),
        ("producer_registry_digest", "0" * 64),
        ("producer_registry_version", "9.0.0"),
        ("configuration", "conventional_tokamak"),
        ("actionable", True),
        ("authority", "actuation"),
        ("assessment_sha256", "0" * 64),
        ("source_handoff_sha256", "0" * 64),
        ("source_handoff_schema", "unknown.v1"),
        ("assessment_json", "{}"),
        ("source_handoff_json", "{}"),
    ],
)
def test_resealed_outer_contract_mutations_refuse(field: str, value: object) -> None:
    """Refuse outer identity changes even when the carrier digest is resealed."""
    outer = json.loads(_carrier(REACTOR_REGISTRY_V1_0_0))
    outer[field] = value
    data = _encode(outer)
    with pytest.raises(ValueError):
        producer_bound_assessment_from_bytes(
            data, expected_sha256=hashlib.sha256(data).hexdigest(), **POLICY
        )


def test_canonical_encoding_bounds_and_flat_nesting() -> None:
    """Enforce canonical flat JSON and the carrier byte-size limit."""
    data = _carrier(REACTOR_REGISTRY_V1_0_0)
    candidates = [
        b"",
        b"\xff",
        b"{}",
        b"[[]]",
        b'{"nested":' + b"[" * 2000,
        b'{"x":1,"x":2}',
        data + b"\n",
        b" " * (MAX_PRODUCER_BOUND_ASSESSMENT_BYTES + 1),
        data.replace(b'"actionable":false', b'"actionable":NaN'),
        data.replace(b'"schema_version":"1.0.0"', b'"schema_version":1'),
    ]
    for candidate in candidates:
        with pytest.raises(ValueError):
            producer_bound_assessment_from_bytes(
                candidate,
                expected_sha256=hashlib.sha256(candidate).hexdigest(),
                **POLICY,
            )
    with pytest.raises(ValueError, match="digest mismatch"):
        producer_bound_assessment_from_bytes(data, expected_sha256="0" * 64, **POLICY)
    record = json.loads(data)
    record["source_handoff_json"] = "\ud800"
    invalid_text = _encode(record)
    with pytest.raises(ValueError, match="inner document encoding"):
        producer_bound_assessment_from_bytes(
            invalid_text,
            expected_sha256=hashlib.sha256(invalid_text).hexdigest(),
            **POLICY,
        )


def test_every_configuration_obeys_historical_outer_schema_binding() -> None:
    """Compare historical schema availability for every configuration."""
    outer = json.loads(_carrier(REACTOR_REGISTRY_V1_0_0))
    validator = Draft202012Validator(
        producer_bound_assessment_schema(**REGISTRY_POLICY)
    )
    for configuration in DEFAULT_REACTOR_REGISTRY.configurations:
        record = {**outer, "configuration": configuration}
        data = _encode(record)
        if configuration == "frc_compression_mif":
            assert validator.is_valid(record)
            producer_bound_assessment_from_bytes(
                data,
                expected_sha256=hashlib.sha256(data).hexdigest(),
                **{**POLICY, "configuration": configuration},
            )
        else:
            assert not validator.is_valid(record)
            with pytest.raises(ValueError):
                producer_bound_assessment_from_bytes(
                    data,
                    expected_sha256=hashlib.sha256(data).hexdigest(),
                    **{**POLICY, "configuration": configuration},
                )


def test_generated_schema_file_matches_runtime() -> None:
    """Keep the shipped assessment schema equal to the public generator."""
    schema = json.loads(
        Path("docs/specs/producer_bound_historical_assessment.schema.json").read_bytes()
    )
    assert schema == producer_bound_assessment_schema(**REGISTRY_POLICY)


def test_builder_rejects_invalid_inner_encoding() -> None:
    """Refuse invalid UTF-8 in either preserved inner document."""
    outer = json.loads(_carrier(REACTOR_REGISTRY_V1_0_0))
    for source in (b"\xff", b" " * (MAX_PRODUCER_BOUND_ASSESSMENT_BYTES + 1)):
        with pytest.raises(ValueError):
            producer_bound_assessment_to_bytes(
                source,
                outer["assessment_json"].encode(),
                source_handoff_schema=outer["source_handoff_schema"],
                expected_source_sha256=outer["source_handoff_sha256"],
                expected_assessment_sha256=outer["assessment_sha256"],
                **POLICY,
            )
