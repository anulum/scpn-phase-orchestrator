# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — FAIR-MAST magnetic source review tests
"""Exercise the exact physical-source-bytes to review-only SPO boundary."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator

import scpn_phase_orchestrator.reactor_semantics as rs
import scpn_phase_orchestrator.reactor_semantics.mast_magnetic_review as mast_module

FIXTURES = Path("tests/fixtures/mast_magnetic_source_review")
SOURCE_REVISION = "c30fb3932b47a812dc26d5846761030cdd0bc94c"
SOURCE_WHEEL_SHA256 = "a709b8aeecbd9483254bc3df1b29b87bf9df59ada92255af41631d861db430c9"
ARCHIVE_SHA256 = "6d4bf38305eeaab2e0d583877330763a32f2272420e0d191713b9ab734a613db"
QUALIFICATION_SHA256 = (
    "a8fbbf9149337d55590d4a382f264d8e9e16eadd6ad4c50fd01916e483c1270f"
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _source_records() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        json.loads((FIXTURES / "MAGNETIC_ARCHIVE_ENVELOPE.json").read_text()),
        json.loads((FIXTURES / "MAGNETIC_DIAGNOSTIC_QUALIFICATION.json").read_text()),
    )


def _reseal(document: dict[str, Any]) -> bytes:
    document["payload_sha256"] = hashlib.sha256(
        _canonical(document["payload"])
    ).hexdigest()
    return _canonical(document)


def _review(
    archive: dict[str, Any] | None = None,
    qualification: dict[str, Any] | None = None,
) -> rs.MastMagneticSourceReview:
    defaults = _source_records()
    return rs.mast_magnetic_source_review_from_producer_bytes(
        source_revision=SOURCE_REVISION,
        source_artifact_sha256=SOURCE_WHEEL_SHA256,
        archive_bytes=_reseal(deepcopy(defaults[0] if archive is None else archive)),
        qualification_bytes=_reseal(
            deepcopy(defaults[1] if qualification is None else qualification)
        ),
    )


def _assert_refusal(
    expected: rs.MastMagneticSourceRefusalCode,
    *,
    archive: dict[str, Any] | None = None,
    qualification: dict[str, Any] | None = None,
) -> None:
    with pytest.raises(rs.MastMagneticSourceRefusal) as caught:
        _review(archive, qualification)
    assert caught.value.code is expected
    assert caught.value.detail
    assert str(caught.value).startswith(f"{expected.value}: ")


def _archive_mutation(
    mutate: Callable[[dict[str, Any]], None],
    expected: rs.MastMagneticSourceRefusalCode,
) -> None:
    archive, _ = _source_records()
    mutate(archive["payload"])
    _assert_refusal(expected, archive=archive)


def _qualification_mutation(
    mutate: Callable[[dict[str, Any]], None],
    expected: rs.MastMagneticSourceRefusalCode,
) -> None:
    _, qualification = _source_records()
    mutate(qualification["payload"])
    _assert_refusal(expected, qualification=qualification)


def test_authentic_fair_mast_bytes_are_exact_and_produce_complete_review() -> None:
    archive_bytes = (FIXTURES / "MAGNETIC_ARCHIVE_ENVELOPE.json").read_bytes()
    qualification_bytes = (
        FIXTURES / "MAGNETIC_DIAGNOSTIC_QUALIFICATION.json"
    ).read_bytes()
    assert hashlib.sha256(archive_bytes).hexdigest() == ARCHIVE_SHA256
    assert hashlib.sha256(qualification_bytes).hexdigest() == QUALIFICATION_SHA256

    review = rs.mast_magnetic_source_review_from_producer_bytes(
        source_revision=SOURCE_REVISION,
        source_artifact_sha256=SOURCE_WHEEL_SHA256,
        archive_bytes=archive_bytes,
        qualification_bytes=qualification_bytes,
    )

    assert review.source_project == "SCPN-FUSION-CORE"
    assert review.device_project == "SCPN-TOKAMAK-CORE"
    assert review.facility == "MAST"
    assert review.configuration == "spherical_tokamak"
    assert review.source_archive == "FAIR-MAST"
    assert review.shot_id == 27707
    assert review.observation_id == "mast-27707-complete-magnetics-e7e9556899829607"
    assert review.source_revision == SOURCE_REVISION
    assert review.source_artifact_sha256 == SOURCE_WHEEL_SHA256
    assert review.source_archive_sha256 == ARCHIVE_SHA256
    assert review.source_qualification_sha256 == QUALIFICATION_SHA256
    assert (
        review.source_ingestion_revision == "ab435c799d892956fb042d55391f7d1be0c950e6"
    )
    assert review.source_ingestion_tree_state == "dirty"
    assert (review.array_count, review.measurement_count, review.channel_count) == (
        72,
        11,
        132,
    )
    assert tuple(item.name for item in review.clock_reviews) == (
        "time",
        "time_mirnov",
        "time_omaha",
        "time_saddle",
    )
    assert all(
        item.spo_clock_kind_candidate is rs.ClockKind.SHOT_RELATIVE
        and item.archive_grid_reproduced
        and not item.instrument_clock_relation_claimed
        and not item.mapping_evidence_claimed
        for item in review.clock_reviews
    )
    assert sum(item.channel_count for item in review.measurement_reviews) == 132
    assert all(
        item.source_valid_for_shot
        and item.applied_transform_recorded
        and not item.calibration_lineage_available
        and not item.observation_operator_available
        and not item.provider_quality_flags_supplied
        and not item.uncertainty_supplied
        and not item.phase_eligible
        for item in review.measurement_reviews
    )


def test_physical_source_custody_never_promotes_semantic_or_control_authority() -> None:
    review = _review()

    assert review.accepted_as_physical_source_review
    assert review.physical_source_recorded
    assert review.semantic_ingress_state == "not_declared"
    assert not review.observation_admitted
    assert not review.qualified_phase_evidence
    assert not review.phase_inference_performed
    assert not review.semantic_ingress_declared
    assert not review.classification_performed
    assert not review.control_intent_created
    assert not review.actionable
    assert not review.execution_permitted
    assert not review.direct_actuation
    assert review.review_only
    assert review.machine_protection_final_veto
    assert review.unresolved_qualification_fields == (
        "calibration_state",
        "channel_geometry_mapping_state",
        "event_identity_state",
        "observation_operator_state",
        "provider_quality_state",
        "source_clock_relationship_state",
        "uncertainty_state",
        "validity_state",
    )


def test_review_is_byte_canonical_digest_sealed_and_round_trips() -> None:
    review = _review()
    record = rs.mast_magnetic_source_review_to_record(review)
    encoded = rs.mast_magnetic_source_review_to_bytes(review)

    assert encoded.endswith(b"\n")
    assert encoded == _canonical(json.loads(encoded))
    assert rs.mast_magnetic_source_review_from_record(record) == review
    assert rs.mast_magnetic_source_review_from_bytes(encoded) == review
    assert (
        rs.mast_magnetic_source_review_digest(review)
        == hashlib.sha256(encoded).hexdigest()
    )
    assert len(review.review_id) == 64


def test_portable_review_matches_its_published_json_schema() -> None:
    schema = json.loads(
        Path("docs/specs/mast_magnetic_source_review.schema.json").read_text()
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(
        json.loads(rs.mast_magnetic_source_review_to_bytes(_review()))
    )


@pytest.mark.parametrize("revision", ["", "A" * 40, "0" * 39, 1])
def test_source_revision_requires_exact_lowercase_git_sha(revision: object) -> None:
    archive, qualification = _source_records()
    with pytest.raises(rs.MastMagneticSourceRefusal) as caught:
        rs.mast_magnetic_source_review_from_producer_bytes(
            source_revision=cast(str, revision),
            source_artifact_sha256=SOURCE_WHEEL_SHA256,
            archive_bytes=_reseal(archive),
            qualification_bytes=_reseal(qualification),
        )
    assert (
        caught.value.code is rs.MastMagneticSourceRefusalCode.SOURCE_IDENTITY_MISMATCH
    )


@pytest.mark.parametrize("digest", ["", "A" * 64, "0" * 63, 1])
def test_source_artifact_requires_exact_lowercase_sha256(digest: object) -> None:
    archive, qualification = _source_records()
    with pytest.raises(rs.MastMagneticSourceRefusal) as caught:
        rs.mast_magnetic_source_review_from_producer_bytes(
            source_revision=SOURCE_REVISION,
            source_artifact_sha256=cast(str, digest),
            archive_bytes=_reseal(archive),
            qualification_bytes=_reseal(qualification),
        )
    assert (
        caught.value.code is rs.MastMagneticSourceRefusalCode.SOURCE_IDENTITY_MISMATCH
    )


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (cast(bytes, "text"), rs.MastMagneticSourceRefusalCode.INVALID_INPUT),
        (b"", rs.MastMagneticSourceRefusalCode.INVALID_INPUT),
        (b"\xff", rs.MastMagneticSourceRefusalCode.INVALID_JSON),
        (b"{", rs.MastMagneticSourceRefusalCode.INVALID_JSON),
        (b'{"schema":NaN}\n', rs.MastMagneticSourceRefusalCode.INVALID_JSON),
        (
            b'{"payload":{},"payload_sha256":"x","schema":"a",'
            b'"schema":"b","schema_version":"1"}\n',
            rs.MastMagneticSourceRefusalCode.DUPLICATE_JSON_KEY,
        ),
        (
            b'{ "payload": {}, "payload_sha256": "x", "schema": "x", '
            b'"schema_version": "x" }\n',
            rs.MastMagneticSourceRefusalCode.NONCANONICAL_BYTES,
        ),
    ],
)
def test_source_documents_refuse_invalid_transport_bytes(
    data: bytes, code: rs.MastMagneticSourceRefusalCode
) -> None:
    _, qualification = _source_records()
    with pytest.raises(rs.MastMagneticSourceRefusal) as caught:
        rs.mast_magnetic_source_review_from_producer_bytes(
            source_revision=SOURCE_REVISION,
            source_artifact_sha256=SOURCE_WHEEL_SHA256,
            archive_bytes=data,
            qualification_bytes=_reseal(qualification),
        )
    assert caught.value.code is code


def test_source_document_size_limit_is_fail_closed() -> None:
    _, qualification = _source_records()
    with pytest.raises(rs.MastMagneticSourceRefusal) as caught:
        rs.mast_magnetic_source_review_from_producer_bytes(
            source_revision=SOURCE_REVISION,
            source_artifact_sha256=SOURCE_WHEEL_SHA256,
            archive_bytes=b"x" * (rs.MAX_MAST_MAGNETIC_SOURCE_BYTES + 1),
            qualification_bytes=_reseal(qualification),
        )
    assert caught.value.code is rs.MastMagneticSourceRefusalCode.INVALID_INPUT


def test_source_schema_payload_keys_and_digest_are_sealed() -> None:
    archive, qualification = _source_records()

    archive["schema_version"] = "2.0.0"
    with pytest.raises(rs.MastMagneticSourceRefusal) as unsupported:
        rs.mast_magnetic_source_review_from_producer_bytes(
            source_revision=SOURCE_REVISION,
            source_artifact_sha256=SOURCE_WHEEL_SHA256,
            archive_bytes=_canonical(archive),
            qualification_bytes=_reseal(qualification),
        )
    assert unsupported.value.code is rs.MastMagneticSourceRefusalCode.UNSUPPORTED_SCHEMA

    archive, qualification = _source_records()
    archive["payload_sha256"] = "0" * 64
    with pytest.raises(rs.MastMagneticSourceRefusal) as digest:
        rs.mast_magnetic_source_review_from_producer_bytes(
            source_revision=SOURCE_REVISION,
            source_artifact_sha256=SOURCE_WHEEL_SHA256,
            archive_bytes=_canonical(archive),
            qualification_bytes=_reseal(qualification),
        )
    assert digest.value.code is rs.MastMagneticSourceRefusalCode.SOURCE_DIGEST_MISMATCH

    archive, _ = _source_records()
    archive["payload"]["unexpected"] = False
    _assert_refusal(
        rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        archive=archive,
    )


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda p: p.__setitem__("producer_project", "other"),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p.__setitem__(
                "authority", {**p["authority"], "actionable": True}
            ),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p.__setitem__(
                "qualification", {**p["qualification"], "phase_eligible": True}
            ),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p.__setitem__("observation_id", "not-shot-bound"),
            rs.MastMagneticSourceRefusalCode.ARCHIVE_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p.__setitem__("event_id", "invented"),
            rs.MastMagneticSourceRefusalCode.AUTHORITY_ESCALATION,
        ),
        (
            lambda p: p.__setitem__("source_ingestion_revision", "bad"),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p.__setitem__("source_ingestion_tree_state", "unknown"),
            rs.MastMagneticSourceRefusalCode.ARCHIVE_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["completeness"].__setitem__("array_count", 71),
            rs.MastMagneticSourceRefusalCode.ARCHIVE_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["arrays"].__setitem__(1, deepcopy(p["arrays"][0])),
            rs.MastMagneticSourceRefusalCode.ARCHIVE_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["arrays"].pop(),
            rs.MastMagneticSourceRefusalCode.ARCHIVE_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p.__setitem__("arrays", {}),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["arrays"][0].__setitem__("decoded_content_sha256", "bad"),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["arrays"][0].__setitem__("decoded_nonfinite_count", -1),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["arrays"][0].__setitem__("shape", [True]),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["arrays"][0].__setitem__("dimension_names", [""]),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["clocks"].__setitem__(1, deepcopy(p["clocks"][0])),
            rs.MastMagneticSourceRefusalCode.ARCHIVE_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["clocks"][0].__setitem__("mapping_evidence_claimed", True),
            rs.MastMagneticSourceRefusalCode.AUTHORITY_ESCALATION,
        ),
        (
            lambda p: p["clocks"][0].__setitem__("sample_count", 0),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["clocks"][0].__setitem__("first_value_s", "bad"),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["clocks"].pop(),
            rs.MastMagneticSourceRefusalCode.ARCHIVE_CONTRACT_MISMATCH,
        ),
    ],
)
def test_archive_contract_mutations_fail_closed(
    mutate: Callable[[dict[str, Any]], None],
    code: rs.MastMagneticSourceRefusalCode,
) -> None:
    _archive_mutation(mutate, code)


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda p: p.__setitem__(
                "authority", {**p["authority"], "phase_inference_performed": True}
            ),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p.__setitem__(
                "qualification_summary",
                {**p["qualification_summary"], "uncertainty_state": "known"},
            ),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p.__setitem__("shot_id", 1),
            rs.MastMagneticSourceRefusalCode.CROSS_SOURCE_MISMATCH,
        ),
        (
            lambda p: p.__setitem__("archive_observation_id", "other"),
            rs.MastMagneticSourceRefusalCode.CROSS_SOURCE_MISMATCH,
        ),
        (
            lambda p: p.__setitem__("archive_envelope_sha256", "0" * 64),
            rs.MastMagneticSourceRefusalCode.CROSS_SOURCE_MISMATCH,
        ),
        (
            lambda p: p["event_identity"].__setitem__("event_id", "invented"),
            rs.MastMagneticSourceRefusalCode.AUTHORITY_ESCALATION,
        ),
        (
            lambda p: p["ingestion_mapping"].__setitem__("source_revision", "0" * 40),
            rs.MastMagneticSourceRefusalCode.CROSS_SOURCE_MISMATCH,
        ),
        (
            lambda p: p["ingestion_mapping"].__setitem__("mapping_sha256", "bad"),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["array_inventory"].pop(),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["array_inventory"].__setitem__(
                1, deepcopy(p["array_inventory"][0])
            ),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["array_inventory"][0].__setitem__("role", "geometry"),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["array_inventory"][0].__setitem__("name", "absent"),
            rs.MastMagneticSourceRefusalCode.CROSS_SOURCE_MISMATCH,
        ),
        (
            lambda p: p["array_inventory"][0].__setitem__("shape", [4]),
            rs.MastMagneticSourceRefusalCode.CROSS_SOURCE_MISMATCH,
        ),
        (
            lambda p: p["clock_evidence"][0].__setitem__("name", "absent"),
            rs.MastMagneticSourceRefusalCode.CROSS_SOURCE_MISMATCH,
        ),
        (
            lambda p: p["clock_evidence"][0].__setitem__(
                "source_clock_relation_claimed", True
            ),
            rs.MastMagneticSourceRefusalCode.AUTHORITY_ESCALATION,
        ),
        (
            lambda p: p["clock_evidence"][0].__setitem__("step_s", -1),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["clock_evidence"][0].__setitem__("sample_count", 1),
            rs.MastMagneticSourceRefusalCode.CROSS_SOURCE_MISMATCH,
        ),
        (
            lambda p: p["clock_evidence"].pop(),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["measurement_evidence"][0].__setitem__("clock_name", "absent"),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["measurement_evidence"][0].__setitem__(
                "uncertainty_supplied", True
            ),
            rs.MastMagneticSourceRefusalCode.AUTHORITY_ESCALATION,
        ),
        (
            lambda p: p["measurement_evidence"][0].__setitem__(
                "archive_channel_ids", ["201", "201"]
            ),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["measurement_evidence"][0]["channel_quality"].pop(),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["measurement_evidence"][0]["channel_quality"][0].__setitem__(
                "archive_channel_id", "absent"
            ),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["measurement_evidence"][0]["empirical_quality"].__setitem__(
                "nan_fraction", 2.0
            ),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["measurement_evidence"][0]["empirical_quality"].__setitem__(
                "minimum_positive_level_spacing_hex", 1
            ),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["measurement_evidence"].pop(),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["channel_geometry_evidence"][0].__setitem__(
                "measurement_array", "absent"
            ),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["channel_geometry_evidence"][0].__setitem__(
                "physical_mapping_claimed", True
            ),
            rs.MastMagneticSourceRefusalCode.AUTHORITY_ESCALATION,
        ),
        (
            lambda p: p["completeness"].__setitem__("channel_record_count", 131),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p.__setitem__("external_limitations", []),
            rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        ),
        (
            lambda p: p["external_limitations"][0].__setitem__(
                "applicability_to_shot", "assumed"
            ),
            rs.MastMagneticSourceRefusalCode.AUTHORITY_ESCALATION,
        ),
    ],
)
def test_qualification_contract_mutations_fail_closed(
    mutate: Callable[[dict[str, Any]], None],
    code: rs.MastMagneticSourceRefusalCode,
) -> None:
    _qualification_mutation(mutate, code)


def test_review_envelope_and_reconstructed_fields_are_fail_closed() -> None:
    review = _review()
    encoded = json.loads(rs.mast_magnetic_source_review_to_bytes(review))

    encoded["schema_version"] = "2.0.0"
    with pytest.raises(rs.MastMagneticSourceRefusal) as schema:
        rs.mast_magnetic_source_review_from_bytes(_canonical(encoded))
    assert schema.value.code is rs.MastMagneticSourceRefusalCode.UNSUPPORTED_SCHEMA

    encoded = json.loads(rs.mast_magnetic_source_review_to_bytes(review))
    encoded["payload_sha256"] = "0" * 64
    with pytest.raises(rs.MastMagneticSourceRefusal) as digest:
        rs.mast_magnetic_source_review_from_bytes(_canonical(encoded))
    assert digest.value.code is rs.MastMagneticSourceRefusalCode.SOURCE_DIGEST_MISMATCH

    record = rs.mast_magnetic_source_review_to_record(review)
    record["shot_id"] = 1
    with pytest.raises(rs.MastMagneticSourceRefusal) as replay:
        rs.mast_magnetic_source_review_from_record(record)
    assert replay.value.code is rs.MastMagneticSourceRefusalCode.SOURCE_DIGEST_MISMATCH


def test_invalid_object_and_empty_text_fields_are_refused() -> None:
    archive, qualification = _source_records()
    archive["payload"] = []
    with pytest.raises(rs.MastMagneticSourceRefusal) as invalid_object:
        rs.mast_magnetic_source_review_from_producer_bytes(
            source_revision=SOURCE_REVISION,
            source_artifact_sha256=SOURCE_WHEEL_SHA256,
            archive_bytes=_canonical(archive),
            qualification_bytes=_reseal(qualification),
        )
    assert invalid_object.value.code is (
        rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH
    )

    archive, _ = _source_records()
    archive["payload"]["observation_id"] = ""
    _assert_refusal(
        rs.MastMagneticSourceRefusalCode.QUALIFICATION_CONTRACT_MISMATCH,
        archive=archive,
    )


def test_registry_drift_is_refused_through_public_ingress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drifted = SimpleNamespace(
        device_project="wrong",
        producer_project=None,
        ingress_state=SimpleNamespace(value="not_declared"),
        actionable=False,
        machine_protection_final_veto=True,
    )
    monkeypatch.setattr(
        mast_module,
        "DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY",
        SimpleNamespace(resolve=lambda _configuration: drifted),
    )
    with pytest.raises(rs.MastMagneticSourceRefusal) as caught:
        _review()
    assert caught.value.code is (
        rs.MastMagneticSourceRefusalCode.REGISTRY_ASSIGNMENT_MISMATCH
    )


def test_review_byte_transport_refuses_noncanonical_and_oversize_documents() -> None:
    review = _review()
    encoded = rs.mast_magnetic_source_review_to_bytes(review)
    with pytest.raises(rs.MastMagneticSourceRefusal) as noncanonical:
        rs.mast_magnetic_source_review_from_bytes(b" " + encoded)
    assert (
        noncanonical.value.code is rs.MastMagneticSourceRefusalCode.NONCANONICAL_BYTES
    )

    with pytest.raises(rs.MastMagneticSourceRefusal) as oversized:
        rs.mast_magnetic_source_review_from_bytes(
            b"x" * (rs.MAX_MAST_MAGNETIC_SOURCE_REVIEW_BYTES + 1)
        )
    assert oversized.value.code is rs.MastMagneticSourceRefusalCode.INVALID_INPUT
