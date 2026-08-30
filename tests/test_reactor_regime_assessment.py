# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor regime assessment tests

"""Portable, fail-closed, eight-axis regime-assessment contract tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from scpn_phase_orchestrator.reactor_semantics import (
    DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY,
    DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY,
    DEFAULT_REACTOR_REGISTRY,
    DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY,
    MAX_REACTOR_REGIME_ASSESSMENT_BYTES,
    REACTOR_REGIME_ASSESSMENT_SCHEMA,
    REACTOR_REGIME_ASSESSMENT_VERSION,
    REVIEW_ONLY_AUTHORITY,
    AxisApplicability,
    ClockKind,
    EvidenceClass,
    QualityState,
    ReactorRegimeAssessment,
    ReactorRegimeAxisAssessment,
    ReactorRegimeAxisDisposition,
    ReactorRegimeEvidenceBinding,
    ValidityState,
    regime_assessment_digest,
    regime_assessment_from_bytes,
    regime_assessment_from_record,
    regime_assessment_to_bytes,
    regime_assessment_to_record,
)

SCHEMA_PATH = Path("docs/specs/reactor_regime_assessment.schema.json")
CONFIGURATION = "frc_compression_mif"


def _bindings(axis_id: str) -> tuple[ReactorRegimeEvidenceBinding, ...]:
    definition = DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.resolve_axis(axis_id)
    return tuple(
        ReactorRegimeEvidenceBinding(
            role_id=role,
            reference_id=f"mif.test.reference.{axis_id}.{role}",
        )
        for role in sorted(definition.required_evidence)
    )


def _axis(
    axis_id: str,
    *,
    classified: bool = False,
    **changes: object,
) -> ReactorRegimeAxisAssessment:
    ontology = DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY
    static = ontology.applicability_for(axis_id, CONFIGURATION)
    fields: dict[str, object] = {
        "axis_id": axis_id,
        "static_applicability": static,
        "disposition": ReactorRegimeAxisDisposition.UNKNOWN,
        "label": None,
        "confidence": 0.0,
        "observability": 0.0,
        "uncertainty_probability": 1.0,
        "uncertainty_basis_id": "mif.test.uncertainty.unqualified",
        "evidence_ids": (),
        "evidence_bindings": (),
        "evidence_class": EvidenceClass.UNKNOWN,
        "validity": ValidityState.UNKNOWN,
        "quality": QualityState.UNKNOWN,
        "validity_id": f"mif.test.validity.{axis_id}",
        "quality_id": f"mif.test.quality.{axis_id}",
        "provenance_id": f"mif.test.provenance.{axis_id}",
        "applicability_basis": (),
        "unknown_reason_id": f"mif.test.unknown.no-qualified-evidence.{axis_id}",
        "classifier_id": None,
        "classifier_version": None,
        "classifier_sha256": None,
        "threshold_policy_id": None,
        "threshold_policy_version": None,
        "threshold_policy_sha256": None,
        "hysteresis_policy_id": None,
        "hysteresis_policy_version": None,
        "hysteresis_policy_sha256": None,
        "dwell_samples": None,
    }
    if static is AxisApplicability.NOT_APPLICABLE:
        fields.update(
            disposition=ReactorRegimeAxisDisposition.NOT_APPLICABLE,
            uncertainty_probability=0.0,
            uncertainty_basis_id=None,
            evidence_class=EvidenceClass.REVIEW_HYPOTHESIS,
            validity=ValidityState.VALID,
            quality=QualityState.VALID,
            applicability_basis=(
                f"spo.ontology.{axis_id}.configuration-not-applicable",
            ),
            unknown_reason_id=None,
        )
    if classified:
        definition = ontology.resolve_axis(axis_id)
        label = next(item for item in definition.labels if item != "unknown")
        fields.update(
            disposition=ReactorRegimeAxisDisposition.CLASSIFIED,
            label=label,
            confidence=0.7,
            observability=0.8,
            uncertainty_probability=0.2,
            uncertainty_basis_id=f"mif.test.uncertainty.{axis_id}",
            evidence_ids=(f"mif.test.evidence.{axis_id}",),
            evidence_bindings=_bindings(axis_id),
            evidence_class=EvidenceClass.SIMULATION,
            validity=ValidityState.VALID,
            quality=QualityState.VALID,
            unknown_reason_id=None,
        )
        if "classifier" in definition.required_evidence:
            fields.update(
                classifier_id=f"mif.test.classifier.{axis_id}",
                classifier_version="1.0.0",
                classifier_sha256="1" * 64,
            )
        if "threshold_provenance" in definition.required_evidence:
            fields.update(
                threshold_policy_id=f"mif.test.threshold.{axis_id}",
                threshold_policy_version="1.0.0",
                threshold_policy_sha256="2" * 64,
                hysteresis_policy_id=f"mif.test.hysteresis.{axis_id}",
                hysteresis_policy_version="1.0.0",
                hysteresis_policy_sha256="3" * 64,
                dwell_samples=4,
            )
    fields.update(changes)
    return ReactorRegimeAxisAssessment(**fields)  # type: ignore[arg-type]


def _assessment(**changes: object) -> ReactorRegimeAssessment:
    axis_ids = sorted(DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.axes)
    axes = tuple(_axis(axis_id) for axis_id in axis_ids)
    fields: dict[str, object] = {
        "assessment_id": "spo.assessment.mif.0001",
        "reactor_context_id": "mif.frc.merge-compression.context.0001",
        "configuration": CONFIGURATION,
        "event_id": "mif.frc.event.0001",
        "producer_project": "SCPN-PHASE-ORCHESTRATOR",
        "producer_revision": "a" * 40,
        "producer_artifact_sha256": "b" * 64,
        "source_project": "SCPN-MIF-CORE",
        "source_revision": "c" * 40,
        "source_handoff_schema": "scpn-mif-core.merge-compression-observation.v1",
        "source_handoff_sha256": "d" * 64,
        "source_semantic_ids": (
            "mif.frc.semantic.merge-window",
            "mif.frc.semantic.numerical-phase",
        ),
        "clock_domain": "mif.simulation.monotonic",
        "clock_kind": ClockKind.SIMULATION_MONOTONIC,
        "clock_epoch": "mif.frc.event.0001.start",
        "clock_synchronization_id": "mif.simulation.clock-correlation.0001",
        "evidence_timestamp_ns": 10_000,
        "assessed_at_ns": 20_000,
        "valid_from_ns": 9_000,
        "valid_until_ns": 30_000,
        "sample_rate_hz": 100_000.0,
        "latency_s": 0.0,
        "timestamp_offset_ps": 0,
        "axes": axes,
        "reactor_registry_version": DEFAULT_REACTOR_REGISTRY.version,
        "reactor_registry_digest": DEFAULT_REACTOR_REGISTRY.digest,
        "semantic_profile_registry_version": (
            DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.version
        ),
        "semantic_profile_registry_digest": (
            DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.digest
        ),
        "observability_registry_version": (
            DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.version
        ),
        "observability_registry_digest": (
            DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.digest
        ),
        "ontology_version": DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.version,
        "ontology_digest": DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.digest,
    }
    fields.update(changes)
    return ReactorRegimeAssessment(**fields)  # type: ignore[arg-type]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _reseal(record: dict[str, Any]) -> dict[str, Any]:
    record["payload_sha256"] = hashlib.sha256(_canonical(record["payload"])).hexdigest()
    return record


def test_public_round_trip_digest_schema_and_authority() -> None:
    assessment = _assessment()
    record = regime_assessment_to_record(assessment)
    encoded = regime_assessment_to_bytes(assessment)

    assert regime_assessment_from_record(record) == assessment
    assert regime_assessment_from_bytes(encoded) == assessment
    assert regime_assessment_digest(assessment) == hashlib.sha256(encoded).hexdigest()
    assert record["schema"] == REACTOR_REGIME_ASSESSMENT_SCHEMA
    assert record["schema_version"] == REACTOR_REGIME_ASSESSMENT_VERSION
    assert assessment.authority == REVIEW_ONLY_AUTHORITY
    assert assessment.actionable is False
    assert assessment.classification_performed is False
    assert len(assessment.axes) == 8

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(record)


def test_axis_construction_distinguishes_all_three_dispositions() -> None:
    readiness = _axis("plant_readiness", classified=True)
    stability = _axis("stability_or_symmetry", classified=True)
    exhaust = _axis("exhaust_or_boundary")

    assert readiness.disposition is ReactorRegimeAxisDisposition.CLASSIFIED
    assert readiness.classifier_id is None
    assert readiness.dwell_samples is None
    assert stability.classifier_id is None
    assert stability.threshold_policy_id is not None
    assert stability.dwell_samples == 4
    assert exhaust.static_applicability is AxisApplicability.NOT_APPLICABLE
    assert exhaust.disposition is ReactorRegimeAxisDisposition.NOT_APPLICABLE


def test_classifier_axis_requires_exact_classifier_and_threshold_policy() -> None:
    axis = _axis("confinement_or_assembly", classified=True)

    assert axis.classifier_id is not None
    assert axis.threshold_policy_id is not None
    assert axis.dwell_samples == 4
    assert tuple(item.role_id for item in axis.evidence_bindings) == tuple(
        sorted(
            DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.resolve_axis(
                "confinement_or_assembly"
            ).required_evidence
        )
    )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"role_id": "bad role"}, "namespaced identifier"),
        ({"reference_id": "bad ref"}, "namespaced identifier"),
    ],
)
def test_evidence_binding_refuses_invalid_identity(
    changes: dict[str, object],
    match: str,
) -> None:
    fields = {"role_id": "validity", "reference_id": "mif.test.validity.1"}
    fields.update(changes)
    with pytest.raises(ValueError, match=match):
        ReactorRegimeEvidenceBinding(**fields)  # type: ignore[arg-type]


def test_evidence_binding_strict_record_round_trip() -> None:
    binding = ReactorRegimeEvidenceBinding("validity", "mif.test.validity.1")

    assert ReactorRegimeEvidenceBinding.from_record(binding.to_record()) == binding
    with pytest.raises(ValueError, match="missing fields"):
        ReactorRegimeEvidenceBinding.from_record({"role_id": "validity"})


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"static_applicability": AxisApplicability.UNKNOWN}, "cannot be unknown"),
        ({"disposition": "unknown"}, "must be a ReactorRegimeAxisDisposition"),
        ({"confidence": 2.0}, "must be in"),
        ({"observability": -1.0}, "must be in"),
        ({"uncertainty_probability": 2.0}, "must be in"),
        ({"evidence_ids": ("z", "a")}, "unique and sorted"),
        (
            {
                "evidence_bindings": (
                    ReactorRegimeEvidenceBinding("validity", "z.ref"),
                    ReactorRegimeEvidenceBinding("quality", "a.ref"),
                )
            },
            "bindings must be unique and sorted",
        ),
        (
            {
                "evidence_bindings": (
                    ReactorRegimeEvidenceBinding("validity", "a.ref"),
                    ReactorRegimeEvidenceBinding("validity", "b.ref"),
                )
            },
            "bindings must be unique and sorted",
        ),
        ({"authority": "control"}, "remain review-only"),
        ({"actionable": True}, "remain review-only"),
        (
            {
                "disposition": ReactorRegimeAxisDisposition.CLASSIFIED,
                "label": "unknown",
            },
            "requires a classified label",
        ),
        (
            {
                "disposition": ReactorRegimeAxisDisposition.CLASSIFIED,
                "label": "not-a-label",
            },
            "not defined by the ontology",
        ),
    ],
)
def test_axis_refuses_generic_identity_and_disposition_drift(
    changes: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _axis("plant_readiness", **changes)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"label": "operational"}, "forbids a physics label"),
        ({"confidence": 0.1}, "confidence must be zero"),
        ({"unknown_reason_id": None}, "requires unknown_reason_id"),
        ({"uncertainty_probability": 0.9}, "must be one"),
        ({"uncertainty_basis_id": None}, "requires uncertainty_basis_id"),
        ({"classifier_id": "mif.test.classifier"}, "forbids classifier policies"),
        ({"dwell_samples": 1}, "forbids dwell_samples"),
        (
            {
                "evidence_ids": ("mif.test.partial-evidence",),
                "evidence_bindings": (
                    ReactorRegimeEvidenceBinding("undefined_role", "mif.test.ref"),
                ),
                "evidence_class": EvidenceClass.SIMULATION,
                "validity": ValidityState.VALID,
                "quality": QualityState.VALID,
                "observability": 0.1,
            },
            "undefined evidence role",
        ),
        ({"quality": QualityState.VALID}, "explicit unknown states"),
    ],
)
def test_unknown_axis_refuses_false_classification(
    changes: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _axis("plant_readiness", **changes)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"label": "regulated"}, "forbids a physics label"),
        ({"confidence": 0.1}, "metrics must be zero"),
        ({"observability": 0.1}, "metrics must be zero"),
        ({"uncertainty_probability": 0.1}, "metrics must be zero"),
        ({"evidence_ids": ("mif.test.evidence",)}, "forbids evidence"),
        ({"applicability_basis": ()}, "requires applicability basis"),
        (
            {
                "evidence_bindings": (
                    ReactorRegimeEvidenceBinding("validity", "mif.test.ref"),
                )
            },
            "forbids evidence bindings",
        ),
        ({"unknown_reason_id": "mif.test.reason"}, "forbids unknown_reason_id"),
        ({"uncertainty_basis_id": "mif.test.basis"}, "forbids uncertainty_basis"),
        ({"threshold_policy_id": "mif.test.policy"}, "forbids classifier"),
        ({"dwell_samples": 1}, "forbids dwell_samples"),
        ({"evidence_class": EvidenceClass.UNKNOWN}, "ontology review identity"),
        ({"validity": ValidityState.UNKNOWN}, "ontology review identity"),
        ({"quality": QualityState.UNKNOWN}, "ontology review identity"),
    ],
)
def test_not_applicable_axis_refuses_measurement_claims(
    changes: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _axis("exhaust_or_boundary", **changes)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"label": None}, "requires a classified label"),
        ({"confidence": 0.0}, "non-zero confidence"),
        ({"observability": 0.0}, "non-zero confidence"),
        ({"evidence_ids": ()}, "requires evidence"),
        ({"unknown_reason_id": "mif.test.reason"}, "forbids unknown_reason"),
        ({"uncertainty_basis_id": None}, "requires uncertainty_basis"),
        ({"validity": ValidityState.STALE}, "usable validity"),
        ({"quality": QualityState.INVALID}, "usable quality"),
        ({"evidence_class": EvidenceClass.CONCEPT}, "qualified evidence class"),
        ({"evidence_bindings": ()}, "every ontology evidence role"),
        ({"classifier_id": None}, "requires classifier policies"),
        ({"classifier_version": "bad"}, "MAJOR.MINOR.PATCH"),
        ({"classifier_sha256": "bad"}, "64 lowercase"),
        ({"threshold_policy_id": None}, "requires classifier policies"),
        ({"dwell_samples": 0}, "positive dwell_samples"),
    ],
)
def test_classified_axis_refuses_incomplete_epistemic_identity(
    changes: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _axis("confinement_or_assembly", classified=True, **changes)


def test_non_classifier_axis_refuses_invented_classifier_policy() -> None:
    with pytest.raises(ValueError, match="not required by ontology"):
        _axis(
            "plant_readiness",
            classified=True,
            classifier_id="mif.test.classifier",
        )


def test_non_threshold_axis_refuses_dwell() -> None:
    with pytest.raises(ValueError, match="without threshold provenance"):
        _axis("plant_readiness", classified=True, dwell_samples=1)


def test_disposition_must_agree_with_static_applicability() -> None:
    with pytest.raises(ValueError, match="classified axis must be statically"):
        _axis(
            "exhaust_or_boundary",
            disposition=ReactorRegimeAxisDisposition.CLASSIFIED,
        )
    with pytest.raises(ValueError, match="unknown axis must be statically"):
        _axis(
            "exhaust_or_boundary",
            disposition=ReactorRegimeAxisDisposition.UNKNOWN,
        )
    with pytest.raises(ValueError, match="requires static support"):
        _axis(
            "plant_readiness",
            disposition=ReactorRegimeAxisDisposition.NOT_APPLICABLE,
        )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"schema": "other.schema"}, "unsupported.*schema"),
        ({"schema_version": "2.0.0"}, "unsupported.*version"),
        ({"schema_version": "v1"}, "MAJOR.MINOR.PATCH"),
        ({"configuration": "missing"}, "unregistered reactor configuration"),
        ({"producer_project": "SCPN-MIF-CORE"}, "producer must be SPO"),
        ({"source_project": "SCPN-CONTROL"}, "plant-truth owner"),
        ({"producer_revision": "bad"}, "40-character Git revision"),
        ({"source_revision": "bad"}, "40-character Git revision"),
        ({"producer_artifact_sha256": "bad"}, "64 lowercase"),
        ({"source_semantic_ids": ()}, "requires source semantic"),
        ({"source_semantic_ids": ("z.semantic", "a.semantic")}, "unique and sorted"),
        ({"clock_kind": ClockKind.UNKNOWN}, "known clock kind"),
        ({"valid_from_ns": 10_001}, "times are inconsistent"),
        ({"evidence_timestamp_ns": 20_001}, "times are inconsistent"),
        ({"assessed_at_ns": 30_001}, "times are inconsistent"),
        ({"sample_rate_hz": 0.0}, "must be positive"),
        ({"latency_s": -1.0}, "must be non-negative"),
        ({"timestamp_offset_ps": 1000}, "must be in"),
        ({"axes": ()}, "exactly eight unique axes"),
        ({"reactor_registry_version": "2.0.0"}, "reactor registry binding"),
        ({"semantic_profile_registry_digest": "1" * 64}, "semantic profile registry"),
        ({"observability_registry_version": "2.0.0"}, "observability registry"),
        ({"ontology_digest": "1" * 64}, "ontology binding"),
        ({"classification_performed": True}, "cannot claim classifier execution"),
        ({"authority": "control"}, "must remain review-only"),
        ({"actionable": True}, "must remain review-only"),
    ],
)
def test_assessment_refuses_identity_clock_registry_and_authority_drift(
    changes: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _assessment(**changes)


def test_assessment_refuses_axis_order_and_static_binding_drift() -> None:
    axes = list(_assessment().axes)
    axes[0], axes[1] = axes[1], axes[0]
    with pytest.raises(ValueError, match="canonical order"):
        _assessment(axes=tuple(axes))

    changed = list(_assessment().axes)
    changed[0] = _axis(
        changed[0].axis_id,
        static_applicability=AxisApplicability.NOT_APPLICABLE,
        disposition=ReactorRegimeAxisDisposition.NOT_APPLICABLE,
        uncertainty_probability=0.0,
        uncertainty_basis_id=None,
        applicability_basis=("spo.test.invalid-static-basis",),
        unknown_reason_id=None,
        evidence_class=EvidenceClass.REVIEW_HYPOTHESIS,
        validity=ValidityState.VALID,
        quality=QualityState.VALID,
    )
    with pytest.raises(ValueError, match="static applicability binding mismatch"):
        _assessment(axes=tuple(changed))


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda r: r.update(schema="other.schema"), "unsupported.*schema"),
        (lambda r: r.update(schema_version="2.0.0"), "unsupported.*version"),
        (lambda r: r["payload"].pop("assessment_id"), "missing fields"),
        (lambda r: r["payload"].update(extra=True), "unknown fields"),
        (lambda r: r["payload"].update(clock_kind="bad"), "unknown.*enum"),
        (lambda r: r["payload"].update(axes={}), "axes must be a list"),
        (
            lambda r: r["payload"].update(source_semantic_ids={}),
            "source_semantic_ids must be a list of strings",
        ),
        (
            lambda r: r["payload"].update(source_semantic_ids=[1]),
            "source_semantic_ids must be a list of strings",
        ),
        (lambda r: r["payload"]["axes"][0].pop("axis_id"), "missing fields"),
        (
            lambda r: r["payload"]["axes"][0].update(disposition="bad"),
            "unknown reactor regime axis enum",
        ),
        (
            lambda r: r["payload"]["axes"][0].update(evidence_bindings={}),
            "evidence_bindings must be a list",
        ),
        (
            lambda r: r["payload"].update(assessment_id="spo.assessment.tampered"),
            "payload digest mismatch",
        ),
    ],
)
def test_record_decoder_refuses_shape_enum_and_digest_drift(
    mutator: Any,
    match: str,
) -> None:
    record = regime_assessment_to_record(_assessment())
    mutator(record)
    if "digest mismatch" not in match:
        _reseal(record)
    with pytest.raises(ValueError, match=match):
        regime_assessment_from_record(record)


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (b"", "input size is invalid"),
        (b"x" * (MAX_REACTOR_REGIME_ASSESSMENT_BYTES + 1), "input size"),
        (b"\xff", "invalid.*JSON"),
        (b"{", "invalid.*JSON"),
        (b'{"schema":"a","schema":"b"}', "duplicate JSON key"),
    ],
)
def test_bytes_decoder_refuses_invalid_transport(data: bytes, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        regime_assessment_from_bytes(data)


def test_bytes_decoder_refuses_non_bytes_and_noncanonical_json() -> None:
    with pytest.raises(ValueError, match="must be bytes"):
        regime_assessment_from_bytes("not bytes")  # type: ignore[arg-type]
    pretty = json.dumps(regime_assessment_to_record(_assessment()), indent=2).encode()
    with pytest.raises(ValueError, match="not canonical"):
        regime_assessment_from_bytes(pretty)
