# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Topos-theoretic symbolic-binding validation tests

"""Tests for deterministic symbolic binding categorical validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

import pytest

from scpn_phase_orchestrator.binding.semantic import (
    GeneratedBindingArtifacts,
    RetrievalEvidence,
    compile_symbolic_binding,
)
from scpn_phase_orchestrator.binding.topos_semantic import (
    _build_report_hash,
    validate_symbolic_binding_functor,
)


def _hash_record(record: dict[str, Any]) -> str:
    normalized = dict(record)
    normalized.pop("report_hash", None)
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode(
            "utf-8",
        )
    ).hexdigest()


def _all_obligations_passed(payload: dict[str, Any]) -> bool:
    return all(item["status"] == "passed" for item in payload["obligation_records"])


def _find_obligation(payload: dict[str, Any], name: str) -> dict[str, Any]:
    for obligation in payload["obligation_records"]:
        if obligation["name"] == name:
            return obligation
    raise AssertionError(f"missing obligation {name!r}")


def test_validate_symbolic_binding_functor_passes_for_compiled_artifacts() -> None:
    artifacts = compile_symbolic_binding(
        "A 2 layer power and grid symbolic control prompt",
        name="test_topos_symbolic_valid",
        oscillators_per_layer=3,
        dry_run_steps=2,
    )
    report = validate_symbolic_binding_functor(artifacts)
    payload = report.to_audit_record()

    assert report.passed is True
    assert report.schema_name == "symbolic_binding_functor"
    assert report.schema_version == "0.1.0"
    assert (
        report.proof_boundary
        == "categorical_validation_prototype_not_formal_topos_proof"
    )
    assert report.non_actuating is True
    assert _all_obligations_passed(payload)
    assert _hash_record(payload) == report.report_hash
    assert isinstance(json.dumps(payload, sort_keys=True, separators=(",", ":")), str)


def test_validation_report_is_deterministic_for_same_artifacts() -> None:
    artifacts = compile_symbolic_binding(
        "A 2 layer power and grid symbolic control prompt",
        name="test_topos_symbolic_deterministic",
        oscillators_per_layer=2,
        dry_run_steps=1,
    )
    first = validate_symbolic_binding_functor(artifacts)
    second = validate_symbolic_binding_functor(artifacts)

    assert first.report_hash == second.report_hash
    assert first.to_audit_record() == second.to_audit_record()


def test_validate_symbolic_binding_functor_fails_for_malformed_input_type() -> None:
    with pytest.raises(
        ValueError,
        match="artifacts must be a GeneratedBindingArtifacts",
    ):
        validate_symbolic_binding_functor(None)

    with pytest.raises(
        ValueError,
        match="artifacts must be a GeneratedBindingArtifacts",
    ):
        validate_symbolic_binding_functor("not-artifacts")


def test_modified_binding_spec_fails_schema_obligation() -> None:
    artifacts = compile_symbolic_binding(
        "A 2 layer biological prompt",
        name="test_topos_symbolic_invalid_schema",
        oscillators_per_layer=2,
        dry_run_steps=1,
    )
    mutated = replace(
        artifacts,
        binding_spec=replace(
            artifacts.binding_spec,
            version="bad.version",
        ),
    )
    report = validate_symbolic_binding_functor(mutated)
    payload = report.to_audit_record()

    assert report.passed is False
    schema_obligation = _find_obligation(payload, "schema_validation_has_no_errors")
    assert schema_obligation["status"] == "failed"


@pytest.mark.parametrize("layer_index", [True, -1])
def test_invalid_layer_indexes_do_not_create_categorical_objects(
    layer_index: object,
) -> None:
    artifacts = compile_symbolic_binding(
        "A 2 layer biological prompt",
        name="test_topos_symbolic_invalid_layer_index",
        oscillators_per_layer=2,
        dry_run_steps=1,
    )
    first_layer = replace(
        artifacts.binding_spec.layers[0],
        index=layer_index,
        name=f"layer_{layer_index}",
    )
    mutated = replace(
        artifacts,
        binding_spec=replace(
            artifacts.binding_spec,
            layers=[first_layer, *artifacts.binding_spec.layers[1:]],
        ),
    )

    report = validate_symbolic_binding_functor(mutated)
    payload = report.to_audit_record()
    index_obligation = _find_obligation(
        payload,
        "layer_indexes_are_non_negative_integers",
    )

    assert report.passed is False
    assert index_obligation["status"] == "failed"
    assert "non-negative integer" in index_obligation["evidence"]
    assert all(obj["name"] != f"layer_{layer_index}" for obj in payload["objects"])


def test_retrieval_evidence_creates_evidence_morphisms_when_present() -> None:
    artifacts = compile_symbolic_binding(
        "A 1 layer cardiac rhythm prompt",
        name="test_topos_symbolic_evidence",
        oscillators_per_layer=2,
        dry_run_steps=1,
    )

    extra_evidence = RetrievalEvidence(
        domainpack="synthetic-domain",
        path="synthetic-domain.yaml",
        score=0.93,
        matched_terms=["synthetic", "domain"],
        summary="synthetic evidence",
        source="domainpack",
        ranking_features={"matched_term_count": 2.0, "source_priority": 1.0},
    )
    evidence_artifacts = replace(artifacts, retrieval_evidence=[extra_evidence])
    report = validate_symbolic_binding_functor(evidence_artifacts)
    payload = report.to_audit_record()

    obligation = _find_obligation(
        payload,
        "retrieval_evidence_to_evidence_morphisms",
    )
    assert obligation["status"] == "passed"
    assert payload["morphism_count"] >= 1


def test_report_is_json_safe_for_valid_artifacts() -> None:
    artifacts = compile_symbolic_binding(
        "A 1 layer financial market prompt",
        name="test_topos_symbolic_json",
        oscillators_per_layer=1,
        dry_run_steps=1,
    )
    report = validate_symbolic_binding_functor(artifacts)
    payload = report.to_audit_record()
    serialised = json.dumps(payload)

    assert isinstance(serialised, str)
    assert all(
        isinstance(item["name"], str)
        and isinstance(item["status"], str)
        and isinstance(item["evidence"], str)
        for item in payload["obligation_records"]
    )


def test_report_hash_rejects_non_finite_audit_payload_numbers() -> None:
    with pytest.raises(ValueError, match="finite JSON"):
        _build_report_hash({"report_hash": "", "object_count": float("nan")})


def _valid_artifacts() -> GeneratedBindingArtifacts:
    return compile_symbolic_binding(
        "A 2 layer power and grid symbolic control prompt",
        name="topos_coverage_base",
        oscillators_per_layer=3,
        dry_run_steps=2,
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("validation_errors", "x", "validation_errors must be a list"),
        ("retrieval_evidence", "x", "retrieval_evidence must be a list"),
        ("audit_record", "x", "audit_record must be a dict"),
        ("binding_yaml", 1, "binding_yaml must be a string"),
        ("policy_yaml", 1, "policy_yaml must be a string"),
        ("notebook_json", 1, "notebook_json must be a string"),
        ("retrieval_evidence", ["not-evidence"], "must be RetrievalEvidence"),
    ],
)
def test_validate_rejects_malformed_artifact_fields(field, value, match) -> None:
    artifacts = replace(_valid_artifacts(), **{field: value})
    with pytest.raises(ValueError, match=match):
        validate_symbolic_binding_functor(artifacts)


def test_empty_binding_layers_and_families_fail_presence_obligations() -> None:
    artifacts = _valid_artifacts()
    empty_spec = replace(
        artifacts.binding_spec,
        layers=[],
        oscillator_families={},
    )
    report = validate_symbolic_binding_functor(
        replace(artifacts, binding_spec=empty_spec, retrieval_evidence=[]),
    )
    payload = report.to_audit_record()

    assert report.passed is False
    assert _find_obligation(payload, "binding_layers_non_empty")["status"] == "failed"
    assert (
        _find_obligation(payload, "binding_oscillator_families_non_empty")["status"]
        == "failed"
    )
    assert (
        _find_obligation(payload, "binding_layer_and_family_presence")["status"]
        == "failed"
    )
    assert (
        _find_obligation(payload, "retrieval_evidence_to_evidence_morphisms")[
            "status"
        ]
        == "passed"
    )


def test_duplicate_and_noncanonical_layers_fail_mapping_obligations() -> None:
    artifacts = _valid_artifacts()
    first_layer = replace(artifacts.binding_spec.layers[0], name="renamed_layer")
    duplicate_layer = replace(
        artifacts.binding_spec.layers[1],
        index=artifacts.binding_spec.layers[0].index,
    )
    mutated_spec = replace(
        artifacts.binding_spec,
        layers=[first_layer, duplicate_layer],
    )
    report = validate_symbolic_binding_functor(
        replace(artifacts, binding_spec=mutated_spec),
    )
    payload = report.to_audit_record()

    assert report.passed is False
    assert _find_obligation(payload, "layer_indexes_unique")["status"] == "failed"
    assert (
        _find_obligation(payload, "layer_indexes_map_to_stable_object_names")[
            "status"
        ]
        == "failed"
    )


def test_retrieval_evidence_without_layer_targets_fails_morphism_obligation() -> None:
    artifacts = _valid_artifacts()
    invalid_layers = [
        replace(layer, index=-1, name=f"invalid_{position}")
        for position, layer in enumerate(artifacts.binding_spec.layers)
    ]
    extra_evidence = RetrievalEvidence(
        domainpack="synthetic-domain",
        path="synthetic-domain.yaml",
        score=0.93,
        matched_terms=["synthetic", "domain"],
        summary="synthetic evidence",
        source="domainpack",
        ranking_features={"matched_term_count": 2.0, "source_priority": 1.0},
    )
    report = validate_symbolic_binding_functor(
        replace(
            artifacts,
            binding_spec=replace(artifacts.binding_spec, layers=invalid_layers),
            retrieval_evidence=[extra_evidence],
        ),
    )
    payload = report.to_audit_record()
    obligation = _find_obligation(
        payload,
        "retrieval_evidence_to_evidence_morphisms",
    )

    assert report.passed is False
    assert obligation["status"] == "failed"
    assert (
        obligation["evidence"]
        == "cannot map retrieval evidence without layer objects"
    )


def test_audit_record_boundary_fields_fail_when_mutated() -> None:
    artifacts = _valid_artifacts()
    report = validate_symbolic_binding_functor(
        replace(
            artifacts,
            audit_record={
                **artifacts.audit_record,
                "schema_valid": "yes",
                "proof_boundary": "unbounded_claim",
                "non_actuating": False,
            },
        ),
    )
    payload = report.to_audit_record()

    assert report.passed is False
    assert (
        _find_obligation(payload, "audit_record_preserves_schema_status")["status"]
        == "failed"
    )
    assert (
        _find_obligation(payload, "audit_record_boundary_stability")["status"]
        == "failed"
    )
    assert (
        _find_obligation(payload, "audit_record_non_actuation_boundary")["status"]
        == "failed"
    )
