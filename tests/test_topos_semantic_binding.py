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

import pytest

from scpn_phase_orchestrator.binding.semantic import (
    RetrievalEvidence,
    compile_symbolic_binding,
)
from scpn_phase_orchestrator.binding.topos_semantic import (
    _build_report_hash,
    validate_symbolic_binding_functor,
)


def _hash_record(record: dict) -> str:
    normalized = dict(record)
    normalized.pop("report_hash", None)
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode(
            "utf-8",
        )
    ).hexdigest()


def _all_obligations_passed(payload: dict) -> bool:
    return all(item["status"] == "passed" for item in payload["obligation_records"])


def _find_obligation(payload: dict, name: str) -> dict:
    for obligation in payload["obligation_records"]:
        if obligation["name"] == name:
            return obligation
    raise AssertionError(f"missing obligation {name!r}")


def test_validate_symbolic_binding_functor_passes_for_compiled_artifacts():
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


def test_validation_report_is_deterministic_for_same_artifacts():
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


def test_validate_symbolic_binding_functor_fails_for_malformed_input_type():
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


def test_validate_symbolic_binding_functor_fails_schema_obligation_on_modified():
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


def test_retrieval_evidence_creates_evidence_morphisms_when_present():
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


def test_report_is_json_safe_for_valid_artifacts():
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


def test_report_hash_rejects_non_finite_audit_payload_numbers():
    with pytest.raises(ValueError, match="finite JSON"):
        _build_report_hash({"report_hash": "", "object_count": float("nan")})
