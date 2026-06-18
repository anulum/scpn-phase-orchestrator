# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Topos-theoretic symbolic-binding validation

"""Deterministic audit/proof-obligation validation for symbolic bindings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from scpn_phase_orchestrator.binding.semantic import (
    GeneratedBindingArtifacts,
    RetrievalEvidence,
)
from scpn_phase_orchestrator.binding.validator import validate_binding_spec

_SCHEMA_NAME = "symbolic_binding_functor"
_SCHEMA_VERSION = "0.1.0"
_PROOF_BOUNDARY = "categorical_validation_prototype_not_formal_topos_proof"
_LAYER_NAME_PREFIX = "layer_"

__all__ = [
    "SymbolicBindingObligation",
    "SymbolicBindingObject",
    "SymbolicBindingMorphism",
    "SymbolicBindingValidationReport",
    "validate_symbolic_binding_functor",
]


@dataclass(frozen=True)
class SymbolicBindingObligation:
    """Single proof obligation outcome for the symbolic-binding functor."""

    name: str
    status: str
    evidence: str

    def to_audit_record(self) -> dict[str, str]:
        """Return a deterministic JSON-safe audit record.

        Returns
        -------
        dict[str, str]
            Deterministic, JSON-safe audit mapping of the SymbolicBindingObligation
            fields.
        """
        return {
            "name": self.name,
            "status": self.status,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class SymbolicBindingObject:
    """Category object used in the symbolic-binding proof sketch."""

    name: str
    kind: str
    detail: str

    def to_audit_record(self) -> dict[str, str]:
        """Return a deterministic JSON-safe audit record.

        Returns
        -------
        dict[str, str]
            Deterministic, JSON-safe audit mapping of the SymbolicBindingObject fields.
        """
        return {
            "name": self.name,
            "kind": self.kind,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SymbolicBindingMorphism:
    """Deterministic relation between symbolic-binding validation objects."""

    source: str
    target: str
    label: str
    deterministic: bool = True

    def to_audit_record(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe audit record.

        Returns
        -------
        dict[str, Any]
            Deterministic, JSON-safe audit mapping of the SymbolicBindingMorphism
            fields.
        """
        return {
            "source": self.source,
            "target": self.target,
            "label": self.label,
            "deterministic": self.deterministic,
        }


@dataclass(frozen=True)
class SymbolicBindingValidationReport:
    """JSON-safe deterministic report for symbolic-binding validation."""

    schema_name: str
    schema_version: str
    object_count: int
    morphism_count: int
    obligation_records: tuple[SymbolicBindingObligation, ...]
    objects: tuple[SymbolicBindingObject, ...]
    morphisms: tuple[SymbolicBindingMorphism, ...]
    passed: bool
    report_hash: str
    proof_boundary: str
    non_actuating: bool = True

    def to_audit_record(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe audit record.

        Returns
        -------
        dict[str, Any]
            Deterministic, JSON-safe audit mapping of the
            SymbolicBindingValidationReport fields.
        """
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "object_count": self.object_count,
            "morphism_count": self.morphism_count,
            "obligation_records": [
                obligation.to_audit_record() for obligation in self.obligation_records
            ],
            "objects": [obj.to_audit_record() for obj in self.objects],
            "morphisms": [morphism.to_audit_record() for morphism in self.morphisms],
            "passed": self.passed,
            "report_hash": self.report_hash,
            "proof_boundary": self.proof_boundary,
            "non_actuating": self.non_actuating,
        }


def _build_report_hash(record: dict[str, Any]) -> str:
    """Build a deterministic report hash from JSON-sorted payload."""
    payload = dict(record)
    payload.pop("report_hash", None)
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except ValueError as exc:
        raise ValueError(
            "report payload must contain only finite JSON numbers"
        ) from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _add_obligation(
    obligations: list[SymbolicBindingObligation],
    *,
    name: str,
    passed: bool,
    evidence: str,
) -> None:
    obligations.append(
        SymbolicBindingObligation(
            name=name,
            status="passed" if passed else "failed",
            evidence=evidence,
        )
    )


def _validate_binding_artefact_inputs(
    artifacts: Any,
) -> GeneratedBindingArtifacts:
    if not isinstance(artifacts, GeneratedBindingArtifacts):
        raise ValueError("artifacts must be a GeneratedBindingArtifacts object")

    if not isinstance(artifacts.validation_errors, list):
        raise ValueError("artifacts.validation_errors must be a list")

    if not isinstance(artifacts.retrieval_evidence, list):
        raise ValueError("artifacts.retrieval_evidence must be a list")

    if not isinstance(artifacts.audit_record, dict):
        raise ValueError("artifacts.audit_record must be a dict")

    if not isinstance(artifacts.binding_yaml, str):
        raise ValueError("artifacts.binding_yaml must be a string")

    if not isinstance(artifacts.policy_yaml, str):
        raise ValueError("artifacts.policy_yaml must be a string")

    if not isinstance(artifacts.notebook_json, str):
        raise ValueError("artifacts.notebook_json must be a string")

    for item in artifacts.retrieval_evidence:
        if not isinstance(item, RetrievalEvidence):
            raise ValueError("each retrieval evidence entry must be RetrievalEvidence")

    return artifacts


def _collect_layer_objects_and_morphisms(
    artifacts: GeneratedBindingArtifacts,
    obligations: list[SymbolicBindingObligation],
) -> tuple[
    tuple[SymbolicBindingObject, ...],
    tuple[SymbolicBindingMorphism, ...],
    bool,
]:
    spec = artifacts.binding_spec
    objects: list[SymbolicBindingObject] = []
    morphisms: list[SymbolicBindingMorphism] = []
    layer_indices: set[int] = set()
    map_ok = True
    invalid_index_evidence: list[str] = []

    if not spec.layers:
        _add_obligation(
            obligations,
            name="binding_layers_non_empty",
            passed=False,
            evidence="generated binding layers must be non-empty",
        )

    if not spec.oscillator_families:
        _add_obligation(
            obligations,
            name="binding_oscillator_families_non_empty",
            passed=False,
            evidence="generated binding must define at least one oscillator family",
        )

    for layer in spec.layers:
        if (
            isinstance(layer.index, bool)
            or not isinstance(layer.index, Integral)
            or layer.index < 0
        ):
            map_ok = False
            invalid_index_evidence.append(
                f"layer index must be a non-negative integer, got {layer.index!r}"
            )
            continue

        layer_index = int(layer.index)
        if layer_index in layer_indices:
            map_ok = False
            _add_obligation(
                obligations,
                name="layer_indexes_unique",
                passed=False,
                evidence=f"layer index {layer_index} is duplicated",
            )
            continue

        layer_indices.add(layer_index)
        canonical = f"{_LAYER_NAME_PREFIX}{layer_index}"
        if layer.name != canonical:
            map_ok = False
            _add_obligation(
                obligations,
                name="layer_indexes_map_to_stable_object_names",
                passed=False,
                evidence=(
                    f"layer index {layer_index} should map to canonical name "
                    f"{canonical!r}; got {layer.name!r}"
                ),
            )

        objects.append(
            SymbolicBindingObject(
                name=canonical,
                kind="layer",
                detail=(
                    f"index={layer_index}, name={layer.name}, "
                    f"oscs={len(layer.oscillator_ids)}"
                ),
            )
        )
        morphisms.append(
            SymbolicBindingMorphism(
                source=f"index:{layer_index}",
                target=canonical,
                label="index_to_stable_layer_name",
            )
        )

    if spec.layers:
        _add_obligation(
            obligations,
            name="layer_indexes_are_non_negative_integers",
            passed=not invalid_index_evidence,
            evidence=(
                "all layer indexes are non-negative integers"
                if not invalid_index_evidence
                else "; ".join(invalid_index_evidence)
            ),
        )
        _add_obligation(
            obligations,
            name="layer_indexes_map_to_stable_object_names",
            passed=map_ok,
            evidence=(
                "all layer indexes map to stable object names"
                if map_ok
                else "layer index to object-name mapping is inconsistent"
            ),
        )

    return (
        tuple(sorted(objects, key=lambda item: item.name)),
        tuple(
            sorted(morphisms, key=lambda item: (item.source, item.target, item.label))
        ),
        map_ok,
    )


def _collect_evidence_objects_and_morphisms(
    artifacts: GeneratedBindingArtifacts,
    obligations: list[SymbolicBindingObligation],
) -> tuple[tuple[SymbolicBindingObject, ...], tuple[SymbolicBindingMorphism, ...]]:
    objects: list[SymbolicBindingObject] = []
    morphisms: list[SymbolicBindingMorphism] = []

    if not artifacts.retrieval_evidence:
        _add_obligation(
            obligations,
            name="retrieval_evidence_to_evidence_morphisms",
            passed=True,
            evidence=(
                "no retrieval evidence supplied; "
                "morphism check is intentionally vacuous"
            ),
        )
        return (), ()

    layer_targets = sorted(
        [
            f"{_LAYER_NAME_PREFIX}{int(layer.index)}"
            for layer in artifacts.binding_spec.layers
            if not isinstance(layer.index, bool)
            and isinstance(layer.index, Integral)
            and layer.index >= 0
        ]
    )

    for index, evidence in enumerate(artifacts.retrieval_evidence):
        obj_name = f"evidence[{index}]"
        objects.append(
            SymbolicBindingObject(
                name=obj_name,
                kind="retrieval_evidence",
                detail=f"{evidence.domainpack}:{evidence.path}",
            )
        )

        if layer_targets:
            target = layer_targets[index % len(layer_targets)]
            morphisms.append(
                SymbolicBindingMorphism(
                    source=obj_name,
                    target=target,
                    label=f"{evidence.domainpack}|{evidence.source}",
                )
            )

    _add_obligation(
        obligations,
        name="retrieval_evidence_to_evidence_morphisms",
        passed=bool(morphisms),
        evidence=(
            f"mapped {len(artifacts.retrieval_evidence)} retrieval evidence "
            f"to {len(morphisms)} morphism(s)"
            if morphisms
            else "cannot map retrieval evidence without layer objects"
        ),
    )

    return (
        tuple(objects),
        tuple(
            sorted(morphisms, key=lambda item: (item.source, item.target, item.label))
        ),
    )


def _check_audit_boundary_preserved(
    artifacts: GeneratedBindingArtifacts,
    obligations: list[SymbolicBindingObligation],
    *,
    schema_valid: bool,
) -> None:
    audit = artifacts.audit_record

    audit_schema_valid = audit.get("schema_valid")
    if not isinstance(audit_schema_valid, bool):
        _add_obligation(
            obligations,
            name="audit_record_preserves_schema_status",
            passed=False,
            evidence="audit_record.schema_valid must be a boolean",
        )
    else:
        _add_obligation(
            obligations,
            name="audit_record_preserves_schema_status",
            passed=audit_schema_valid == schema_valid,
            evidence=(
                "audit_record.schema_valid matches revalidated schema result"
                if audit_schema_valid == schema_valid
                else (
                    "audit_record.schema_valid does not match "
                    f"revalidated schema result (audit={audit_schema_valid!r}, "
                    f"computed={schema_valid!r})"
                )
            ),
        )

    boundary = audit.get("proof_boundary", audit.get("claim_boundary"))
    if boundary is not None:
        _add_obligation(
            obligations,
            name="audit_record_boundary_stability",
            passed=boundary == _PROOF_BOUNDARY,
            evidence=(
                f"proof boundary preserved: {boundary!r}"
                if boundary == _PROOF_BOUNDARY
                else f"proof boundary should be {_PROOF_BOUNDARY!r}, got {boundary!r}"
            ),
        )
    else:
        _add_obligation(
            obligations,
            name="audit_record_boundary_stability",
            passed=True,
            evidence=(
                "audit record does not yet encode explicit proof_boundary; "
                "validation boundary is enforced by this report"
            ),
        )

    if "non_actuating" in audit:
        non_actuating = audit.get("non_actuating")
        _add_obligation(
            obligations,
            name="audit_record_non_actuation_boundary",
            passed=non_actuating is True,
            evidence=(
                "audit_record.non_actuating is true"
                if non_actuating is True
                else "audit_record.non_actuating must be true"
            ),
        )
    else:
        _add_obligation(
            obligations,
            name="audit_record_non_actuation_boundary",
            passed=True,
            evidence=(
                "audit record does not carry explicit non_actuating, "
                "and this validator report is explicitly non_actuating"
            ),
        )


def validate_symbolic_binding_functor(
    artifacts: GeneratedBindingArtifacts,
) -> SymbolicBindingValidationReport:
    """Validate symbolic compiler output as a source-to-binding functor.

    Parameters
    ----------
    artifacts : GeneratedBindingArtifacts
        The generated binding artefacts to check for functorial consistency.

    Returns
    -------
    SymbolicBindingValidationReport
        The validation report: objects, morphisms, and any obligation failures.
    """
    artifacts = _validate_binding_artefact_inputs(artifacts)
    obligations: list[SymbolicBindingObligation] = []

    _add_obligation(
        obligations,
        name="artifacts_input_type",
        passed=True,
        evidence="artifacts instance is GeneratedBindingArtifacts",
    )

    schema_errors = validate_binding_spec(artifacts.binding_spec)
    schema_valid = len(schema_errors) == 0
    _add_obligation(
        obligations,
        name="schema_validation_has_no_errors",
        passed=schema_valid,
        evidence=(
            "binding schema has no validation errors"
            if schema_valid
            else f"schema errors: {', '.join(schema_errors)}"
        ),
    )

    layer_objects, layer_morphisms, _ = _collect_layer_objects_and_morphisms(
        artifacts,
        obligations,
    )
    evidence_objects, evidence_morphisms = _collect_evidence_objects_and_morphisms(
        artifacts,
        obligations,
    )

    _add_obligation(
        obligations,
        name="binding_layer_and_family_presence",
        passed=bool(
            artifacts.binding_spec.layers and artifacts.binding_spec.oscillator_families
        ),
        evidence=(
            "binding has non-empty layers and oscillator families"
            if artifacts.binding_spec.layers
            and artifacts.binding_spec.oscillator_families
            else "binding must define both layers and oscillator families"
        ),
    )

    _check_audit_boundary_preserved(
        artifacts,
        obligations,
        schema_valid=schema_valid,
    )

    objects = tuple(
        sorted((*layer_objects, *evidence_objects), key=lambda item: item.name)
    )
    morphisms = tuple(
        sorted(
            (*layer_morphisms, *evidence_morphisms),
            key=lambda item: (item.source, item.target, item.label),
        )
    )

    obligations = sorted(obligations, key=lambda item: item.name)
    report = SymbolicBindingValidationReport(
        schema_name=_SCHEMA_NAME,
        schema_version=_SCHEMA_VERSION,
        object_count=len(objects),
        morphism_count=len(morphisms),
        obligation_records=tuple(obligations),
        objects=objects,
        morphisms=morphisms,
        passed=all(item.status == "passed" for item in obligations),
        report_hash="",
        proof_boundary=_PROOF_BOUNDARY,
        non_actuating=True,
    )

    return SymbolicBindingValidationReport(
        schema_name=report.schema_name,
        schema_version=report.schema_version,
        object_count=report.object_count,
        morphism_count=report.morphism_count,
        obligation_records=report.obligation_records,
        objects=report.objects,
        morphisms=report.morphisms,
        passed=report.passed,
        report_hash=_build_report_hash(report.to_audit_record()),
        proof_boundary=report.proof_boundary,
        non_actuating=report.non_actuating,
    )
