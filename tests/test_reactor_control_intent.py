# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor research ControlIntent tests

"""Portable, fail-closed, and non-actuating ControlIntent contract tests."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from scpn_phase_orchestrator.reactor_semantics import (
    DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY,
    DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY,
    DEFAULT_REACTOR_REGISTRY,
    MAX_REACTOR_CONTROL_INTENT_BYTES,
    REACTOR_CONTROL_INTENT_SCHEMA,
    REACTOR_CONTROL_INTENT_VERSION,
    REVIEW_ONLY_AUTHORITY,
    ClockKind,
    ControlVariableDirection,
    ControlVariableEnvelope,
    EvidenceClass,
    QualityState,
    ReactorControlObjective,
    ReactorResearchControlIntent,
    ValidityState,
    control_intent_digest,
    control_intent_from_bytes,
    control_intent_from_record,
    control_intent_to_bytes,
    control_intent_to_record,
)

SCHEMA_PATH = Path("docs/specs/reactor_control_intent.schema.json")


def _variable(**changes: object) -> ControlVariableEnvelope:
    fields: dict[str, object] = {
        "variable_id": "mif.driver.arrival_offset",
        "units": "s",
        "lower_bound": -0.001,
        "upper_bound": 0.001,
        "max_abs_delta": 0.0001,
        "max_abs_rate_per_s": 0.001,
        "baseline_value": 0.00002,
        "proposed_value": 0.00001,
        "proposed_delta": -0.00001,
        "proposed_rate_per_s": -0.0001,
        "rate_horizon_s": 0.1,
        "baseline_evidence_id": "mif.frc.evidence.driver-arrival",
        "baseline_timestamp_ns": 15_000,
        "direction": ControlVariableDirection.DECREASE,
    }
    fields.update(changes)
    return ControlVariableEnvelope(**fields)  # type: ignore[arg-type]


def _intent(**changes: object) -> ReactorResearchControlIntent:
    fields: dict[str, object] = {
        "intent_id": "spo.intent.mif.driver-sync.0001",
        "reactor_context_id": "mif.frc.merge-compression.context.0001",
        "configuration": "frc_compression_mif",
        "event_id": "mif.frc.event.0001",
        "producer_project": "SCPN-PHASE-ORCHESTRATOR",
        "producer_revision": "c" * 40,
        "producer_artifact_sha256": "d" * 64,
        "source_handoff_schema": (
            "scpn-phase-orchestrator.mif-merge-compression-handoff.v1"
        ),
        "source_handoff_sha256": "a" * 64,
        "source_revision": "1" * 40,
        "source_admission_schema": "scpn-control.reactor-semantic-admission.v1",
        "source_admission_sha256": "2" * 64,
        "source_admission_decision_digest": "3" * 64,
        "source_regime_id": "mif.frc.regime.0001",
        "source_regime_assignment_sha256": "4" * 64,
        "source_regime_label": "aligning",
        "source_semantic_ids": (
            "mif.frc.semantic.arrival-offset",
            "mif.frc.semantic.trigger-gate",
        ),
        "evidence_ids": (
            "mif.frc.evidence.clock-correlation",
            "mif.frc.evidence.driver-arrival",
        ),
        "evidence_class": EvidenceClass.SIMULATION,
        "source_validity": ValidityState.VALID,
        "source_quality": QualityState.VALID,
        "objective": ReactorControlObjective.DRIVER_SYNCHRONIZATION,
        "hypothesized_target_regime_label": "synchronized",
        "effect_hypothesis": (
            "Reducing the simulated arrival offset may improve driver alignment."
        ),
        "device_control_contract_id": "scpn-mif-core.control.driver-timing.v1",
        "device_control_contract_schema": "scpn-mif-core.control-contract.v1",
        "device_control_contract_sha256": "b" * 64,
        "variable": _variable(),
        "clock_domain": "mif.simulation.monotonic",
        "clock_kind": ClockKind.SIMULATION_MONOTONIC,
        "clock_epoch": "mif.frc.event.0001.start",
        "evidence_timestamp_ns": 10_000,
        "sample_rate_hz": 100_000.0,
        "latency_s": 0.0,
        "timestamp_offset_ps": 0,
        "issued_at_ns": 20_000,
        "valid_until_ns": 30_000,
        "confidence_subject_id": "mif.frc.hypothesis.driver-alignment",
        "confidence": 0.6,
        "observability": 0.8,
        "uncertainty_abs": 0.000002,
        "uncertainty_units": "s",
        "uncertainty_basis": "simulation ensemble absolute timing spread",
        "reactor_registry_version": DEFAULT_REACTOR_REGISTRY.version,
        "reactor_registry_digest": DEFAULT_REACTOR_REGISTRY.digest,
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
    return ReactorResearchControlIntent(**fields)  # type: ignore[arg-type]


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


def test_public_round_trip_digest_and_schema() -> None:
    intent = _intent()
    record = control_intent_to_record(intent)
    encoded = control_intent_to_bytes(intent)

    assert control_intent_from_record(record) == intent
    assert control_intent_from_bytes(encoded) == intent
    assert control_intent_digest(intent) == hashlib.sha256(encoded).hexdigest()
    assert record["schema"] == REACTOR_CONTROL_INTENT_SCHEMA
    assert record["schema_version"] == REACTOR_CONTROL_INTENT_VERSION
    assert intent.authority == REVIEW_ONLY_AUTHORITY
    assert intent.actionable is False
    assert intent.execution_permitted is False
    assert intent.downstream_control_review_required is True
    assert intent.device_adapter_required is True
    assert intent.operator_approval_required is True
    assert intent.machine_protection_veto_required is True

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(record)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"lower_bound": 1.0, "upper_bound": 1.0}, "below upper_bound"),
        ({"max_abs_delta": 0.0}, "limits must be positive"),
        ({"max_abs_rate_per_s": 0.0}, "limits must be positive"),
        ({"baseline_value": 2.0}, "baseline_value lies outside"),
        ({"proposed_value": 2.0}, "proposed_value lies outside"),
        ({"proposed_delta": -0.0002}, "proposed_delta exceeds"),
        ({"proposed_rate_per_s": -0.002}, "proposed_rate_per_s exceeds"),
        ({"proposed_value": 0.0}, "must equal baseline_value"),
        ({"rate_horizon_s": 0.0}, "rate_horizon_s must be positive"),
        ({"rate_horizon_s": 0.2}, "proposed_delta must equal"),
        (
            {
                "direction": ControlVariableDirection.HOLD,
                "proposed_delta": -0.00001,
            },
            "hold direction requires zero proposed_delta",
        ),
        (
            {
                "direction": ControlVariableDirection.HOLD,
                "proposed_value": 0.00002,
                "proposed_delta": 0.0,
                "proposed_rate_per_s": -0.0001,
            },
            "hold direction requires zero proposed_rate_per_s",
        ),
        (
            {
                "direction": ControlVariableDirection.INCREASE,
                "proposed_delta": -0.00001,
            },
            "increase direction requires positive proposed_delta",
        ),
        (
            {
                "direction": ControlVariableDirection.INCREASE,
                "baseline_value": 0.0,
                "proposed_value": 0.00001,
                "proposed_delta": 0.00001,
                "proposed_rate_per_s": -0.0001,
            },
            "increase direction requires positive proposed_rate_per_s",
        ),
        (
            {
                "direction": ControlVariableDirection.DECREASE,
                "proposed_delta": 0.00001,
                "proposed_value": 0.00003,
            },
            "decrease direction requires negative proposed_delta",
        ),
        (
            {
                "direction": ControlVariableDirection.DECREASE,
                "proposed_rate_per_s": 0.0001,
            },
            "decrease direction requires negative proposed_rate_per_s",
        ),
        ({"direction": "decrease"}, "must be a ControlVariableDirection"),
    ],
)
def test_variable_refuses_inconsistent_candidates(
    changes: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _variable(**changes)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"schema": "other.schema"}, "unsupported reactor ControlIntent schema"),
        ({"schema_version": "2.0.0"}, "unsupported reactor ControlIntent version"),
        ({"schema_version": "v1"}, "must use MAJOR.MINOR.PATCH"),
        ({"configuration": "missing_reactor"}, "unregistered reactor configuration"),
        ({"source_semantic_ids": ()}, "requires semantic and evidence"),
        ({"evidence_ids": ()}, "requires semantic and evidence"),
        (
            {
                "evidence_ids": (
                    "mif.frc.evidence.clock-correlation",
                    "mif.frc.evidence.other",
                )
            },
            "baseline_evidence_id must be present",
        ),
        (
            {"source_semantic_ids": ("z.semantic", "a.semantic")},
            "unique and sorted",
        ),
        ({"evidence_ids": ("a.evidence", "a.evidence")}, "unique and sorted"),
        ({"source_regime_label": "unknown"}, "classified source regime"),
        ({"source_regime_label": "not_applicable"}, "classified source regime"),
        (
            {"source_regime_label": "regulated"},
            "not defined for the objective axis",
        ),
        (
            {"hypothesized_target_regime_label": "desynchronized"},
            "research-safe vocabulary",
        ),
        ({"valid_until_ns": 19_999}, "times are inconsistent"),
        ({"producer_project": "SCPN-CONTROL"}, "producer must be SPO"),
        ({"producer_revision": "bad"}, "40-character Git revision"),
        ({"source_revision": "bad"}, "40-character Git revision"),
        (
            {"evidence_class": EvidenceClass.REVIEW_HYPOTHESIS},
            "requires observed",
        ),
        ({"source_validity": ValidityState.STALE}, "requires valid source evidence"),
        ({"source_quality": QualityState.UNKNOWN}, "requires valid source quality"),
        ({"clock_kind": ClockKind.UNKNOWN}, "requires a known clock kind"),
        ({"sample_rate_hz": 0.0}, "sample_rate_hz must be positive"),
        ({"timestamp_offset_ps": 1000}, "must be in"),
        ({"evidence_timestamp_ns": 16_000}, "baseline timestamp must lie"),
        ({"issued_at_ns": 9_000}, "baseline timestamp must lie"),
        ({"confidence": 1.1}, "must be in"),
        ({"confidence": 0.0}, "must be non-zero"),
        ({"observability": 0.0}, "must be non-zero"),
        ({"uncertainty_abs": -1.0}, "must be non-negative"),
        ({"uncertainty_units": "m"}, "must match"),
        ({"reactor_registry_version": "2.0.0"}, "reactor registry binding"),
        ({"reactor_registry_digest": "c" * 64}, "reactor registry binding"),
        (
            {"observability_registry_version": "2.0.0"},
            "observability registry binding",
        ),
        (
            {"observability_registry_digest": "c" * 64},
            "observability registry binding",
        ),
        ({"ontology_version": "2.0.0"}, "ontology binding"),
        ({"ontology_digest": "c" * 64}, "ontology binding"),
        (
            {"downstream_control_review_required": False},
            "safety gates are mandatory",
        ),
        ({"device_adapter_required": False}, "safety gates are mandatory"),
        ({"operator_approval_required": False}, "safety gates are mandatory"),
        (
            {"machine_protection_veto_required": False},
            "safety gates are mandatory",
        ),
        ({"execution_permitted": True}, "can never permit execution"),
        ({"authority": "control"}, "must remain review-only"),
        ({"actionable": True}, "must remain review-only"),
    ],
)
def test_intent_refuses_identity_semantic_and_authority_drift(
    changes: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _intent(**changes)


def test_nonapplicable_objective_is_refused() -> None:
    with pytest.raises(ValueError, match="not applicable to configuration"):
        _intent(
            configuration="laser_icf_direct_drive",
            objective=ReactorControlObjective.EXHAUST_OR_BOUNDARY,
            source_regime_label="conditioned",
            hypothesized_target_regime_label="regulated",
        )


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda r: r.update(schema="other.schema"), "unsupported.*schema"),
        (lambda r: r.update(schema_version="2.0.0"), "unsupported.*version"),
        (lambda r: r["payload"].pop("intent_id"), "missing fields"),
        (lambda r: r["payload"].update(extra=True), "unknown fields"),
        (
            lambda r: r["payload"].update(objective="plant_readiness"),
            "unknown reactor ControlIntent enum value",
        ),
        (lambda r: r["payload"]["variable"].pop("units"), "missing fields"),
        (
            lambda r: r["payload"]["variable"].update(direction="sideways"),
            "unknown control variable direction",
        ),
        (lambda r: r["payload"].update(evidence_ids="bad"), "array of strings"),
    ],
)
def test_record_decoder_refuses_closed_contract_drift(
    mutator: Any,
    match: str,
) -> None:
    record = control_intent_to_record(_intent())
    mutator(record)
    if (
        record.get("schema") == REACTOR_CONTROL_INTENT_SCHEMA
        and record.get("schema_version") == REACTOR_CONTROL_INTENT_VERSION
    ):
        _reseal(record)
    with pytest.raises(ValueError, match=match):
        control_intent_from_record(record)


def test_record_decoder_refuses_payload_digest_tampering() -> None:
    record = control_intent_to_record(_intent())
    record["payload"]["event_id"] = "mif.frc.event.tampered"  # type: ignore[index]
    with pytest.raises(ValueError, match="payload digest mismatch"):
        control_intent_from_record(record)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"", "non-empty bytes"),
        ("not-bytes", "non-empty bytes"),
        (b"\xff", "strict UTF-8"),
        (b"{", "JSON is invalid"),
        (b'{"schema":"x","schema":"y"}', "duplicate JSON key"),
    ],
)
def test_byte_decoder_refuses_invalid_transport(payload: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        control_intent_from_bytes(payload)  # type: ignore[arg-type]


def test_byte_decoder_refuses_size_and_noncanonical_json() -> None:
    with pytest.raises(ValueError, match="maximum size"):
        control_intent_from_bytes(b"x" * (MAX_REACTOR_CONTROL_INTENT_BYTES + 1))

    noncanonical = json.dumps(control_intent_to_record(_intent()), indent=2).encode()
    with pytest.raises(ValueError, match="canonical JSON"):
        control_intent_from_bytes(noncanonical)


def test_variable_record_refuses_shape_and_unknown_direction() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        ControlVariableEnvelope.from_record([])
    record = _variable().to_record()
    record["direction"] = "sideways"
    with pytest.raises(ValueError, match="unknown control variable direction"):
        ControlVariableEnvelope.from_record(record)


def test_facade_import_does_not_load_action_paths() -> None:
    code = """
import json
import sys
import scpn_phase_orchestrator.reactor_semantics as rs
blocked = [
    name for name in sys.modules
    if any(part in name.lower() for part in ("supervisor", "actuation", "scpn_control"))
]
print(json.dumps({"schema": rs.REACTOR_CONTROL_INTENT_SCHEMA, "blocked": blocked}))
"""
    output = subprocess.check_output([sys.executable, "-c", code], text=True)
    result = json.loads(output)
    assert result["schema"] == REACTOR_CONTROL_INTENT_SCHEMA
    assert result["blocked"] == []


def test_public_facade_exports_contract_without_root_widening() -> None:
    facade = importlib.import_module("scpn_phase_orchestrator.reactor_semantics")
    root = importlib.import_module("scpn_phase_orchestrator")
    assert facade.ReactorResearchControlIntent is ReactorResearchControlIntent
    assert not hasattr(root, "ReactorResearchControlIntent")
