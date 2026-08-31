# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupled-transport semantic exchange tests

"""Public FUSION-bytes to SPO-handoff mapping and refusal tests."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from scpn_phase_orchestrator.reactor_semantics import (
    DEFAULT_REACTOR_REGISTRY,
    FUSION_COUPLED_TRANSPORT_SCHEMA,
    FUSION_TORAX_OUTCOME_SCHEMA,
    FUSION_TORAX_REVIEW_SCHEMA,
    QualityState,
    RegimeState,
    SemanticCarrier,
    ValidityState,
    build_abstaining_regime_assessment,
    coupled_transport_handoff_from_fusion_bytes,
    handoff_from_bytes,
    handoff_to_bytes,
)
from scpn_phase_orchestrator.reactor_semantics import (
    coupled_transport as adapter_module,
)

SOURCE_REVISION = "1" * 40
MODEL_REVISION = "2" * 40
EVENT_ID = "fusion.torax.event.u1"
PROFILE_UNITS = {
    "electron_density": "m^-3",
    "electron_temperature": "keV",
    "ion_temperature": "keV",
    "poloidal_flux": "Wb/rad",
}
SOURCE_UNITS = {
    "driven_current": "A",
    "electron_heat": "W",
    "ion_electron_exchange": "W",
    "ion_heat": "W",
    "particles": "s^-1",
}
BUDGET_UNITS = {
    "particle_inventory": "1",
    "poloidal_flux_l2": "Wb/rad",
    "thermal_energy": "J",
}


def _calibration() -> dict[str, object]:
    return {
        "basis": "simulation_declared_units",
        "calibrated_at_ns": 0,
        "calibration_id": "fusion.torax.simulation_declared_units.v1",
        "empirical": False,
        "transfer": "identity",
        "transfer_function_id": "fusion.torax.identity_projection.v1",
    }


def _observable(unit: str, samples: object) -> dict[str, object]:
    return {"calibration": _calibration(), "samples": samples, "unit": unit}


def _metrics(units: dict[str, str]) -> dict[str, object]:
    return {
        name: {
            "absolute_rms_difference": float(index + 1) / 100.0,
            "relative_l2": float(index + 1) / 1_000.0,
            "unit": unit,
        }
        for index, (name, unit) in enumerate(units.items())
    }


def _record() -> dict[str, Any]:
    profiles = {
        name: _observable(
            unit,
            [
                [float(index + 1), float(index + 2)],
                [float(index + 2), float(index + 3)],
                [float(index + 3), float(index + 4)],
            ],
        )
        for index, (name, unit) in enumerate(PROFILE_UNITS.items())
    }
    sources = {
        name: _observable(unit, [float(index + 1), float(index + 2), float(index + 3)])
        for index, (name, unit) in enumerate(SOURCE_UNITS.items())
    }
    budgets = {
        name: _observable(unit, [float(index + 1), float(index + 2), float(index + 3)])
        for index, (name, unit) in enumerate(BUDGET_UNITS.items())
    }
    payload: dict[str, object] = {
        "clock": {
            "domain": "simulation_monotonic",
            "epoch": "scenario_start",
            "kind": "simulation_monotonic",
            "latency_s": 0.0,
            "picosecond_offset": 0,
            "requested_final_ns": 20_000_000,
            "reset_policy": "fresh_process_no_hidden_state",
            "sample_ns": [0, 10_000_000, 20_000_000],
            "sample_rate_hz": 100.0,
            "synchronized_to": None,
            "timestamp_ns": 20_000_000,
        },
        "completion": {
            "complete": True,
            "reached_final_ns": 20_000_000,
            "sim_error": "NO_ERROR",
        },
        "observables": {
            "numerics": {
                "inner_solver_iterations": [0, 1, 1],
                "outer_solver_iterations": [0, 1, 1],
                "sawtooth_crash": [False, False, False],
                "sim_error": 0,
                "sim_status": "completed",
            },
            "profiles": profiles,
            "rho": {
                "frame": "axisymmetric_circular_toroidal_flux_rho_norm",
                "name": "rho_norm",
                "samples": [0.0, 1.0],
                "unit": "1",
            },
            "source_totals": sources,
            "state_budgets": budgets,
        },
        "reactor": {
            "cadence": "single_experiment",
            "configuration": "conventional_tokamak",
            "configuration_version": "1.0.0",
            "confinement_family": "magnetic_closed",
            "context_id": "fusion.torax.circular_iter_scale_comparison",
            "conversion": "experimental_no_power_conversion",
            "coordinate_frame": "axisymmetric_circular_toroidal_flux_rho_norm",
            "drivers": ["external_magnetic_coils", "plasma_current"],
            "event_id": EVENT_ID,
            "evidence_class": "S",
            "facility": "simulation_only_no_facility",
            "operating_point": {
                "effective_charge": 1.5,
                "fuel_class_basis": (
                    "deuterium_only_input_no_fusion_power_or_burn_model"
                ),
                "impurity": "Ne",
                "magnetic_field_t": 5.3,
                "main_ion": "D",
                "major_radius_m": 6.2,
                "minor_radius_m": 2.0,
                "plasma_current_a": 5_000_000.0,
            },
            "reaction": "deuterium_deuterium",
            "registry_digest": DEFAULT_REACTOR_REGISTRY.digest,
            "registry_version": DEFAULT_REACTOR_REGISTRY.version,
            "schema_version": "1.0.0",
            "topology": "axisymmetric torus",
        },
        "uncertainty": {
            "kind": "numerical_refinement",
            "observables": {
                "profiles": _metrics(PROFILE_UNITS),
                "source_totals": _metrics(SOURCE_UNITS),
                "state_budgets": _metrics(BUDGET_UNITS),
            },
            "primary_dt_ns": 10_000_000,
            "refined_dt_ns": 5_000_000,
        },
        "validity": {
            "authority": "review_only_non_actuating",
            "ood": False,
            "quality": "frozen_model_intersection_reference",
            "state": "VALID",
        },
    }
    record: dict[str, Any] = {
        "event_id": EVENT_ID,
        "model_intersection_schema": FUSION_COUPLED_TRANSPORT_SCHEMA,
        "payload": payload,
        "payload_sha256": "",
        "provenance": {
            "artifact_content_sha256": "3" * 64,
            "deck_sha256": "4" * 64,
            "manifest_inventory_sha256": "5" * 64,
            "model_intersection_revision": MODEL_REVISION,
            "primary_projection_sha256": "6" * 64,
            "refined_projection_sha256": "7" * 64,
            "refined_request_sha256": "8" * 64,
            "request_sha256": "9" * 64,
            "runner_sha256": "a" * 64,
            "runtime_source_sha256": "b" * 64,
        },
        "schema": FUSION_TORAX_REVIEW_SCHEMA,
        "source_revision": SOURCE_REVISION,
        "source_schema": FUSION_TORAX_OUTCOME_SCHEMA,
    }
    return record


def _encode(record: dict[str, Any], *, seal_payload: bool = True) -> bytes:
    record = deepcopy(record)
    if seal_payload:
        record["payload_sha256"] = hashlib.sha256(
            json.dumps(
                record["payload"],
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
    return json.dumps(
        record,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _decode(record: dict[str, Any] | None = None):
    encoded = _encode(_record() if record is None else record)
    return coupled_transport_handoff_from_fusion_bytes(
        encoded,
        expected_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _path(record: dict[str, Any], *parts: str) -> Any:
    value: Any = record
    for part in parts:
        value = value[part]
    return value


def test_public_adapter_maps_exactly_twelve_nonphase_review_observables() -> None:
    handoff = _decode()

    assert len(handoff.observables) == len(handoff.semantics) == 12
    assert handoff.phase_relations == ()
    assert handoff.regime.state is RegimeState.UNKNOWN
    assert handoff.regime.confidence == 0.0
    assert handoff.actionable is False
    assert handoff.authority == "review_only"
    assert handoff.context.event_id == EVENT_ID
    assert handoff.context.operating_point["fuel_class_basis"].endswith(
        "no_fusion_power_or_burn_model"
    )
    assert handoff.observables[0].value == (3.0, 4.0)
    assert handoff.observables[4].value == 3.0
    assert handoff.observables[0].uncertainty.standard_deviation == 0.01
    assert handoff.observables[0].uncertainty.confidence_level == 0.0
    assert all(item.quality.state is QualityState.VALID for item in handoff.observables)
    assert all(
        item.carrier_type is SemanticCarrier.BOUNDED_FEATURE
        and item.phase_rad is None
        and item.validity.state is ValidityState.UNOBSERVABLE
        and item.quality.state is QualityState.UNKNOWN
        and item.confidence == item.observability == 0.0
        for item in handoff.semantics
    )
    encoded = handoff_to_bytes(handoff)
    assert handoff_from_bytes(encoded) == handoff
    assert handoff_to_bytes(_decode()) == encoded


def test_abstaining_builder_enforces_fusion_clock_and_validity_boundary() -> None:
    handoff = _decode()
    assessment = build_abstaining_regime_assessment(
        handoff,
        producer_revision="a" * 40,
        producer_artifact_sha256="b" * 64,
    )

    assert assessment.source_handoff_schema == handoff.schema
    assert (
        assessment.source_handoff_sha256
        == hashlib.sha256(handoff_to_bytes(handoff)).hexdigest()
    )
    assert assessment.actionable is False

    final = handoff.observables[-1]
    shifted_timestamp = final.clock.timestamp_ns + 1
    shifted = replace(
        final,
        clock=replace(final.clock, timestamp_ns=shifted_timestamp),
        validity=replace(
            final.validity,
            valid_from_ns=shifted_timestamp,
            valid_until_ns=shifted_timestamp,
        ),
    )
    with pytest.raises(ValueError, match="no common validity"):
        build_abstaining_regime_assessment(
            replace(
                handoff,
                observables=(*handoff.observables[:-1], shifted),
            ),
            producer_revision="a" * 40,
            producer_artifact_sha256="b" * 64,
        )

    mismatched_clock = replace(
        final,
        clock=replace(final.clock, sample_rate_hz=final.clock.sample_rate_hz * 2.0),
    )
    with pytest.raises(ValueError, match="identical assessment clock metadata"):
        build_abstaining_regime_assessment(
            replace(
                handoff,
                observables=(*handoff.observables[:-1], mismatched_clock),
            ),
            producer_revision="a" * 40,
            producer_artifact_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema", "other.v1", "review schema"),
        ("source_schema", "other.v1", "outcome schema"),
        ("model_intersection_schema", "other.v1", "coupled-transport schema"),
        ("source_revision", "deadbeef", "Git revision"),
        ("event_id", "", "event_id"),
        ("payload_sha256", "0" * 64, "payload digest mismatch"),
    ],
)
def test_adapter_refuses_outer_identity_and_digest_drift(
    field: str,
    value: object,
    match: str,
) -> None:
    record = _record()
    record[field] = value
    with pytest.raises(ValueError, match=match):
        coupled_transport_handoff_from_fusion_bytes(
            _encode(record, seal_payload=field != "payload_sha256")
        )


def test_adapter_refuses_noncanonical_duplicate_encoding_and_expected_digest() -> None:
    encoded = _encode(_record())
    with pytest.raises(ValueError, match="non-empty bytes"):
        coupled_transport_handoff_from_fusion_bytes(b"")
    with pytest.raises(ValueError, match="non-empty bytes"):
        coupled_transport_handoff_from_fusion_bytes(bytearray(encoded))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="byte digest mismatch"):
        coupled_transport_handoff_from_fusion_bytes(encoded, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        coupled_transport_handoff_from_fusion_bytes(encoded, expected_sha256="bad")
    with pytest.raises(ValueError, match="strict UTF-8"):
        coupled_transport_handoff_from_fusion_bytes(b"\xff")
    with pytest.raises(ValueError, match="JSON is invalid"):
        coupled_transport_handoff_from_fusion_bytes(b"{")
    with pytest.raises(ValueError, match="must be an object"):
        coupled_transport_handoff_from_fusion_bytes(b"[]")
    with pytest.raises(ValueError, match="canonical JSON"):
        coupled_transport_handoff_from_fusion_bytes(encoded + b"\n")
    with pytest.raises(ValueError, match="duplicate FUSION JSON key"):
        coupled_transport_handoff_from_fusion_bytes(b'{"schema":"a","schema":"b"}')


def test_adapter_refuses_source_above_the_public_size_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter_module, "MAX_SOURCE_ENVELOPE_BYTES", 1)

    with pytest.raises(ValueError, match="maximum byte size"):
        coupled_transport_handoff_from_fusion_bytes(b"{}")


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("payload", "clock", "domain"), "plant_monotonic", "simulation_monotonic"),
        (("payload", "clock", "kind"), "wall_clock", "simulation_monotonic"),
        (("payload", "clock", "sample_ns"), [0], "increasing samples"),
        (("payload", "clock", "sample_ns"), [0, 10, 9], "increasing samples"),
        (("payload", "clock", "sample_ns"), [0, 10, 30], "one declared cadence"),
        (("payload", "clock", "timestamp_ns"), 10_000_000, "requested final"),
        (("payload", "clock", "requested_final_ns"), 10_000_000, "requested final"),
        (("payload", "clock", "timestamp_ns"), True, "must be an integer"),
        (("payload", "clock", "sample_rate_hz"), 99.0, "does not match"),
        (("payload", "clock", "latency_s"), 10**400, "finite number"),
        (("payload", "clock", "latency_s"), 0.1, "latency must be zero"),
        (("payload", "clock", "picosecond_offset"), 1, "picosecond offset"),
        (("payload", "clock", "synchronized_to"), "wall", "cannot imply"),
        (("payload", "clock", "reset_policy"), "reuse", "reset policy"),
        (("payload", "completion", "complete"), False, "complete NO_ERROR"),
        (("payload", "completion", "sim_error"), "FAILED", "complete NO_ERROR"),
        (("payload", "completion", "reached_final_ns"), 10_000_000, "completion time"),
        (("payload", "validity", "ood"), True, "validity must be exact"),
    ],
)
def test_adapter_refuses_clock_completion_and_validity_drift(
    path: tuple[str, ...],
    value: object,
    match: str,
) -> None:
    record = _record()
    parent = _path(record, *path[:-1])
    parent[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        coupled_transport_handoff_from_fusion_bytes(_encode(record))


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("payload", "reactor", "event_id"), "other", "event_id"),
        (("payload", "reactor", "registry_version"), "9.0.0", "registry version"),
        (("payload", "reactor", "registry_digest"), "0" * 64, "registry digest"),
        (("payload", "reactor", "schema_version"), "2.0.0", "U0"),
        (("payload", "reactor", "drivers"), ["combined"], "nonredundant"),
        (("payload", "reactor", "drivers"), [], "non-empty list"),
        (("payload", "reactor", "drivers"), [1], "invalid item type"),
        (
            ("payload", "reactor", "operating_point", "fuel_class_basis"),
            "modeled_burn",
            "no-fusion/no-burn",
        ),
        (("payload", "reactor", "operating_point", "main_ion"), "T", "D main ions"),
        (("payload", "reactor", "operating_point", "major_radius_m"), 0.0, "positive"),
        (("payload", "reactor", "topology"), "wrong", "configuration"),
    ],
)
def test_adapter_refuses_reactor_identity_and_fuel_semantic_drift(
    path: tuple[str, ...],
    value: object,
    match: str,
) -> None:
    record = _record()
    parent = _path(record, *path[:-1])
    parent[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        coupled_transport_handoff_from_fusion_bytes(_encode(record))


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("payload", "observables", "rho", "frame"), "wrong", "rho coordinate"),
        (("payload", "observables", "rho", "samples"), [0.0], "strictly increasing"),
        (("payload", "observables", "rho", "samples"), [0.0, 2.0], r"\[0, 1\]"),
        (
            ("payload", "observables", "profiles", "electron_density", "unit"),
            "cm^-3",
            "unit mismatch",
        ),
        (
            ("payload", "observables", "profiles", "electron_density", "samples"),
            [[1.0, 2.0]],
            "one sample per clock",
        ),
        (
            ("payload", "observables", "profiles", "electron_density", "samples"),
            [[1.0], [2.0], [3.0]],
            "profile shape",
        ),
        (
            ("payload", "observables", "source_totals", "particles", "samples"),
            [1.0, 2.0],
            "one sample per clock",
        ),
        (
            ("payload", "observables", "source_totals", "particles", "samples"),
            [1.0, "not-finite", 3.0],
            "finite number",
        ),
        (
            (
                "payload",
                "observables",
                "profiles",
                "electron_density",
                "calibration",
                "empirical",
            ),
            True,
            "calibration semantics",
        ),
        (
            ("payload", "observables", "numerics", "inner_solver_iterations"),
            [0, -1, 1],
            "nonnegative",
        ),
        (
            ("payload", "observables", "numerics", "sawtooth_crash"),
            [False, 0, False],
            "boolean",
        ),
        (("payload", "observables", "numerics", "sim_error"), 1, "must be zero"),
        (
            ("payload", "observables", "numerics", "sim_status"),
            "running",
            "must be completed",
        ),
    ],
)
def test_adapter_refuses_observable_shape_unit_calibration_and_numeric_drift(
    path: tuple[str, ...],
    value: object,
    match: str,
) -> None:
    record = _record()
    parent = _path(record, *path[:-1])
    parent[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        coupled_transport_handoff_from_fusion_bytes(_encode(record))


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("payload", "uncertainty", "kind"), "statistical", "numerical_refinement"),
        (("payload", "uncertainty", "primary_dt_ns"), 1, "refined timestep"),
        (
            (
                "payload",
                "uncertainty",
                "observables",
                "profiles",
                "electron_density",
                "unit",
            ),
            "cm^-3",
            "uncertainty unit",
        ),
        (
            (
                "payload",
                "uncertainty",
                "observables",
                "profiles",
                "electron_density",
                "absolute_rms_difference",
            ),
            -1.0,
            "nonnegative",
        ),
        (
            (
                "payload",
                "uncertainty",
                "observables",
                "profiles",
                "electron_density",
                "relative_l2",
            ),
            "not-finite",
            "finite number",
        ),
    ],
)
def test_adapter_refuses_uncertainty_semantic_drift(
    path: tuple[str, ...],
    value: object,
    match: str,
) -> None:
    record = _record()
    parent = _path(record, *path[:-1])
    parent[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        coupled_transport_handoff_from_fusion_bytes(_encode(record))


def test_adapter_refuses_missing_extra_names_and_provenance_drift() -> None:
    record = _record()
    del record["payload"]["observables"]["profiles"]["electron_density"]
    with pytest.raises(ValueError, match="fields differ"):
        coupled_transport_handoff_from_fusion_bytes(_encode(record))

    record = _record()
    record["payload"]["uncertainty"]["observables"]["extra"] = {}
    with pytest.raises(ValueError, match="fields differ"):
        coupled_transport_handoff_from_fusion_bytes(_encode(record))

    record = _record()
    record["provenance"]["runtime_source_sha256"] = "bad"
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        coupled_transport_handoff_from_fusion_bytes(_encode(record))

    record = _record()
    record["provenance"]["extra"] = "0" * 64
    with pytest.raises(ValueError, match="fields differ"):
        coupled_transport_handoff_from_fusion_bytes(_encode(record))


def test_adapter_defensively_copies_input_before_semantic_use() -> None:
    record = _record()
    original = deepcopy(record)
    handoff = coupled_transport_handoff_from_fusion_bytes(_encode(record))

    assert record == original
    assert handoff.source_revision == SOURCE_REVISION
