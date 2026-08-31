# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupled-transport semantic adapter

"""Strict FUSION coupled-transport evidence to nonphase U0 semantics."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeVar, cast

from .contracts import (
    JsonValue,
    ObservableDescriptor,
    PhaseSemanticRecord,
    ReactorContext,
    RegimeAxis,
    RegimeEstimate,
)
from .evidence import (
    CalibrationReference,
    ClockReference,
    ProvenanceRecord,
    QualityAssessment,
    Uncertainty,
    ValidityWindow,
)
from .handoff import MAX_SOURCE_ENVELOPE_BYTES, ReactorSemanticHandoff
from .registry import DEFAULT_REACTOR_REGISTRY, ReactorConfigurationRegistry
from .vocabulary import (
    ClockKind,
    ConfinementFamily,
    ConversionKind,
    DriverKind,
    EvidenceClass,
    OperatingCadence,
    QualityState,
    ReactionKind,
    RegimeState,
    SemanticCarrier,
    ValidityState,
)

FUSION_TORAX_REVIEW_SCHEMA = "scpn-fusion-core.torax-runtime-review-envelope.v1"
FUSION_TORAX_OUTCOME_SCHEMA = "scpn-fusion-core.torax-runtime-outcome.v1"
FUSION_COUPLED_TRANSPORT_SCHEMA = (
    "scpn-fusion-core.coupled-transport-model-intersection.v1"
)
_FUSION_PROJECT = "SCPN-FUSION-CORE"
_CALIBRATION_ID = "fusion.torax.simulation_declared_units.v1"
_TRANSFER_ID = "fusion.torax.identity_projection.v1"
_FUEL_CLASS_BASIS = "deuterium_only_input_no_fusion_power_or_burn_model"
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class _ObservableSpec:
    """Static metadata for one supported TORAX observable."""

    category: str
    name: str
    physical_quantity: str
    unit: str
    support: str

    @property
    def observable_id(self) -> str:
        """Return the namespaced observable identifier."""
        return f"fusion.torax.{self.category}.{self.name}"


_SPECS = (
    _ObservableSpec(
        "profiles", "electron_density", "electron density profile", "m^-3", "rho"
    ),
    _ObservableSpec(
        "profiles", "electron_temperature", "electron temperature profile", "keV", "rho"
    ),
    _ObservableSpec(
        "profiles", "ion_temperature", "ion temperature profile", "keV", "rho"
    ),
    _ObservableSpec(
        "profiles", "poloidal_flux", "poloidal flux profile", "Wb/rad", "rho"
    ),
    _ObservableSpec(
        "source_totals",
        "driven_current",
        "total prescribed driven-current source",
        "A",
        "global_source_total",
    ),
    _ObservableSpec(
        "source_totals",
        "electron_heat",
        "total prescribed electron-heating source",
        "W",
        "global_source_total",
    ),
    _ObservableSpec(
        "source_totals",
        "ion_electron_exchange",
        "total ion-electron exchange source",
        "W",
        "global_source_total",
    ),
    _ObservableSpec(
        "source_totals",
        "ion_heat",
        "total prescribed ion-heating source",
        "W",
        "global_source_total",
    ),
    _ObservableSpec(
        "source_totals",
        "particles",
        "total prescribed particle source",
        "s^-1",
        "global_source_total",
    ),
    _ObservableSpec(
        "state_budgets",
        "particle_inventory",
        "model particle inventory budget",
        "1",
        "global_state_budget",
    ),
    _ObservableSpec(
        "state_budgets",
        "poloidal_flux_l2",
        "model poloidal-flux L2 budget",
        "Wb/rad",
        "global_state_budget",
    ),
    _ObservableSpec(
        "state_budgets",
        "thermal_energy",
        "model thermal-energy budget",
        "J",
        "global_state_budget",
    ),
)
_SPECS_BY_CATEGORY = {
    category: tuple(spec for spec in _SPECS if spec.category == category)
    for category in ("profiles", "source_totals", "state_budgets")
}
_PROVENANCE_KEYS = frozenset(
    {
        "artifact_content_sha256",
        "deck_sha256",
        "manifest_inventory_sha256",
        "model_intersection_revision",
        "primary_projection_sha256",
        "refined_projection_sha256",
        "refined_request_sha256",
        "request_sha256",
        "runner_sha256",
        "runtime_source_sha256",
    }
)


def coupled_transport_handoff_from_fusion_bytes(
    source_envelope: bytes,
    *,
    expected_sha256: str | None = None,
    registry: ReactorConfigurationRegistry = DEFAULT_REACTOR_REGISTRY,
) -> ReactorSemanticHandoff:
    """Validate canonical FUSION bytes and map twelve noncyclic quantities.

    Parameters
    ----------
    source_envelope : bytes
        Canonical FUSION TORAX review-envelope bytes.
    expected_sha256 : str | None
        Optional expected digest of the exact source bytes.
    registry : ReactorConfigurationRegistry
        Reactor registry used to validate the projected context.

    Returns
    -------
    ReactorSemanticHandoff
        Review-only handoff containing bounded-feature semantics.

    Raises
    ------
    ValueError
        If source custody, schema, payload, evidence, or context validation fails.
    """
    source, source_json, source_digest = _decode_source(
        source_envelope,
        expected_sha256=expected_sha256,
    )
    _exact_keys(
        source,
        "FUSION review envelope",
        {
            "event_id",
            "model_intersection_schema",
            "payload",
            "payload_sha256",
            "provenance",
            "schema",
            "source_revision",
            "source_schema",
        },
    )
    if source["schema"] != FUSION_TORAX_REVIEW_SCHEMA:
        raise ValueError("unsupported FUSION TORAX review schema")
    if source["source_schema"] != FUSION_TORAX_OUTCOME_SCHEMA:
        raise ValueError("unsupported FUSION TORAX outcome schema")
    if source["model_intersection_schema"] != FUSION_COUPLED_TRANSPORT_SCHEMA:
        raise ValueError("unsupported FUSION coupled-transport schema")
    source_revision = _commit(source["source_revision"], "source_revision")
    event_id = _text(source["event_id"], "event_id")
    payload = _object(source["payload"], "payload")
    payload_digest = _digest(source["payload_sha256"], "payload_sha256")
    if payload_digest != _canonical_digest(payload):
        raise ValueError("FUSION review payload digest mismatch")
    provenance = _validate_provenance(source["provenance"])
    return _build_handoff(
        payload=payload,
        provenance=provenance,
        source_json=source_json,
        source_digest=source_digest,
        source_revision=source_revision,
        payload_digest=payload_digest,
        event_id=event_id,
        registry=registry,
    )


def _build_handoff(
    *,
    payload: Mapping[str, object],
    provenance: Mapping[str, str],
    source_json: str,
    source_digest: str,
    source_revision: str,
    payload_digest: str,
    event_id: str,
    registry: ReactorConfigurationRegistry,
) -> ReactorSemanticHandoff:
    """Build a review-only U0 handoff from validated FUSION payload fields."""
    _exact_keys(
        payload,
        "payload",
        {"clock", "completion", "observables", "reactor", "uncertainty", "validity"},
    )
    clock_raw = _validate_clock(payload["clock"])
    timestamp_ns = cast(int, clock_raw["timestamp_ns"])
    sample_ns = cast(tuple[int, ...], clock_raw["sample_ns"])
    _validate_completion(payload["completion"], timestamp_ns=timestamp_ns)
    _validate_validity(payload["validity"])
    context = _build_context(payload["reactor"], event_id=event_id, registry=registry)
    observables_raw = _object(payload["observables"], "payload.observables")
    _exact_keys(
        observables_raw,
        "payload.observables",
        {"numerics", "profiles", "rho", "source_totals", "state_budgets"},
    )
    rho = _validate_rho(observables_raw["rho"], context=context)
    _validate_numerics(observables_raw["numerics"], sample_count=len(sample_ns))
    uncertainty = _validate_uncertainty(payload["uncertainty"])
    clock = ClockReference(
        domain=cast(str, clock_raw["domain"]),
        kind=ClockKind.SIMULATION_MONOTONIC,
        epoch=cast(str, clock_raw["epoch"]),
        timestamp_ns=timestamp_ns,
        sample_rate_hz=cast(float, clock_raw["sample_rate_hz"]),
        latency_s=cast(float, clock_raw["latency_s"]),
        picosecond_offset=cast(int, clock_raw["picosecond_offset"]),
        synchronized_to=None,
    )
    observables: list[ObservableDescriptor] = []
    for spec in _SPECS:
        category = _object(
            observables_raw[spec.category], f"observables.{spec.category}"
        )
        expected_names = {item.name for item in _SPECS_BY_CATEGORY[spec.category]}
        _exact_keys(category, f"observables.{spec.category}", expected_names)
        item = _object(category[spec.name], f"{spec.category}.{spec.name}")
        value, calibration = _validate_observable_item(
            item,
            spec=spec,
            sample_ns=sample_ns,
            rho_count=len(rho),
        )
        metric_category = _object(uncertainty["observables"], "uncertainty.observables")
        metrics = _object(
            metric_category[spec.category], f"uncertainty.{spec.category}"
        )
        metric = _validate_metric(metrics[spec.name], spec=spec)
        spatial_support = spec.support
        if spec.support == "rho":
            spatial_support = _canonical_json(
                {"coordinate": "rho_norm", "unit": "1", "values": list(rho)}
            )
        observables.append(
            ObservableDescriptor(
                observable_id=spec.observable_id,
                reactor_context=context,
                physical_quantity=spec.physical_quantity,
                units=spec.unit,
                coordinate_frame=context.coordinate_frame,
                spatial_support=spatial_support,
                diagnostic="torax_coupled_transport",
                channel=f"{spec.category}.{spec.name}",
                value=value,
                clock=clock,
                calibration=calibration,
                uncertainty=Uncertainty(
                    standard_deviation=cast(float, metric["absolute_rms_difference"]),
                    confidence_level=0.0,
                ),
                quality=QualityAssessment(QualityState.VALID),
                validity=ValidityWindow(
                    ValidityState.VALID,
                    valid_from_ns=timestamp_ns,
                    valid_until_ns=timestamp_ns,
                ),
                provenance=_observable_provenance(
                    spec=spec,
                    provenance=provenance,
                    source_digest=source_digest,
                    source_revision=source_revision,
                    payload_digest=payload_digest,
                    event_id=event_id,
                    metric=metric,
                ),
            )
        )
    observable_tuple = tuple(observables)
    semantics = tuple(_bounded_semantic(item) for item in observable_tuple)
    regime = RegimeEstimate(
        regime_id="spo.coupled_transport.regime.unknown",
        reactor_context_id=context.context_id,
        axes=(RegimeAxis("operational_regime", "unknown", 0.0),),
        state=RegimeState.UNKNOWN,
        evidence_ids=tuple(item.observable_id for item in observable_tuple),
        classifier="spo.coupled_transport.no_classifier",
        classifier_version="1.0.0",
        threshold_provenance=("no_regime_classifier_declared",),
        confidence=0.0,
        hysteresis=0.0,
        dwell_time_s=0.0,
        transition_reason="transport evidence does not identify a reactor regime",
        safety_effect="review only; no control consequence",
        validity=ValidityWindow(
            ValidityState.UNKNOWN,
            valid_from_ns=timestamp_ns,
            valid_until_ns=timestamp_ns,
            reasons=("no regime classifier declared by producer",),
        ),
    )
    return ReactorSemanticHandoff(
        source_schema=FUSION_TORAX_REVIEW_SCHEMA,
        source_revision=source_revision,
        source_envelope_json=source_json,
        event_id=event_id,
        context=context,
        observables=observable_tuple,
        semantics=semantics,
        phase_relations=(),
        regime=regime,
    )


def _build_context(
    value: object,
    *,
    event_id: str,
    registry: ReactorConfigurationRegistry,
) -> ReactorContext:
    """Validate and construct the TORAX reactor context."""
    reactor = _object(value, "payload.reactor")
    _exact_keys(
        reactor,
        "payload.reactor",
        {
            "cadence",
            "configuration",
            "configuration_version",
            "confinement_family",
            "context_id",
            "conversion",
            "coordinate_frame",
            "drivers",
            "evidence_class",
            "event_id",
            "facility",
            "operating_point",
            "reaction",
            "registry_digest",
            "registry_version",
            "schema_version",
            "topology",
        },
    )
    if reactor["event_id"] != event_id:
        raise ValueError("FUSION reactor event_id does not match the envelope")
    if reactor["registry_version"] != registry.version:
        raise ValueError("FUSION reactor registry version does not match SPO")
    if reactor["registry_digest"] != registry.digest:
        raise ValueError("FUSION reactor registry digest does not match SPO")
    if reactor["schema_version"] != "1.0.0":
        raise ValueError("unsupported U0 reactor schema version")
    drivers = _string_list(reactor["drivers"], "reactor.drivers")
    if drivers != ("external_magnetic_coils", "plasma_current"):
        raise ValueError("TORAX driver set must name its two nonredundant components")
    operating_point = _object(reactor["operating_point"], "reactor.operating_point")
    _exact_keys(
        operating_point,
        "reactor.operating_point",
        {
            "effective_charge",
            "fuel_class_basis",
            "impurity",
            "magnetic_field_t",
            "main_ion",
            "major_radius_m",
            "minor_radius_m",
            "plasma_current_a",
        },
    )
    for field in (
        "effective_charge",
        "magnetic_field_t",
        "major_radius_m",
        "minor_radius_m",
        "plasma_current_a",
    ):
        if _number(operating_point[field], f"operating_point.{field}") <= 0.0:
            raise ValueError(f"operating_point.{field} must be positive")
    if operating_point["main_ion"] != "D" or operating_point["impurity"] != "Ne":
        raise ValueError("frozen TORAX evidence requires D main ions and Ne impurity")
    if operating_point["fuel_class_basis"] != _FUEL_CLASS_BASIS:
        raise ValueError("D-D must be declared as a no-fusion/no-burn fuel class")
    context = ReactorContext(
        context_id=_text(reactor["context_id"], "reactor.context_id"),
        configuration=_text(reactor["configuration"], "reactor.configuration"),
        confinement_family=ConfinementFamily(
            _text(reactor["confinement_family"], "reactor.confinement_family")
        ),
        topology=_text(reactor["topology"], "reactor.topology"),
        coordinate_frame=_text(reactor["coordinate_frame"], "reactor.coordinate_frame"),
        drivers=tuple(DriverKind(item) for item in drivers),
        cadence=OperatingCadence(_text(reactor["cadence"], "reactor.cadence")),
        reaction=ReactionKind(_text(reactor["reaction"], "reactor.reaction")),
        conversion=ConversionKind(_text(reactor["conversion"], "reactor.conversion")),
        facility=_text(reactor["facility"], "reactor.facility"),
        event_id=event_id,
        configuration_version=_text(
            reactor["configuration_version"], "reactor.configuration_version"
        ),
        operating_point=cast(Mapping[str, JsonValue], operating_point),
        evidence_class=EvidenceClass(
            _text(reactor["evidence_class"], "reactor.evidence_class")
        ),
        registry_version=registry.version,
        registry_digest=registry.digest,
        schema_version=reactor["schema_version"],
    )
    return context.validate_registry(registry)


def _validate_clock(value: object) -> Mapping[str, object]:
    """Validate the fixed-cadence simulation clock and return normalized fields."""
    clock = _object(value, "payload.clock")
    _exact_keys(
        clock,
        "payload.clock",
        {
            "domain",
            "epoch",
            "kind",
            "latency_s",
            "picosecond_offset",
            "requested_final_ns",
            "reset_policy",
            "sample_ns",
            "sample_rate_hz",
            "synchronized_to",
            "timestamp_ns",
        },
    )
    if (
        clock["domain"] != "simulation_monotonic"
        or clock["kind"] != "simulation_monotonic"
    ):
        raise ValueError("FUSION clock must be simulation_monotonic")
    _text(clock["epoch"], "clock.epoch")
    samples = _integer_list(clock["sample_ns"], "clock.sample_ns")
    if len(samples) < 2 or any(
        right <= left for left, right in zip(samples, samples[1:], strict=False)
    ):
        raise ValueError("clock.sample_ns must contain increasing samples")
    intervals = tuple(
        right - left for left, right in zip(samples, samples[1:], strict=False)
    )
    if len(set(intervals)) != 1:
        raise ValueError("clock.sample_ns must use one declared cadence")
    timestamp = _integer(clock["timestamp_ns"], "clock.timestamp_ns")
    requested = _integer(clock["requested_final_ns"], "clock.requested_final_ns")
    if timestamp != samples[-1] or requested != samples[-1]:
        raise ValueError("FUSION clock must reach its requested final sample")
    rate = _number(clock["sample_rate_hz"], "clock.sample_rate_hz")
    expected_rate = 1_000_000_000.0 / intervals[0]
    if rate <= 0.0 or not math.isclose(rate, expected_rate, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("clock.sample_rate_hz does not match every sample interval")
    if _number(clock["latency_s"], "clock.latency_s") != 0.0:
        raise ValueError("direct simulation projection latency must be zero")
    if _integer(clock["picosecond_offset"], "clock.picosecond_offset") != 0:
        raise ValueError("TORAX projection does not declare a picosecond offset")
    if clock["synchronized_to"] is not None:
        raise ValueError(
            "simulation-monotonic TORAX evidence cannot imply synchronization"
        )
    if clock["reset_policy"] != "fresh_process_no_hidden_state":
        raise ValueError("unsupported TORAX reset policy")
    return {**clock, "sample_ns": samples, "sample_rate_hz": rate}


def _validate_completion(value: object, *, timestamp_ns: int) -> None:
    """Require a successful run completed at the final clock sample."""
    completion = _object(value, "payload.completion")
    _exact_keys(
        completion, "payload.completion", {"complete", "reached_final_ns", "sim_error"}
    )
    if completion["complete"] is not True or completion["sim_error"] != "NO_ERROR":
        raise ValueError("FUSION review requires a complete NO_ERROR run")
    if (
        _integer(completion["reached_final_ns"], "completion.reached_final_ns")
        != timestamp_ns
    ):
        raise ValueError("completion time does not match the final clock sample")


def _validate_validity(value: object) -> None:
    """Require the exact in-distribution review-only validity declaration."""
    validity = _object(value, "payload.validity")
    _exact_keys(validity, "payload.validity", {"authority", "ood", "quality", "state"})
    expected = {
        "authority": "review_only_non_actuating",
        "ood": False,
        "quality": "frozen_model_intersection_reference",
        "state": "VALID",
    }
    if validity != expected:
        raise ValueError(
            "FUSION validity must be exact, in-distribution, and review-only"
        )


def _validate_rho(value: object, *, context: ReactorContext) -> tuple[float, ...]:
    """Validate normalized radial support against the reactor coordinate frame."""
    rho = _object(value, "observables.rho")
    _exact_keys(rho, "observables.rho", {"frame", "name", "samples", "unit"})
    if (
        rho["frame"] != context.coordinate_frame
        or rho["name"] != "rho_norm"
        or rho["unit"] != "1"
    ):
        raise ValueError("rho coordinate identity, frame, or unit drifted")
    samples = _number_list(rho["samples"], "rho.samples")
    if len(samples) < 2 or any(
        right <= left for left, right in zip(samples, samples[1:], strict=False)
    ):
        raise ValueError("rho.samples must be strictly increasing")
    if samples[0] < 0.0 or samples[-1] > 1.0:
        raise ValueError("rho_norm samples must lie in [0, 1]")
    return samples


def _validate_numerics(value: object, *, sample_count: int) -> None:
    """Validate per-sample solver status and iteration evidence."""
    numerics = _object(value, "observables.numerics")
    _exact_keys(
        numerics,
        "observables.numerics",
        {
            "inner_solver_iterations",
            "outer_solver_iterations",
            "sawtooth_crash",
            "sim_error",
            "sim_status",
        },
    )
    for field in ("inner_solver_iterations", "outer_solver_iterations"):
        items = _integer_list(numerics[field], f"numerics.{field}")
        if len(items) != sample_count or any(item < 0 for item in items):
            raise ValueError(
                f"numerics.{field} must be nonnegative per-sample integers"
            )
    crashes = _list(numerics["sawtooth_crash"], "numerics.sawtooth_crash")
    if len(crashes) != sample_count or any(
        not isinstance(item, bool) for item in crashes
    ):
        raise ValueError("numerics.sawtooth_crash must be one boolean per sample")
    if _integer(numerics["sim_error"], "numerics.sim_error") != 0:
        raise ValueError("numerics.sim_error must be zero")
    if numerics["sim_status"] != "completed":
        raise ValueError("numerics.sim_status must be completed")


def _validate_uncertainty(value: object) -> Mapping[str, object]:
    """Validate the timestep-refinement uncertainty inventory."""
    uncertainty = _object(value, "payload.uncertainty")
    _exact_keys(
        uncertainty,
        "payload.uncertainty",
        {"kind", "observables", "primary_dt_ns", "refined_dt_ns"},
    )
    if uncertainty["kind"] != "numerical_refinement":
        raise ValueError("uncertainty must be numerical_refinement")
    primary_dt = _integer(uncertainty["primary_dt_ns"], "uncertainty.primary_dt_ns")
    refined_dt = _integer(uncertainty["refined_dt_ns"], "uncertainty.refined_dt_ns")
    if refined_dt <= 0 or primary_dt <= refined_dt:
        raise ValueError("refined timestep must be positive and smaller than primary")
    categories = _object(uncertainty["observables"], "uncertainty.observables")
    _exact_keys(categories, "uncertainty.observables", set(_SPECS_BY_CATEGORY))
    for category, specs in _SPECS_BY_CATEGORY.items():
        metrics = _object(categories[category], f"uncertainty.{category}")
        _exact_keys(metrics, f"uncertainty.{category}", {spec.name for spec in specs})
    return uncertainty


def _validate_observable_item(
    item: Mapping[str, object],
    *,
    spec: _ObservableSpec,
    sample_ns: tuple[int, ...],
    rho_count: int,
) -> tuple[JsonValue, CalibrationReference]:
    """Validate one observable series and return its final value and calibration."""
    _exact_keys(
        item, f"{spec.category}.{spec.name}", {"calibration", "samples", "unit"}
    )
    if item["unit"] != spec.unit:
        raise ValueError(f"unit mismatch for {spec.category}.{spec.name}")
    calibration = _object(item["calibration"], f"{spec.name}.calibration")
    _exact_keys(
        calibration,
        f"{spec.name}.calibration",
        {
            "basis",
            "calibrated_at_ns",
            "calibration_id",
            "empirical",
            "transfer",
            "transfer_function_id",
        },
    )
    expected = {
        "basis": "simulation_declared_units",
        "calibrated_at_ns": sample_ns[0],
        "calibration_id": _CALIBRATION_ID,
        "empirical": False,
        "transfer": "identity",
        "transfer_function_id": _TRANSFER_ID,
    }
    if calibration != expected:
        raise ValueError(f"calibration semantics drifted for {spec.name}")
    samples = _list(item["samples"], f"{spec.name}.samples")
    if len(samples) != len(sample_ns):
        raise ValueError(f"{spec.name} must contain one sample per clock timestamp")
    if spec.category == "profiles":
        rows = tuple(_number_list(row, f"{spec.name}.samples row") for row in samples)
        if any(len(row) != rho_count for row in rows):
            raise ValueError(f"{spec.name} profile shape does not match rho support")
        value: JsonValue = cast(JsonValue, rows[-1])
    else:
        values = _number_list(samples, f"{spec.name}.samples")
        value = values[-1]
    return value, CalibrationReference(
        calibration_id=_CALIBRATION_ID,
        transfer_function_id=_TRANSFER_ID,
        calibrated_at_ns=sample_ns[0],
    )


def _validate_metric(value: object, *, spec: _ObservableSpec) -> Mapping[str, object]:
    """Validate nonnegative refinement metrics for one observable."""
    metric = _object(value, f"uncertainty.{spec.category}.{spec.name}")
    _exact_keys(
        metric,
        f"uncertainty.{spec.name}",
        {"absolute_rms_difference", "relative_l2", "unit"},
    )
    if metric["unit"] != spec.unit:
        raise ValueError(f"uncertainty unit mismatch for {spec.name}")
    absolute = _number(
        metric["absolute_rms_difference"], f"{spec.name}.absolute_rms_difference"
    )
    relative = _number(metric["relative_l2"], f"{spec.name}.relative_l2")
    if absolute < 0.0 or relative < 0.0:
        raise ValueError(f"uncertainty metrics for {spec.name} must be nonnegative")
    return {**metric, "absolute_rms_difference": absolute, "relative_l2": relative}


def _validate_provenance(value: object) -> Mapping[str, str]:
    """Validate the complete FUSION revision and digest inventory."""
    provenance = _object(value, "provenance")
    _exact_keys(provenance, "provenance", set(_PROVENANCE_KEYS))
    result: dict[str, str] = {}
    for field in sorted(_PROVENANCE_KEYS):
        if field == "model_intersection_revision":
            result[field] = _commit(provenance[field], f"provenance.{field}")
        else:
            result[field] = _digest(provenance[field], f"provenance.{field}")
    return result


def _observable_provenance(
    *,
    spec: _ObservableSpec,
    provenance: Mapping[str, str],
    source_digest: str,
    source_revision: str,
    payload_digest: str,
    event_id: str,
    metric: Mapping[str, object],
) -> ProvenanceRecord:
    """Build custody metadata for one projected FUSION observable."""
    attributes = {
        "calibration_basis": "simulation_declared_units",
        "calibration_empirical": "false",
        "event_id": event_id,
        "fuel_class_basis": _FUEL_CLASS_BASIS,
        "model_intersection_revision": provenance["model_intersection_revision"],
        "numerical_uncertainty": (
            "timestep_refinement_absolute_rms_difference_"
            "not_statistical_or_empirical_confidence"
        ),
        "payload_sha256": payload_digest,
        "producer_revision": source_revision,
        "relative_l2": format(cast(float, metric["relative_l2"]), ".17g"),
        "runtime_source_sha256": provenance["runtime_source_sha256"],
        "source_envelope_sha256": source_digest,
        "transfer": "identity",
    }
    return ProvenanceRecord(
        source_project=_FUSION_PROJECT,
        component="scpn_fusion.integrations.torax.review",
        symbol="ToraxReviewEnvelope",
        artifact_uri=f"artifact:sha256:{provenance['artifact_content_sha256']}",
        sha256=source_digest,
        attributes=tuple(attributes.items()),
    )


def _bounded_semantic(observable: ObservableDescriptor) -> PhaseSemanticRecord:
    """Represent a noncyclic transport observable as a bounded-feature semantic."""
    timestamp_ns = observable.clock.timestamp_ns
    return PhaseSemanticRecord(
        phase_id=f"spo.transport.{observable.observable_id.removeprefix('fusion.torax.')}.bounded_feature",
        reactor_context_id=observable.reactor_context.context_id,
        observable_ids=(observable.observable_id,),
        carrier_type=SemanticCarrier.BOUNDED_FEATURE,
        phenomenon=observable.physical_quantity,
        phase_rad=None,
        amplitude=None,
        frequency_hz=None,
        bandwidth_hz=None,
        mode_identity=None,
        mode_harmonic=None,
        phase_origin=None,
        orientation=None,
        reference_frame=observable.coordinate_frame,
        clock_domain=observable.clock.domain,
        clock_kind=observable.clock.kind,
        clock_epoch=observable.clock.epoch,
        wrap_convention=None,
        reference_signal=None,
        extractor="spo.coupled_transport.bounded_feature",
        extractor_version="1.0.0",
        observation_operator=None,
        uncertainty=observable.uncertainty,
        confidence=0.0,
        observability=0.0,
        observability_threshold=1.0,
        validity=ValidityWindow(
            ValidityState.UNOBSERVABLE,
            valid_from_ns=timestamp_ns,
            valid_until_ns=timestamp_ns,
            reasons=("no cyclic phase observable declared by producer",),
        ),
        quality=QualityAssessment(
            QualityState.UNKNOWN,
            flags=("noncyclic_transport_evidence",),
        ),
        evidence_class=observable.reactor_context.evidence_class,
    )


def _decode_source(
    payload: bytes,
    *,
    expected_sha256: str | None,
) -> tuple[Mapping[str, object], str, str]:
    """Decode and authenticate a canonical FUSION source envelope."""
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("FUSION source envelope must be non-empty bytes")
    if len(payload) > MAX_SOURCE_ENVELOPE_BYTES:
        raise ValueError("FUSION source envelope exceeds the maximum byte size")
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != _digest(
        expected_sha256, "expected_sha256"
    ):
        raise ValueError("FUSION source envelope byte digest mismatch")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("FUSION source envelope must be strict UTF-8") from exc
    try:
        raw = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError("FUSION source envelope JSON is invalid") from exc
    source = _object(raw, "FUSION source envelope")
    if _canonical_json(source) != text:
        raise ValueError("FUSION source envelope must use canonical JSON")
    return source, text, digest


def _canonical_json(value: object) -> str:
    """Return the adapter's canonical compact JSON representation."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_digest(value: object) -> str:
    """Return SHA-256 of the canonical compact JSON representation."""
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _unique_object(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting duplicate keys."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate FUSION JSON key: {key}")
        result[key] = value
    return result


def _object(value: object, label: str) -> Mapping[str, object]:
    """Return a string-keyed mapping or reject the value."""
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _exact_keys(value: Mapping[str, object], label: str, expected: set[str]) -> None:
    """Require an exact object field set."""
    if set(value) != expected:
        raise ValueError(
            f"{label} fields differ; missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _list(value: object, label: str) -> list[object]:
    """Return a non-empty list or reject the value."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    return value


def _typed_list(value: object, label: str, expected: type[_T]) -> tuple[_T, ...]:
    """Return a non-empty tuple whose items match the expected runtime type."""
    items = _list(value, label)
    if any(not isinstance(item, expected) for item in items):
        raise ValueError(f"{label} contains an invalid item type")
    return cast(tuple[_T, ...], tuple(items))


def _string_list(value: object, label: str) -> tuple[str, ...]:
    """Return a validated non-empty tuple of strings."""
    return _typed_list(value, label, str)


def _integer_list(value: object, label: str) -> tuple[int, ...]:
    """Return a validated non-empty tuple of non-boolean integers."""
    items = _list(value, label)
    return tuple(_integer(item, label) for item in items)


def _number_list(value: object, label: str) -> tuple[float, ...]:
    """Return a validated non-empty tuple of finite numbers."""
    items = _list(value, label)
    return tuple(_number(item, label) for item in items)


def _text(value: object, label: str) -> str:
    """Return a non-empty string or reject the value."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    """Return a non-boolean integer or reject the value."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    """Return a finite non-boolean number as a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError:
        result = math.inf
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _digest(value: object, label: str) -> str:
    """Return a lowercase SHA-256 digest or reject the value."""
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _commit(value: object, label: str) -> str:
    """Return a lowercase 40-character Git revision or reject the value."""
    if not isinstance(value, str) or _HEX_40.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase Git revision")
    return value


__all__ = [
    "FUSION_COUPLED_TRANSPORT_SCHEMA",
    "FUSION_TORAX_OUTCOME_SCHEMA",
    "FUSION_TORAX_REVIEW_SCHEMA",
    "coupled_transport_handoff_from_fusion_bytes",
]
