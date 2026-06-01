# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Information-geometry control fixtures

"""Information geometry control scenarios for non-actuating review."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

InformationGeometryBoundary: Final[str] = (
    "information_geometry_control_not_live_actuation"
)
_SUPPORTED_DOMAINS: Final[tuple[str, ...]] = (
    "power_grid",
    "cardiac_rhythm",
    "cyber_industrial",
    "traffic_flow",
)


def _ensure_float64_vector(values: Iterable[float], *, label: str) -> FloatArray:
    if isinstance(values, (bool, np.bool_)) or _contains_boolean_alias(values):
        raise ValueError(f"{label} must contain numeric values")
    if _contains_complex_alias(values):
        raise ValueError(f"{label} must contain real-valued numeric values")
    arr = np.asarray(tuple(values), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be a 1D array, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{label} must not be empty")
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f"{label} must be float64 numeric values")
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} must contain only finite values")
    if np.any(arr < 0.0):
        raise ValueError(f"{label} must contain only non-negative values")
    if float(np.sum(arr)) <= 0.0:
        raise ValueError(f"{label} must have positive total mass")
    return arr


def _distribution_summary(values: FloatArray) -> dict[str, float]:
    return {
        "count": int(values.size),
        "sum": float(np.sum(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _ensure_positive_finite_scalar(value: object, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must not be a boolean")
    if not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{label} must be a numeric value")
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    if float(value) <= 0.0:
        raise ValueError(f"{label} must be strictly positive")
    return float(value)


def _validate_domain(domain: str) -> str:
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("domain must be a non-empty string")
    if domain not in _SUPPORTED_DOMAINS:
        raise ValueError(f"unsupported domain '{domain}'")
    return domain


def _contains_boolean_alias(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in array.flat)


def _contains_complex_alias(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (complex, np.complexfloating)) for item in array.flat)


@dataclass(frozen=True)
class DistributionPair:
    """Discrete distribution pair describing review-only geometry transfer targets."""

    current_distribution: FloatArray
    target_distribution: FloatArray

    @property
    def current_summary(self) -> dict[str, float]:
        """Return summary statistics for the current distribution."""
        return _distribution_summary(self.current_distribution)

    @property
    def target_summary(self) -> dict[str, float]:
        """Return summary statistics for the target distribution."""
        return _distribution_summary(self.target_distribution)

    def to_record(self) -> dict[str, list[float]]:
        """Return a deterministic JSON-safe record."""
        return {
            "current_distribution": self.current_distribution.tolist(),
            "target_distribution": self.target_distribution.tolist(),
        }


@dataclass(frozen=True)
class InformationGeometryScenario:
    """Deterministic control scenario for information-geometry review fixtures."""

    domain: str
    scenario_id: str
    distributions: DistributionPair
    objective_labels: tuple[str, ...]
    control_gradient: tuple[tuple[str, float], ...]
    max_step: float
    knob_hints: tuple[str, ...] = ()
    non_actuating: bool = True
    execution_disabled: bool = True
    claim_boundary: str = InformationGeometryBoundary

    def scenario_hash(self) -> str:
        """Return the deterministic scenario digest."""
        return _compute_scenario_hash(
            domain=self.domain,
            scenario_id=self.scenario_id,
            distributions=self.distributions,
            objective_labels=self.objective_labels,
            control_gradient=self.control_gradient,
            knob_hints=self.knob_hints,
            max_step=self.max_step,
            non_actuating=self.non_actuating,
            execution_disabled=self.execution_disabled,
            claim_boundary=self.claim_boundary,
        )

    def to_audit_record(self) -> dict[str, object]:
        """Return a deterministic JSON-safe audit record."""
        distributions = self.distributions.to_record()
        return {
            "domain": self.domain,
            "scenario_id": self.scenario_id,
            "scenario_hash": self.scenario_hash(),
            "non_actuating": self.non_actuating,
            "execution_disabled": self.execution_disabled,
            "claim_boundary": self.claim_boundary,
            "objective_labels": list(self.objective_labels),
            "max_step": float(self.max_step),
            "control_gradient": [
                [knob, float(value)] for knob, value in self.control_gradient
            ],
            "knob_hints": list(self.knob_hints),
            "current_distribution": distributions["current_distribution"],
            "target_distribution": distributions["target_distribution"],
            "current_distribution_summary": self.distributions.current_summary,
            "target_distribution_summary": self.distributions.target_summary,
        }


def _compute_scenario_hash(
    *,
    domain: str,
    scenario_id: str,
    distributions: DistributionPair,
    objective_labels: tuple[str, ...],
    control_gradient: tuple[tuple[str, float], ...],
    knob_hints: tuple[str, ...],
    max_step: float,
    non_actuating: bool,
    execution_disabled: bool,
    claim_boundary: str,
) -> str:
    canonical = {
        "domain": domain,
        "scenario_id": scenario_id,
        "claim_boundary": claim_boundary,
        "objective_labels": list(objective_labels),
        "max_step": float(max_step),
        "control_gradient": [[knob, float(value)] for knob, value in control_gradient],
        "knob_hints": list(knob_hints),
        "non_actuating": non_actuating,
        "execution_disabled": execution_disabled,
        "current_distribution": [
            float(value) for value in distributions.current_distribution.tolist()
        ],
        "target_distribution": [
            float(value) for value in distributions.target_distribution.tolist()
        ],
    }
    payload = json.dumps(
        canonical,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_distribution_pair(distributions: DistributionPair) -> None:
    if not isinstance(distributions, DistributionPair):
        raise ValueError("distributions must be a DistributionPair")

    current = _ensure_float64_vector(
        distributions.current_distribution,
        label="current_distribution",
    )
    target = _ensure_float64_vector(
        distributions.target_distribution,
        label="target_distribution",
    )
    if current.shape != target.shape:
        raise ValueError("current and target distributions must have matching shape")


def _validate_control_gradient(control_gradient: tuple[tuple[str, float], ...]) -> None:
    if not isinstance(control_gradient, tuple) or len(control_gradient) == 0:
        raise ValueError("control_gradient must be a non-empty tuple")
    for knob_name, knob_value in control_gradient:
        if not isinstance(knob_name, str) or not knob_name.strip():
            raise ValueError("control_gradient keys must be non-empty strings")
        if isinstance(knob_value, (bool, np.bool_)) or not isinstance(
            knob_value, (int, float, np.floating)
        ):
            raise ValueError("control_gradient values must be finite numbers")
        if not math.isfinite(float(knob_value)):
            raise ValueError("control_gradient values must be finite")


def _validate_information_geometry_scenario(
    scenario: InformationGeometryScenario,
) -> None:
    if not isinstance(scenario, InformationGeometryScenario):
        raise ValueError("scenario must be an InformationGeometryScenario")

    _validate_domain(scenario.domain)
    if not isinstance(scenario.scenario_id, str) or not scenario.scenario_id.strip():
        raise ValueError("scenario_id must be a non-empty string")
    if (
        not isinstance(scenario.objective_labels, tuple)
        or len(scenario.objective_labels) == 0
    ):
        raise ValueError("objective_labels must be a non-empty tuple")
    if not all(
        isinstance(label, str) and label.strip() for label in scenario.objective_labels
    ):
        raise ValueError("objective_labels must contain non-empty strings")

    _validate_distribution_pair(scenario.distributions)
    _validate_control_gradient(scenario.control_gradient)

    _ensure_positive_finite_scalar(
        scenario.max_step,
        label=f"{scenario.scenario_id}.max_step",
    )

    if scenario.non_actuating is not True:
        raise ValueError(f"{scenario.scenario_id} requires non_actuating=True")
    if scenario.execution_disabled is not True:
        raise ValueError(f"{scenario.scenario_id} requires execution_disabled=True")
    if scenario.claim_boundary != InformationGeometryBoundary:
        raise ValueError(
            f"{scenario.scenario_id} requires claim_boundary="
            f"{InformationGeometryBoundary}"
        )

    if scenario.scenario_hash() != _compute_scenario_hash(
        domain=scenario.domain,
        scenario_id=scenario.scenario_id,
        distributions=scenario.distributions,
        objective_labels=scenario.objective_labels,
        control_gradient=scenario.control_gradient,
        knob_hints=scenario.knob_hints,
        max_step=scenario.max_step,
        non_actuating=scenario.non_actuating,
        execution_disabled=scenario.execution_disabled,
        claim_boundary=scenario.claim_boundary,
    ):
        raise ValueError(f"{scenario.scenario_id} has invalid scenario_hash")


def _validate_scenario_record(record: dict[str, object]) -> None:
    for key in (
        "domain",
        "scenario_id",
        "scenario_hash",
        "non_actuating",
        "execution_disabled",
        "claim_boundary",
        "objective_labels",
        "max_step",
        "control_gradient",
        "knob_hints",
        "current_distribution",
        "target_distribution",
    ):
        if key not in record:
            raise ValueError(f"missing required record field '{key}'")

    if not isinstance(record["domain"], str):
        raise ValueError("record domain must be a string")
    if not isinstance(record["scenario_id"], str):
        raise ValueError("record scenario_id must be a string")
    if not isinstance(record["objective_labels"], list):
        raise ValueError("record objective_labels must be a list")
    if not isinstance(record["control_gradient"], list):
        raise ValueError("record control_gradient must be a list")
    if not isinstance(record["knob_hints"], list):
        raise ValueError("record knob_hints must be a list")
    if not isinstance(record["non_actuating"], bool):
        raise ValueError("record non_actuating must be a boolean")
    if not isinstance(record["execution_disabled"], bool):
        raise ValueError("record execution_disabled must be a boolean")
    if not isinstance(record["claim_boundary"], str):
        raise ValueError("record claim_boundary must be a string")

    if not isinstance(record["current_distribution"], list | tuple):
        raise ValueError("record current_distribution must be a sequence")
    if not isinstance(record["target_distribution"], list | tuple):
        raise ValueError("record target_distribution must be a sequence")
    max_step_raw = record["max_step"]
    if not isinstance(max_step_raw, int | float) or isinstance(
        max_step_raw, (bool, np.bool_)
    ):
        raise ValueError("record max_step must be numeric")
    current_distribution = _ensure_float64_vector(
        record["current_distribution"],
        label=f"{record['scenario_id']}.current_distribution",
    )
    target_distribution = _ensure_float64_vector(
        record["target_distribution"],
        label=f"{record['scenario_id']}.target_distribution",
    )
    objective_labels = tuple(record["objective_labels"])
    if not all(isinstance(label, str) for label in objective_labels):
        raise ValueError("record objective_labels must contain strings")
    knob_hints = tuple(record["knob_hints"])
    if not all(isinstance(label, str) for label in knob_hints):
        raise ValueError("record knob_hints must contain strings")
    control_gradient: list[tuple[str, float]] = []
    for item in record["control_gradient"]:
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise ValueError("record control_gradient entries must be pairs")
        knob, value = item
        if not isinstance(knob, str):
            raise ValueError("record control_gradient knobs must be strings")
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError("record control_gradient values must be numeric")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError("record control_gradient values must be finite")
        control_gradient.append((knob, numeric_value))

    scenario = InformationGeometryScenario(
        domain=record["domain"],
        scenario_id=record["scenario_id"],
        distributions=DistributionPair(
            current_distribution,
            target_distribution,
        ),
        objective_labels=objective_labels,
        control_gradient=tuple(control_gradient),
        knob_hints=knob_hints,
        max_step=float(max_step_raw),
        non_actuating=record["non_actuating"],
        execution_disabled=record["execution_disabled"],
        claim_boundary=record["claim_boundary"],
    )
    _validate_information_geometry_scenario(scenario)

    expected_hash = scenario.scenario_hash()
    if not isinstance(record["scenario_hash"], str):
        raise ValueError("record scenario_hash must be a string")
    if record["scenario_hash"] != expected_hash:
        raise ValueError(f"record {scenario.scenario_id} has invalid scenario_hash")


def _build_static_scenarios() -> tuple[InformationGeometryScenario, ...]:
    return (
        InformationGeometryScenario(
            domain="power_grid",
            scenario_id="power_grid_information_geometry_control_v1",
            distributions=DistributionPair(
                current_distribution=_ensure_float64_vector(
                    [0.16, 0.27, 0.18, 0.39],
                    label="power_grid current_distribution",
                ),
                target_distribution=_ensure_float64_vector(
                    [0.21, 0.23, 0.27, 0.29],
                    label="power_grid target_distribution",
                ),
            ),
            objective_labels=("load_stability", "frequency_damping", "line_resilience"),
            control_gradient=(("K", 0.041), ("zeta", 0.012), ("rho", -0.008)),
            max_step=0.08,
            knob_hints=("sparsity_bias", "pairing_penalty"),
        ),
        InformationGeometryScenario(
            domain="cardiac_rhythm",
            scenario_id="cardiac_rhythm_information_geometry_control_v1",
            distributions=DistributionPair(
                current_distribution=_ensure_float64_vector(
                    [0.18, 0.21, 0.25, 0.19, 0.17],
                    label="cardiac_rhythm current_distribution",
                ),
                target_distribution=_ensure_float64_vector(
                    [0.20, 0.18, 0.23, 0.21, 0.18],
                    label="cardiac_rhythm target_distribution",
                ),
            ),
            objective_labels=(
                "rhythm_regulation",
                "arrhythmia_reduction",
                "hemodynamic_support",
            ),
            control_gradient=(("K", 0.035), ("eta", 0.005), ("tau", -0.004)),
            max_step=0.05,
            knob_hints=("refractory_limit", "rate_shaping"),
        ),
        InformationGeometryScenario(
            domain="cyber_industrial",
            scenario_id="cyber_industrial_information_geometry_control_v1",
            distributions=DistributionPair(
                current_distribution=_ensure_float64_vector(
                    [0.23, 0.17, 0.14, 0.22, 0.24],
                    label="cyber_industrial current_distribution",
                ),
                target_distribution=_ensure_float64_vector(
                    [0.20, 0.20, 0.20, 0.20, 0.20],
                    label="cyber_industrial target_distribution",
                ),
            ),
            objective_labels=(
                "attack_surface_reduction",
                "latency_safety",
                "service_uptime",
            ),
            control_gradient=(("gamma", 0.022), ("delta", -0.006), ("lambda", 0.011)),
            max_step=0.04,
            knob_hints=("zone_isolation", "adaptive_gate"),
        ),
        InformationGeometryScenario(
            domain="traffic_flow",
            scenario_id="traffic_flow_information_geometry_control_v1",
            distributions=DistributionPair(
                current_distribution=_ensure_float64_vector(
                    [0.14, 0.16, 0.19, 0.21, 0.30],
                    label="traffic_flow current_distribution",
                ),
                target_distribution=_ensure_float64_vector(
                    [0.12, 0.18, 0.20, 0.24, 0.26],
                    label="traffic_flow target_distribution",
                ),
            ),
            objective_labels=(
                "throughput_stability",
                "congestion_reduction",
                "queue_damping",
            ),
            control_gradient=(("alpha", 0.026), ("beta", -0.009), ("omega", 0.014)),
            max_step=0.06,
            knob_hints=("signal_hold", "phase_sync"),
        ),
    )


def build_information_geometry_control_scenarios() -> tuple[dict[str, object], ...]:
    """Build deterministic information-geometry control scenarios."""
    records: list[dict[str, object]] = []
    for scenario in _build_static_scenarios():
        _validate_information_geometry_scenario(scenario)
        record = scenario.to_audit_record()
        _validate_scenario_record(record)
        records.append(record)
    return tuple(records)


__all__ = [
    "FloatArray",
    "InformationGeometryBoundary",
    "InformationGeometryScenario",
    "DistributionPair",
    "build_information_geometry_control_scenarios",
    "_validate_distribution_pair",
    "_validate_control_gradient",
    "_validate_information_geometry_scenario",
    "_validate_scenario_record",
]
