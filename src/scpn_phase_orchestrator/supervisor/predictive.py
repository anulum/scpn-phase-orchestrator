# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Predictive (MPC) supervisor

"""Predictive and free-energy supervisor diagnostics for bounded action proposals.

The module provides Ott-Antonsen horizon prediction, variational free-energy
assessment, and hierarchy-level FEP assessments over validated phase/frequency
state. Predictive supervisors emit conservative ``ControlAction`` proposals for
degradation, critical forecasts, hard boundaries, or high surprise. They do not
apply actuation or mutate caller-owned phase/coupling arrays.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator._validation import non_negative_real
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.prediction import VariationalPredictor
from scpn_phase_orchestrator.upde.reduction import OttAntonsenReduction

__all__ = [
    "FEPHierarchyAssessment",
    "FEPHierarchyChildAssessment",
    "FEPPredictionAssessment",
    "FEPPredictiveSupervisor",
    "PredictiveSupervisor",
    "Prediction",
    "assess_fep_hierarchy",
]

FloatArray: TypeAlias = NDArray[np.float64]

_R_CRITICAL = 0.3
_R_DEGRADED = 0.6
_K_BOOST = 0.1


@dataclass
class Prediction:
    """Forward model output: predicted R trajectory and degradation flags."""

    R_predicted: list[float]
    will_degrade: bool
    will_critical: bool
    steps_to_degradation: int


@dataclass(frozen=True)
class FEPPredictionAssessment:
    """One-step variational free-energy assessment for supervisor control."""

    free_energy: float
    complexity: float
    mean_abs_error: float
    precision_mean: float
    precision_spread: float
    observed_R: float
    observed_psi: float
    predicted_R: float
    target_R: float
    surprise: float

    @property
    def above_target(self) -> bool:
        """Return True when observed coherence exceeds the target.

        Returns
        -------
        bool
            Return True when observed coherence exceeds the target.
        """
        return self.observed_R > self.target_R

    def to_audit_record(self) -> dict[str, float]:
        """Return a serialisable audit payload.

        Returns
        -------
        dict[str, float]
            Return a serialisable audit payload.
        """
        return {
            "free_energy": self.free_energy,
            "complexity": self.complexity,
            "mean_abs_error": self.mean_abs_error,
            "precision_mean": self.precision_mean,
            "precision_spread": self.precision_spread,
            "observed_R": self.observed_R,
            "observed_psi": self.observed_psi,
            "predicted_R": self.predicted_R,
            "target_R": self.target_R,
            "surprise": self.surprise,
        }


@dataclass(frozen=True)
class FEPHierarchyChildAssessment:
    """Assessment for one child node in a hierarchical FEP supervisor."""

    name: str
    assessment: FEPPredictionAssessment
    actions: tuple[ControlAction, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe child hierarchy audit record.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe child hierarchy audit record.
        """
        return {
            "name": self.name,
            "assessment": self.assessment.to_audit_record(),
            "actions": [_action_record(action) for action in self.actions],
        }


@dataclass(frozen=True)
class FEPHierarchyAssessment:
    """Audit-ready child-to-parent FEP hierarchy assessment."""

    hierarchy: str
    children: tuple[FEPHierarchyChildAssessment, ...]
    parent_assessment: FEPPredictionAssessment
    parent_actions: tuple[ControlAction, ...]
    child_R_values: tuple[float, ...]
    parent_phase_encoding: tuple[float, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe hierarchy assessment payload.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe hierarchy assessment payload.
        """
        return {
            "hierarchy": self.hierarchy,
            "children": [child.to_audit_record() for child in self.children],
            "parent": {
                "assessment": self.parent_assessment.to_audit_record(),
                "actions": [_action_record(action) for action in self.parent_actions],
            },
            "child_R_values": list(self.child_R_values),
            "parent_phase_encoding": list(self.parent_phase_encoding),
        }


class PredictiveSupervisor:
    """Model-predictive supervisor using Ott-Antonsen forward model.

    Predicts R trajectory `horizon` steps ahead. Acts preemptively when
    predicted R crosses thresholds, instead of waiting for actual degradation.
    Falls back to reactive supervision if OA prediction diverges.
    """

    def __init__(
        self,
        n_oscillators: int,
        dt: float,
        horizon: int = 10,
        divergence_threshold: float = 0.3,
    ):
        self._n = _require_positive_int(n_oscillators, "n_oscillators")
        self._dt = _require_positive_real(dt, "dt")
        self._horizon = _require_positive_int(horizon, "horizon")
        self._divergence_threshold = non_negative_real(
            divergence_threshold,
            name="divergence_threshold",
        )

    def predict(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        alpha: FloatArray,
    ) -> Prediction:
        """Predict R trajectory using OA reduction as fast forward model.

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.
        omegas : FloatArray
            Natural frequencies in rad/s, shape ``(N,)``.
        knm : FloatArray
            Coupling matrix ``K_nm``, shape ``(N, N)``.
        alpha : FloatArray
            Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.

        Returns
        -------
        Prediction
            The predicted ``R`` trajectory.
        """
        phases, omegas, knm, alpha = _validate_predictive_inputs(
            phases,
            omegas,
            knm,
            alpha,
            self._n,
        )
        R_current, psi = compute_order_parameter(phases)

        # Fit Lorentzian to omegas for OA
        omega_0 = float(np.median(omegas))
        q75, q25 = np.percentile(omegas, [75, 25])
        delta = max((q75 - q25) / 2.0, 0.01)
        K_eff = float(np.mean(knm[knm > 0])) if np.any(knm > 0) else 0.0

        oa = OttAntonsenReduction(omega_0, delta, K_eff, dt=self._dt)
        z0 = complex(R_current * np.cos(psi), R_current * np.sin(psi))

        trajectory = [R_current]
        z = z0
        for _ in range(self._horizon):
            z = oa.step(z)
            trajectory.append(abs(z))

        # Check for divergence (OA prediction unreliable)
        if abs(trajectory[-1] - R_current) > self._divergence_threshold:
            trajectory = [R_current] * (self._horizon + 1)

        will_degrade = any(r < _R_DEGRADED for r in trajectory)
        will_critical = any(r < _R_CRITICAL for r in trajectory)
        steps_to_deg = self._horizon
        for i, r in enumerate(trajectory):
            if r < _R_DEGRADED:
                steps_to_deg = i
                break

        return Prediction(
            R_predicted=trajectory,
            will_degrade=will_degrade,
            will_critical=will_critical,
            steps_to_degradation=steps_to_deg,
        )

    def decide(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        alpha: FloatArray,
        upde_state: UPDEState,
        boundary_state: BoundaryState,
    ) -> list[ControlAction]:
        """Predictive control: act before degradation, not after.

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.
        omegas : FloatArray
            Natural frequencies in rad/s, shape ``(N,)``.
        knm : FloatArray
            Coupling matrix ``K_nm``, shape ``(N, N)``.
        alpha : FloatArray
            Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
        upde_state : UPDEState
            The current UPDE state.
        boundary_state : BoundaryState
            The current boundary-observer state.

        Returns
        -------
        list[ControlAction]
            The predictive control actions for the current state.

        Raises
        ------
        ValueError
            If the state inputs are invalid.
        """
        if not isinstance(upde_state, UPDEState):
            raise ValueError(f"upde_state must be a UPDEState, got {upde_state!r}")
        if not isinstance(boundary_state, BoundaryState):
            raise ValueError(
                f"boundary_state must be a BoundaryState, got {boundary_state!r}"
            )
        if boundary_state.hard_violations:
            return [
                ControlAction(
                    knob="zeta",
                    scope="global",
                    value=0.1,
                    ttl_s=5.0,
                    justification="hard boundary violation",
                )
            ]

        pred = self.predict(phases, omegas, knm, alpha)

        if pred.will_critical:
            return [
                ControlAction(
                    knob="K",
                    scope="global",
                    value=_K_BOOST * 2,
                    ttl_s=10.0,
                    justification=(
                        f"MPC: R predicted to hit CRITICAL "
                        f"in {pred.steps_to_degradation} steps"
                    ),
                )
            ]

        if pred.will_degrade and pred.steps_to_degradation < self._horizon // 2:
            return [
                ControlAction(
                    knob="K",
                    scope="global",
                    value=_K_BOOST,
                    ttl_s=10.0,
                    justification=(
                        f"MPC: R predicted to degrade "
                        f"in {pred.steps_to_degradation} steps"
                    ),
                )
            ]

        return []


class FEPPredictiveSupervisor:
    """Free-energy predictive supervisor built on ``VariationalPredictor``.

    The class turns the existing FEP-Kuramoto variational predictor into a
    bounded supervisor mode. It does not claim a complete biological FEP
    model; it exposes an auditable one-step free-energy signal and maps high
    surprise into conservative ``zeta`` / ``Psi`` control actions.
    """

    def __init__(
        self,
        n_oscillators: int,
        dt: float,
        target_R: float = 0.8,
        free_energy_threshold: float = 1.0,
        error_threshold: float = 0.25,
        drive_gain: float = 0.1,
        learning_rate: float = 0.01,
        prior_precision: float = 1.0,
    ) -> None:
        n_oscillators = _require_positive_int(n_oscillators, "n_oscillators")
        dt = _require_positive_real(dt, "dt")
        _require_unit_interval(target_R, "target_R")
        non_negative_real(free_energy_threshold, name="free_energy_threshold")
        non_negative_real(error_threshold, name="error_threshold")
        non_negative_real(drive_gain, name="drive_gain")
        non_negative_real(learning_rate, name="learning_rate")
        non_negative_real(prior_precision, name="prior_precision")

        self._n = n_oscillators
        self._dt = dt
        self._target_R = target_R
        self._free_energy_threshold = free_energy_threshold
        self._error_threshold = error_threshold
        self._drive_gain = drive_gain
        self._predictor = VariationalPredictor(
            n_oscillators,
            prior_precision=prior_precision,
            learning_rate=learning_rate,
        )
        self._last_assessment: FEPPredictionAssessment | None = None

    @property
    def target_R(self) -> float:
        """Target order parameter used by the free-energy controller.

        Returns
        -------
        float
            Target order parameter used by the free-energy controller.
        """
        return self._target_R

    @property
    def last_assessment(self) -> FEPPredictionAssessment | None:
        """Most recent free-energy assessment, if ``assess`` has run.

        Returns
        -------
        FEPPredictionAssessment | None
            Most recent free-energy assessment, if ``assess`` has run.
        """
        return self._last_assessment

    def assess(self, phases: FloatArray, omegas: FloatArray) -> FEPPredictionAssessment:
        """Update the variational predictor and return audit-ready metrics.

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.
        omegas : FloatArray
            Natural frequencies in rad/s, shape ``(N,)``.

        Returns
        -------
        FEPPredictionAssessment
            The free-energy prediction assessment.
        """
        phases_arr, omegas_arr = _validate_phase_inputs(phases, omegas, self._n)
        variational = self._predictor.update(phases_arr, omegas_arr, self._dt)
        observed_R, observed_psi = compute_order_parameter(phases_arr)
        predicted_R, _ = compute_order_parameter(variational.predicted_phases)
        mean_abs_error = float(np.mean(np.abs(variational.error)))
        precision_mean = float(np.mean(variational.precision))
        precision_spread = float(
            np.max(variational.precision) - np.min(variational.precision)
        )
        surprise = float(abs(observed_R - self._target_R) + mean_abs_error)
        assessment = FEPPredictionAssessment(
            free_energy=float(variational.free_energy),
            complexity=float(variational.complexity),
            mean_abs_error=mean_abs_error,
            precision_mean=precision_mean,
            precision_spread=precision_spread,
            observed_R=observed_R,
            observed_psi=observed_psi,
            predicted_R=predicted_R,
            target_R=self._target_R,
            surprise=surprise,
        )
        self._last_assessment = assessment
        return assessment

    def decide(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        upde_state: UPDEState,
        boundary_state: BoundaryState,
    ) -> list[ControlAction]:
        """Return FEP-MPC control actions for the current observation.

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.
        omegas : FloatArray
            Natural frequencies in rad/s, shape ``(N,)``.
        upde_state : UPDEState
            The current UPDE state.
        boundary_state : BoundaryState
            The current boundary-observer state.

        Returns
        -------
        list[ControlAction]
            The FEP-MPC control actions for the current observation.

        Raises
        ------
        ValueError
            If the state inputs are invalid.
        """
        if not isinstance(upde_state, UPDEState):
            raise ValueError(f"upde_state must be a UPDEState, got {upde_state!r}")
        if not isinstance(boundary_state, BoundaryState):
            raise ValueError(
                f"boundary_state must be a BoundaryState, got {boundary_state!r}"
            )
        if boundary_state.hard_violations:
            return [
                ControlAction(
                    knob="zeta",
                    scope="global",
                    value=self._drive_gain,
                    ttl_s=5.0,
                    justification="FEP-MPC: hard boundary violation",
                )
            ]

        assessment = self.assess(phases, omegas)
        if not self._should_act(assessment, upde_state):
            return []

        psi_target = assessment.observed_psi
        if assessment.above_target:
            psi_target = (psi_target + np.pi) % TWO_PI

        return [
            ControlAction(
                knob="zeta",
                scope="global",
                value=self._drive_gain,
                ttl_s=5.0,
                justification=(
                    "FEP-MPC: free energy "
                    f"{assessment.free_energy:.4g}, surprise "
                    f"{assessment.surprise:.4g}"
                ),
            ),
            ControlAction(
                knob="Psi",
                scope="global",
                value=float(psi_target),
                ttl_s=5.0,
                justification="FEP-MPC: precision-weighted phase target",
            ),
        ]

    def reset(self) -> None:
        """Reset the underlying variational predictor and cached assessment."""
        self._predictor.reset()
        self._last_assessment = None

    def _should_act(
        self,
        assessment: FEPPredictionAssessment,
        upde_state: UPDEState,
    ) -> bool:
        """Return whether the predictive supervisor should act."""
        if assessment.free_energy >= self._free_energy_threshold:
            return True
        if assessment.mean_abs_error >= self._error_threshold:
            return True
        return upde_state.stability_proxy < self._target_R and assessment.surprise > 0.0


def assess_fep_hierarchy(
    children: Mapping[str, tuple[FloatArray, FloatArray]],
    *,
    dt: float,
    child_target_R: float = 0.8,
    parent_target_R: float = 0.8,
    parent_dt: float | None = None,
    free_energy_threshold: float = 0.0,
    child_drive_gain: float = 0.08,
    parent_drive_gain: float = 0.05,
    hierarchy: str = "child_regions_to_parent_fep_supervisor",
) -> FEPHierarchyAssessment:
    """Assess child FEP supervisors and a parent over reduced child coherence.

    Each child receives its own ``FEPPredictiveSupervisor``. The parent encodes
    child coherence as phases via ``arccos(2R - 1)`` so the same FEP machinery
    can reason over cross-child coherence without accessing raw child signals.

    Parameters
    ----------
    children : Mapping[str, tuple[FloatArray, FloatArray]]
        Child supervisor summaries.
    dt : float
        Integration step size.
    child_target_R : float
        Target order parameter for each child.
    parent_target_R : float
        Target order parameter for the parent.
    parent_dt : float | None
        Parent integration step size, or ``None``.
    free_energy_threshold : float
        Free-energy threshold above which control acts.
    child_drive_gain : float
        Drive gain applied at the child level.
    parent_drive_gain : float
        Drive gain applied at the parent level.
    hierarchy : str
        Hierarchy label.

    Returns
    -------
    FEPHierarchyAssessment
        The hierarchical free-energy assessment.
    """
    _validate_hierarchy_inputs(
        children=children,
        dt=dt,
        parent_dt=parent_dt,
        child_target_R=child_target_R,
        parent_target_R=parent_target_R,
        free_energy_threshold=free_energy_threshold,
        child_drive_gain=child_drive_gain,
        parent_drive_gain=parent_drive_gain,
    )
    child_records: list[FEPHierarchyChildAssessment] = []
    child_rs: list[float] = []
    for name, (phases, omegas) in children.items():
        phases_arr, omegas_arr = _validate_child_observation(name, phases, omegas)
        supervisor = FEPPredictiveSupervisor(
            n_oscillators=phases_arr.size,
            dt=dt,
            target_R=child_target_R,
            free_energy_threshold=free_energy_threshold,
            drive_gain=child_drive_gain,
        )
        assessment = supervisor.assess(phases_arr, omegas_arr)
        actions = tuple(
            supervisor.decide(
                phases_arr,
                omegas_arr,
                _state_from_r(assessment.observed_R),
                BoundaryState(),
            )
        )
        child_records.append(
            FEPHierarchyChildAssessment(
                name=name,
                assessment=assessment,
                actions=actions,
            )
        )
        child_rs.append(assessment.observed_R)

    child_r_arr = np.asarray(child_rs, dtype=np.float64)
    parent_phases = _coherence_to_parent_phases(child_r_arr)
    parent_omegas = np.full(parent_phases.shape, 1.0, dtype=np.float64)
    parent = FEPPredictiveSupervisor(
        n_oscillators=parent_phases.size,
        dt=dt if parent_dt is None else parent_dt,
        target_R=parent_target_R,
        free_energy_threshold=free_energy_threshold,
        drive_gain=parent_drive_gain,
    )
    parent_assessment = parent.assess(parent_phases, parent_omegas)
    parent_actions = tuple(
        parent.decide(
            parent_phases,
            parent_omegas,
            _state_from_r(parent_assessment.observed_R),
            BoundaryState(),
        )
    )
    return FEPHierarchyAssessment(
        hierarchy=hierarchy,
        children=tuple(child_records),
        parent_assessment=parent_assessment,
        parent_actions=parent_actions,
        child_R_values=tuple(float(value) for value in child_r_arr),
        parent_phase_encoding=tuple(float(value) for value in parent_phases),
    )


def _coerce_real_array(name: str, value: object) -> FloatArray:
    """Return ``value`` as a validated finite real array, else raise."""
    raw = np.asarray(value, dtype=object)
    if any(isinstance(item, bool | np.bool_) for item in raw.ravel()):
        raise ValueError(f"{name} must not contain boolean values")
    if any(isinstance(item, complex | np.complexfloating) for item in raw.ravel()):
        raise ValueError(f"{name} must contain real-valued samples")
    try:
        return np.ascontiguousarray(raw.astype(np.float64), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _validate_phase_inputs(
    phases: object,
    omegas: object,
    n_oscillators: int,
) -> tuple[FloatArray, FloatArray]:
    """Validate the phase inputs, else raise."""
    phase_arr = _coerce_real_array("phases", phases)
    omega_arr = _coerce_real_array("omegas", omegas)
    if phase_arr.shape != (n_oscillators,):
        raise ValueError(f"phases must have shape ({n_oscillators},)")
    if omega_arr.shape != (n_oscillators,):
        raise ValueError(f"omegas must have shape ({n_oscillators},)")
    if not np.all(np.isfinite(phase_arr)):
        raise ValueError("phases must be finite")
    if not np.all(np.isfinite(omega_arr)):
        raise ValueError("omegas must be finite")
    return phase_arr, omega_arr


def _validate_predictive_inputs(
    phases: object,
    omegas: object,
    knm: object,
    alpha: object,
    n_oscillators: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Validate and normalise the predictive inputs, else raise."""
    phase_arr, omega_arr = _validate_phase_inputs(phases, omegas, n_oscillators)
    knm_arr = _coerce_real_array("knm", knm)
    alpha_arr = _coerce_real_array("alpha", alpha)
    expected_shape = (n_oscillators, n_oscillators)
    if knm_arr.shape != expected_shape:
        raise ValueError(f"knm must have shape {expected_shape}")
    if alpha_arr.shape != expected_shape:
        raise ValueError(f"alpha must have shape {expected_shape}")
    if not np.all(np.isfinite(knm_arr)):
        raise ValueError("knm must be finite")
    if not np.all(np.isfinite(alpha_arr)):
        raise ValueError("alpha must be finite")
    return phase_arr, omega_arr, knm_arr, alpha_arr


def _require_positive_int(value: object, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _require_positive_real(value: object, name: str) -> float:
    """Return ``value`` as a strictly positive finite real, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    float_value = float(value)
    if not np.isfinite(float_value) or float_value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float_value


def _require_unit_interval(value: float, name: str) -> None:
    """Return ``value`` as a float in [0, 1], else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _validate_hierarchy_inputs(
    *,
    children: Mapping[str, tuple[FloatArray, FloatArray]],
    dt: float,
    parent_dt: float | None,
    child_target_R: float,
    parent_target_R: float,
    free_energy_threshold: float,
    child_drive_gain: float,
    parent_drive_gain: float,
) -> None:
    """Validate the hierarchy inputs, else raise."""
    if not isinstance(children, Mapping):
        raise ValueError("children must be a mapping of child observations")
    if not children:
        raise ValueError("children must contain at least one child observation")
    _require_positive_real(dt, "dt")
    if parent_dt is not None:
        _require_positive_real(parent_dt, "parent_dt")
    _require_unit_interval(child_target_R, "child_target_R")
    _require_unit_interval(parent_target_R, "parent_target_R")
    non_negative_real(free_energy_threshold, name="free_energy_threshold")
    non_negative_real(child_drive_gain, name="child_drive_gain")
    non_negative_real(parent_drive_gain, name="parent_drive_gain")


def _validate_child_observation(
    name: str,
    phases: object,
    omegas: object,
) -> tuple[FloatArray, FloatArray]:
    """Validate a child observation, else raise."""
    if not isinstance(name, str) or not name:
        raise ValueError("child names must be non-empty strings")
    phase_arr = _coerce_real_array(f"child {name!r} phases", phases)
    omega_arr = _coerce_real_array(f"child {name!r} omegas", omegas)
    if phase_arr.ndim != 1 or phase_arr.size < 1:
        raise ValueError(f"child {name!r} phases must be a non-empty vector")
    if omega_arr.shape != phase_arr.shape:
        raise ValueError(f"child {name!r} omegas must match phases shape")
    if not np.all(np.isfinite(phase_arr)):
        raise ValueError(f"child {name!r} phases must be finite")
    if not np.all(np.isfinite(omega_arr)):
        raise ValueError(f"child {name!r} omegas must be finite")
    return phase_arr, omega_arr


def _coherence_to_parent_phases(child_rs: FloatArray) -> FloatArray:
    """Return the parent phases implied by child coherence."""
    return np.arccos(np.clip(2.0 * child_rs - 1.0, -1.0, 1.0))


def _state_from_r(r_value: float) -> UPDEState:
    """Return the regime state from an order parameter."""
    return UPDEState(
        layers=[LayerState(R=float(r_value), psi=0.0)],
        cross_layer_alignment=np.eye(1),
        stability_proxy=float(r_value),
        regime_id="hierarchical_fep",
    )


def _action_record(action: ControlAction) -> dict[str, object]:
    """Return the JSON-safe record for an action."""
    return {
        "knob": action.knob,
        "scope": action.scope,
        "value": action.value,
        "ttl_s": action.ttl_s,
        "justification": action.justification,
    }
