# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Predictive (MPC) supervisor

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.upde.metrics import UPDEState
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.prediction import VariationalPredictor
from scpn_phase_orchestrator.upde.reduction import OttAntonsenReduction

__all__ = [
    "FEPPredictionAssessment",
    "FEPPredictiveSupervisor",
    "PredictiveSupervisor",
    "Prediction",
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
        """Return True when observed coherence exceeds the target."""
        return self.observed_R > self.target_R

    def to_audit_record(self) -> dict[str, float]:
        """Return a serialisable audit payload."""
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
        self._n = n_oscillators
        self._dt = dt
        self._horizon = horizon
        self._divergence_threshold = divergence_threshold

    def predict(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        alpha: FloatArray,
    ) -> Prediction:
        """Predict R trajectory using OA reduction as fast forward model."""
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
        """Predictive control: act before degradation, not after."""
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
        if n_oscillators < 1:
            raise ValueError("n_oscillators must be >= 1")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        _require_unit_interval(target_R, "target_R")
        _require_non_negative(free_energy_threshold, "free_energy_threshold")
        _require_non_negative(error_threshold, "error_threshold")
        _require_non_negative(drive_gain, "drive_gain")
        _require_non_negative(learning_rate, "learning_rate")
        _require_non_negative(prior_precision, "prior_precision")

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
        """Target order parameter used by the free-energy controller."""
        return self._target_R

    @property
    def last_assessment(self) -> FEPPredictionAssessment | None:
        """Most recent free-energy assessment, if ``assess`` has run."""
        return self._last_assessment

    def assess(self, phases: FloatArray, omegas: FloatArray) -> FEPPredictionAssessment:
        """Update the variational predictor and return audit-ready metrics."""
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
        """Return FEP-MPC control actions for the current observation."""
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
        if assessment.free_energy >= self._free_energy_threshold:
            return True
        if assessment.mean_abs_error >= self._error_threshold:
            return True
        return upde_state.stability_proxy < self._target_R and assessment.surprise > 0.0


def _validate_phase_inputs(
    phases: FloatArray,
    omegas: FloatArray,
    n_oscillators: int,
) -> tuple[FloatArray, FloatArray]:
    phase_arr = np.asarray(phases, dtype=np.float64)
    omega_arr = np.asarray(omegas, dtype=np.float64)
    if phase_arr.shape != (n_oscillators,):
        raise ValueError(f"phases must have shape ({n_oscillators},)")
    if omega_arr.shape != (n_oscillators,):
        raise ValueError(f"omegas must have shape ({n_oscillators},)")
    if not np.all(np.isfinite(phase_arr)):
        raise ValueError("phases must be finite")
    if not np.all(np.isfinite(omega_arr)):
        raise ValueError("omegas must be finite")
    return phase_arr, omega_arr


def _require_unit_interval(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _require_non_negative(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
