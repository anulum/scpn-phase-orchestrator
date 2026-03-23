# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Predictive (MPC) supervisor

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.regimes import Regime
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import UPDEState
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.reduction import OttAntonsenReduction

__all__ = ["PredictiveSupervisor", "Prediction"]

_R_CRITICAL = 0.3
_R_DEGRADED = 0.6
_K_BOOST = 0.1


@dataclass
class Prediction:
    R_predicted: list[float]
    will_degrade: bool
    will_critical: bool
    steps_to_degradation: int


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
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        alpha: NDArray,
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
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        alpha: NDArray,
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
