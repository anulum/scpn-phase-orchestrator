# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Causal counterfactual supervisor rollouts

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

__all__ = [
    "CausalAttribution",
    "CausalInterventionEngine",
    "CounterfactualRollout",
    "InterventionParameters",
]

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class CausalAttribution:
    """Attribution summary derived from a paired counterfactual rollout."""

    effect: str
    confidence: float
    score: float
    delta_R_final: float
    delta_R_mean: float
    threshold: float

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable attribution payload."""
        return {
            "effect": self.effect,
            "confidence": self.confidence,
            "score": self.score,
            "delta_R_final": self.delta_R_final,
            "delta_R_mean": self.delta_R_mean,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class InterventionParameters:
    """UPDE parameters after applying a supervisor intervention."""

    knm: FloatArray
    alpha: FloatArray
    zeta: float
    psi: float


@dataclass(frozen=True)
class CounterfactualRollout:
    """Paired baseline/intervention rollout summary for audit logging."""

    baseline_R: list[float]
    intervention_R: list[float]
    baseline_psi: list[float]
    intervention_psi: list[float]
    delta_R_final: float
    delta_R_mean: float
    delta_psi_final: float
    actions: tuple[ControlAction, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable counterfactual audit payload."""
        return {
            "baseline_R": self.baseline_R,
            "intervention_R": self.intervention_R,
            "baseline_psi": self.baseline_psi,
            "intervention_psi": self.intervention_psi,
            "delta_R_final": self.delta_R_final,
            "delta_R_mean": self.delta_R_mean,
            "delta_psi_final": self.delta_psi_final,
            "actions": [
                {
                    "knob": action.knob,
                    "scope": action.scope,
                    "value": action.value,
                    "ttl_s": action.ttl_s,
                    "justification": action.justification,
                }
                for action in self.actions
            ],
        }

    def attribute(self, threshold: float = 1e-3) -> CausalAttribution:
        """Summarise whether the intervention caused a measurable R change."""
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("threshold must be finite and non-negative")
        score = 0.5 * (self.delta_R_final + self.delta_R_mean)
        magnitude = abs(score)
        if magnitude <= threshold:
            effect = "neutral"
        elif score > 0.0:
            effect = "stabilising"
        else:
            effect = "destabilising"
        confidence = 0.0 if threshold == 0.0 else min(1.0, magnitude / threshold)
        return CausalAttribution(
            effect=effect,
            confidence=confidence,
            score=float(score),
            delta_R_final=self.delta_R_final,
            delta_R_mean=self.delta_R_mean,
            threshold=threshold,
        )


class CausalInterventionEngine:
    """Counterfactual UPDE rollouts for supervisor actions.

    The engine answers the first causal supervision question: from the same
    state, what would the order-parameter trajectory look like with and
    without the proposed intervention?
    """

    def __init__(
        self,
        n_oscillators: int,
        dt: float,
        horizon: int = 20,
        method: str = "rk4",
    ):
        if isinstance(n_oscillators, bool) or int(n_oscillators) != n_oscillators:
            raise ValueError("n_oscillators must be an integer")
        if int(n_oscillators) < 1:
            raise ValueError("n_oscillators must be >= 1")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and > 0")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self._n = int(n_oscillators)
        self._dt = float(dt)
        self._horizon = int(horizon)
        self._method = method

    def evaluate_actions(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        alpha: FloatArray,
        zeta: float,
        psi: float,
        actions: list[ControlAction] | tuple[ControlAction, ...],
    ) -> CounterfactualRollout:
        """Compare no-action and intervened trajectories."""
        self._validate_inputs(phases, omegas, knm, alpha, zeta, psi)
        action_tuple = tuple(actions)
        intervened = self.apply_actions(knm, alpha, zeta, psi, action_tuple)

        baseline_R, baseline_psi = self._rollout(phases, omegas, knm, alpha, zeta, psi)
        intervention_R, intervention_psi = self._rollout(
            phases,
            omegas,
            intervened.knm,
            intervened.alpha,
            intervened.zeta,
            intervened.psi,
        )

        baseline_arr = np.asarray(baseline_R, dtype=np.float64)
        intervention_arr = np.asarray(intervention_R, dtype=np.float64)
        return CounterfactualRollout(
            baseline_R=baseline_R,
            intervention_R=intervention_R,
            baseline_psi=baseline_psi,
            intervention_psi=intervention_psi,
            delta_R_final=float(intervention_arr[-1] - baseline_arr[-1]),
            delta_R_mean=float(np.mean(intervention_arr - baseline_arr)),
            delta_psi_final=_signed_phase_delta(
                intervention_psi[-1],
                baseline_psi[-1],
            ),
            actions=action_tuple,
        )

    def apply_actions(
        self,
        knm: FloatArray,
        alpha: FloatArray,
        zeta: float,
        psi: float,
        actions: tuple[ControlAction, ...],
    ) -> InterventionParameters:
        """Apply supported supervisor actions to UPDE parameters."""
        next_knm = np.array(knm, dtype=np.float64, copy=True)
        next_alpha = np.array(alpha, dtype=np.float64, copy=True)
        next_zeta = float(zeta)
        next_psi = float(psi)

        for action in actions:
            if action.knob == "K":
                _apply_matrix_delta(next_knm, action.scope, action.value)
            elif action.knob == "alpha":
                _apply_matrix_delta(next_alpha, action.scope, action.value)
            elif action.knob == "zeta":
                next_zeta += float(action.value)
            elif action.knob in {"Psi", "psi"}:
                next_psi = (next_psi + float(action.value)) % TWO_PI
            else:
                msg = f"unsupported causal intervention knob {action.knob!r}"
                raise ValueError(msg)

        np.fill_diagonal(next_knm, 0.0)
        np.fill_diagonal(next_alpha, 0.0)
        return InterventionParameters(
            knm=next_knm,
            alpha=next_alpha,
            zeta=next_zeta,
            psi=next_psi,
        )

    def _rollout(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        alpha: FloatArray,
        zeta: float,
        psi: float,
    ) -> tuple[list[float], list[float]]:
        engine = UPDEEngine(self._n, self._dt, method=self._method)
        theta = np.array(phases, dtype=np.float64, copy=True)
        r0, psi0 = compute_order_parameter(theta)
        r_values = [r0]
        psi_values = [psi0]
        for _ in range(self._horizon):
            theta = engine.step(theta, omegas, knm, zeta, psi, alpha)
            r_value, psi_value = compute_order_parameter(theta)
            r_values.append(r_value)
            psi_values.append(psi_value)
        return r_values, psi_values

    def _validate_inputs(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        alpha: FloatArray,
        zeta: float,
        psi: float,
    ) -> None:
        n = self._n
        checks = (
            ("phases", phases, (n,)),
            ("omegas", omegas, (n,)),
            ("knm", knm, (n, n)),
            ("alpha", alpha, (n, n)),
        )
        for name, arr, expected in checks:
            if arr.shape != expected:
                raise ValueError(f"{name}.shape={arr.shape}, expected {expected}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains NaN/Inf")
        if not (np.isfinite(zeta) and np.isfinite(psi)):
            raise ValueError("zeta and psi must be finite")


def _apply_matrix_delta(matrix: FloatArray, scope: str, value: float) -> None:
    if scope == "global":
        matrix += float(value)
        return
    if scope.startswith("oscillator_"):
        idx = int(scope.removeprefix("oscillator_"))
        matrix[idx, :] += float(value)
        matrix[:, idx] += float(value)
        return
    if scope.startswith("layer_"):
        raise ValueError("layer-scoped causal interventions require layer membership")
    raise ValueError(f"unsupported causal intervention scope {scope!r}")


def _signed_phase_delta(a: float, b: float) -> float:
    return float((a - b + np.pi) % TWO_PI - np.pi)
