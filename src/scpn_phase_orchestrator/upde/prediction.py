# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Forward prediction for UPDE
#
# Adds a prediction-error term to the Kuramoto dynamics. Inspired by
# Friston's predictive coding (2010) and Husserl's protention (1893),
# but implemented as a concrete numerical mechanism — NOT a claim to
# formalize phenomenological time-consciousness.
#
# The prediction error ε_i = θ_i(t) - θ̂_i(t) drives learning of the
# forward model, providing an anticipatory signal absent from the
# standard UPDE which has retention (L9 memory) but no forward prediction.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

__all__ = ["PredictionModel", "PredictionState"]


@dataclass
class PredictionState:
    predicted_phases: NDArray
    prediction_error: NDArray
    mean_error: float
    weights: NDArray


class PredictionModel:
    """Linear forward model for phase prediction.

    Predicts θ̂(t+dt) from θ(t) using learned weights W:
      θ̂(t+dt) = θ(t) + dt · (ω + W · sin(Δθ))

    Prediction error ε = θ_actual - θ̂ (wrapped to [-π, π]).
    Weights updated via gradient descent on ε²:
      W += η · ε ⊗ sin(Δθ)

    The prediction error signal can be injected into the UPDE as an
    additional coupling term, implementing a form of predictive coding
    where the system minimizes its own prediction error.
    """

    def __init__(
        self,
        n_oscillators: int,
        learning_rate: float = 0.01,
        error_gain: float = 0.1,
    ):
        self._n = n_oscillators
        self._lr = learning_rate
        self._error_gain = error_gain
        self._W = np.zeros((n_oscillators, n_oscillators), dtype=np.float64)
        self._prev_phases: NDArray | None = None
        self._prev_predicted: NDArray | None = None

    @property
    def weights(self) -> NDArray:
        return self._W.copy()

    @property
    def error_gain(self) -> float:
        return self._error_gain

    def predict(self, phases: NDArray, omegas: NDArray, dt: float) -> NDArray:
        """Predict phases at next timestep."""
        diff = phases[np.newaxis, :] - phases[:, np.newaxis]
        coupling_pred = np.sum(self._W * np.sin(diff), axis=1)
        predicted: NDArray = (phases + dt * (omegas + coupling_pred)) % TWO_PI
        return predicted

    def update(self, phases: NDArray, omegas: NDArray, dt: float) -> PredictionState:
        """Compute prediction error and update weights.

        Call once per timestep AFTER the solver step.
        """
        if self._prev_phases is None or self._prev_predicted is None:
            # First call — no prediction to compare
            predicted = self.predict(phases, omegas, dt)
            self._prev_phases = phases.copy()
            self._prev_predicted = predicted
            return PredictionState(
                predicted_phases=predicted,
                prediction_error=np.zeros(self._n),
                mean_error=0.0,
                weights=self._W.copy(),
            )

        # Prediction error: actual - predicted (wrapped to [-π, π])
        error = phases - self._prev_predicted
        error = (error + np.pi) % TWO_PI - np.pi

        # Weight update: gradient descent on Σ ε_i²
        diff = self._prev_phases[np.newaxis, :] - self._prev_phases[:, np.newaxis]
        sin_diff = np.sin(diff)
        self._W += self._lr * np.outer(error, np.ones(self._n)) * sin_diff

        # Predict next step
        predicted = self.predict(phases, omegas, dt)

        self._prev_phases = phases.copy()
        self._prev_predicted = predicted

        return PredictionState(
            predicted_phases=predicted,
            prediction_error=error,
            mean_error=float(np.mean(np.abs(error))),
            weights=self._W.copy(),
        )

    def error_coupling(self, phases: NDArray, omegas: NDArray, dt: float) -> NDArray:
        """Prediction-error signal for injection into UPDE.

        Returns ε_gain · ε_i, where ε_i = θ_actual - θ̂_predicted.
        Add this to the UPDE derivative to implement predictive coding:
          dθ/dt = ω + K·sin(Δθ) + gain·ε
        """
        if self._prev_predicted is None:
            return np.zeros(self._n)
        error = phases - self._prev_predicted
        error = (error + np.pi) % TWO_PI - np.pi
        return self._error_gain * error

    def reset(self) -> None:
        self._W[:] = 0.0
        self._prev_phases = None
        self._prev_predicted = None
