# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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

"""Forward and variational prediction models for validated UPDE phase states.

The module supplies a linear prediction-error model and a variational
free-energy predictor over one-dimensional oscillator phase vectors. Public
constructors and update methods validate oscillator counts, positive time
steps, finite phase/frequency arrays, and precision vectors before mutating
internal weights, sufficient statistics, or error histories. The implementation
is a concrete numerical mechanism and does not claim to formalize
phenomenological time-consciousness.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

__all__ = [
    "PredictionModel",
    "PredictionState",
    "VariationalPredictor",
    "VariationalState",
]

FloatArray: TypeAlias = NDArray[np.float64]


def _validate_positive_int(name: str, value: int) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return int(value)


def _validate_nonnegative_float(name: str, value: float) -> float:
    """Return ``value`` as a non-negative finite float, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative real")
    out = float(value)
    if not isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real")
    return out


def _validate_positive_float(name: str, value: float) -> float:
    """Return ``value`` as a strictly positive finite float, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real")
    out = float(value)
    if not isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a finite positive real")
    return out


def _validate_vector(name: str, value: FloatArray, n_oscillators: int) -> FloatArray:
    """Return the value as a validated 1-D finite array, else raise."""
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must not contain boolean values")
    array = raw.astype(np.float64, copy=True)
    if array.shape != (n_oscillators,):
        raise ValueError(f"{name} must have shape ({n_oscillators},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_positive_vector(
    name: str, value: FloatArray, n_oscillators: int
) -> FloatArray:
    """Return the value as a validated 1-D strictly positive array, else raise."""
    array = _validate_vector(name, value, n_oscillators)
    if not np.all(array > 0.0):
        raise ValueError(f"{name} must contain only positive values")
    return array


@dataclass
class PredictionState:
    """Snapshot of the forward prediction model after one update step."""

    predicted_phases: FloatArray
    prediction_error: FloatArray
    mean_error: float
    weights: FloatArray


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
        self._n = _validate_positive_int("n_oscillators", n_oscillators)
        self._lr = _validate_nonnegative_float("learning_rate", learning_rate)
        self._error_gain = _validate_nonnegative_float("error_gain", error_gain)
        self._W = np.zeros((self._n, self._n), dtype=np.float64)
        self._prev_phases: FloatArray | None = None
        self._prev_predicted: FloatArray | None = None

    @property
    def weights(self) -> FloatArray:
        """Copy of the current learned weight matrix W.

        Returns
        -------
        FloatArray
            Copy of the current learned weight matrix W.
        """
        return self._W.copy()

    @property
    def error_gain(self) -> float:
        """Scaling factor applied to prediction error before injection.

        Returns
        -------
        float
            Scaling factor applied to prediction error before injection.
        """
        return self._error_gain

    def predict(self, phases: FloatArray, omegas: FloatArray, dt: float) -> FloatArray:
        """Predict phases at next timestep.

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.
        omegas : FloatArray
            Natural frequencies in rad/s, shape ``(N,)``.
        dt : float
            Integration step size.

        Returns
        -------
        FloatArray
            The predicted phases at the next timestep.
        """
        phases = _validate_vector("phases", phases, self._n)
        omegas = _validate_vector("omegas", omegas, self._n)
        dt = _validate_positive_float("dt", dt)
        diff = phases[np.newaxis, :] - phases[:, np.newaxis]
        coupling_pred = np.sum(self._W * np.sin(diff), axis=1)
        predicted: FloatArray = (phases + dt * (omegas + coupling_pred)) % TWO_PI
        return predicted

    def update(
        self, phases: FloatArray, omegas: FloatArray, dt: float
    ) -> PredictionState:
        """Compute prediction error and update weights.

        Call once per timestep AFTER the solver step.

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.
        omegas : FloatArray
            Natural frequencies in rad/s, shape ``(N,)``.
        dt : float
            Integration step size.

        Returns
        -------
        PredictionState
            The updated prediction state after one learning step.
        """
        phases = _validate_vector("phases", phases, self._n)
        omegas = _validate_vector("omegas", omegas, self._n)
        dt = _validate_positive_float("dt", dt)
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

    def error_coupling(
        self, phases: FloatArray, omegas: FloatArray, dt: float
    ) -> FloatArray:
        """Prediction-error signal for injection into UPDE.

        Returns ε_gain · ε_i, where ε_i = θ_actual - θ̂_predicted.
        Add this to the UPDE derivative to implement predictive coding:
          dθ/dt = ω + K·sin(Δθ) + gain·ε

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.
        omegas : FloatArray
            Natural frequencies in rad/s, shape ``(N,)``.
        dt : float
            Integration step size.

        Returns
        -------
        FloatArray
            The prediction-error coupling signal for UPDE injection.
        """
        phases = _validate_vector("phases", phases, self._n)
        _validate_vector("omegas", omegas, self._n)
        _validate_positive_float("dt", dt)
        if self._prev_predicted is None:
            return np.zeros(self._n)
        error = phases - self._prev_predicted
        error = (error + np.pi) % TWO_PI - np.pi
        out: FloatArray = self._error_gain * error
        return out

    def reset(self) -> None:
        """Zero the weight matrix and clear phase history."""
        self._W[:] = 0.0
        self._prev_phases = None
        self._prev_predicted = None


@dataclass
class VariationalState:
    """Snapshot of the variational predictor after one update step."""

    predicted_phases: FloatArray
    error: FloatArray
    free_energy: float
    precision: FloatArray
    complexity: float


class VariationalPredictor:
    """Variational free energy minimization for phase prediction.

    Implements the formal mapping between SCPN phase dynamics and
    Friston's Free Energy Principle:

    F = E_q[log q(theta) - log p(theta, y)]
      ~ prediction_error^2 / (2 * precision) + complexity

    where:
      theta = phase states (sufficient statistics mu in FEP)
      y = observed phases
      q(theta) = recognition density (Gaussian, parameterized by mu, Sigma)
      prediction_error = y - f(mu) (sensory prediction error)
      precision = 1/sigma^2 (inverse variance, maps to coupling K)
      complexity = KL[q||p] (prior deviation cost)

    The UPDE coupling term K_ij * sin(theta_j - theta_i) maps to
    precision-weighted prediction error under Laplace approximation
    (Friston 2010, Eq. 4).

    This is NOT a claim to formalize Husserl's protention. It is a
    concrete numerical implementation of the mathematical correspondence
    between Kuramoto coupling and variational inference.
    """

    def __init__(
        self,
        n_oscillators: int,
        prior_precision: float = 1.0,
        learning_rate: float = 0.01,
    ):
        self._n = _validate_positive_int("n_oscillators", n_oscillators)
        self._lr = _validate_nonnegative_float("learning_rate", learning_rate)
        self._prior_precision = _validate_positive_float(
            "prior_precision", prior_precision
        )
        # Precision matrix (diagonal): initialized to prior_precision.
        # Under the FEP-Kuramoto correspondence, precision_ij ~ K_ij.
        self._precision = np.full(self._n, self._prior_precision, dtype=np.float64)
        self._mu = np.zeros(self._n, dtype=np.float64)
        self._omegas: FloatArray | None = None
        self._error_history: list[FloatArray] = []
        # Exponential moving average decay for precision updates
        self._ema_alpha = 0.1

    @property
    def precision(self) -> FloatArray:
        """Copy of the current per-oscillator precision vector.

        Returns
        -------
        FloatArray
            Copy of the current per-oscillator precision vector.
        """
        return self._precision.copy()

    def free_energy(
        self,
        predicted: FloatArray,
        observed: FloatArray,
        precision: FloatArray,
    ) -> float:
        """Variational free energy F.

        F = sum_i [ (y_i - f(mu_i))^2 * pi_i / 2 ] + sum_i [ log(pi_i) ]

        First term: precision-weighted prediction error (accuracy).
        Second term: log-precision (complexity under Gaussian q).
        The sign convention follows Friston (2010): F is minimized.

        Parameters
        ----------
        predicted : FloatArray
            Predicted phases in radians, shape ``(N,)``.
        observed : FloatArray
            Observed phases in radians, shape ``(N,)``.
        precision : FloatArray
            Per-oscillator precision vector, shape ``(N,)``.

        Returns
        -------
        float
            The variational free energy ``F``.
        """
        predicted = _validate_vector("predicted", predicted, self._n)
        observed = _validate_vector("observed", observed, self._n)
        precision = _validate_positive_vector("precision", precision, self._n)
        error = observed - predicted
        error = (error + np.pi) % TWO_PI - np.pi
        accuracy = float(np.sum(error**2 * precision / 2.0))
        # KL complexity: log-precision acts as a regularizer pulling
        # precision toward values where q(theta) stays close to prior p(theta).
        complexity = float(np.sum(np.log(np.maximum(precision, 1e-12))))
        return accuracy + complexity

    def update(
        self, phases: FloatArray, omegas: FloatArray, dt: float
    ) -> VariationalState:
        """One variational update step.

        1. Predict phases from current sufficient statistics mu.
        2. Compute precision-weighted prediction error.
        3. Update mu (gradient descent on F).
        4. Update precision from error statistics.

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.
        omegas : FloatArray
            Natural frequencies in rad/s, shape ``(N,)``.
        dt : float
            Integration step size.

        Returns
        -------
        VariationalState
            The updated variational state after one step.
        """
        phases = _validate_vector("phases", phases, self._n)
        omegas = _validate_vector("omegas", omegas, self._n)
        dt = _validate_positive_float("dt", dt)
        self._omegas = omegas

        # Forward model: f(mu) = mu + dt * omega (simplest generative model)
        predicted = (self._mu + dt * omegas) % TWO_PI

        # Prediction error wrapped to [-pi, pi]
        error = phases - predicted
        error = (error + np.pi) % TWO_PI - np.pi

        # Free energy before update
        fe = self.free_energy(predicted, phases, self._precision)

        # Complexity: KL divergence between current and prior precision.
        # For diagonal Gaussian: KL = 0.5 * sum(pi/pi_0 - 1 - log(pi/pi_0))
        ratio = self._precision / self._prior_precision
        complexity = float(0.5 * np.sum(ratio - 1.0 - np.log(np.maximum(ratio, 1e-12))))

        # Gradient descent on F w.r.t. mu:
        # dF/dmu = -precision * error  (since F ~ precision * error^2 / 2)
        # mu_new = mu - lr * dF/dmu = mu + lr * precision * error
        # type ignore: modulo preserves ndarray shape, but mypy narrows to scalar.
        self._mu = (self._mu + self._lr * self._precision * error) % TWO_PI  # type: ignore[assignment]

        # Update precision from error variance (online).
        # Precision = 1/variance. Use EMA of squared error as variance estimate.
        self._error_history.append(error.copy())
        if len(self._error_history) > 1:
            recent = np.array(self._error_history[-min(50, len(self._error_history)) :])
            var_estimate = np.mean(recent**2, axis=0)
            # EMA blend: pi_new = (1-a)*pi_old + a*(1/var)
            new_prec = 1.0 / np.maximum(var_estimate, 1e-8)
            self._precision = (
                1.0 - self._ema_alpha
            ) * self._precision + self._ema_alpha * new_prec

        return VariationalState(
            predicted_phases=predicted,
            error=error,
            free_energy=fe,
            precision=self._precision.copy(),
            complexity=complexity,
        )

    def precision_weighted_coupling(self) -> FloatArray:
        """Precision matrix interpretable as K_ij.

        Under the FEP-Kuramoto correspondence (Friston 2010, Laplace
        approximation), the coupling matrix K_ij maps to the off-diagonal
        elements of the precision matrix of the generative model.

        This returns diag(precision) as the simplest such mapping.
        For a full N x N coupling matrix, use np.diag(result).

        Returns
        -------
        FloatArray
            Precision matrix interpretable as K_ij.
        """
        return np.diag(self._precision)

    def reset(self) -> None:
        """Reset precision to prior, zero sufficient statistics, clear history."""
        self._precision[:] = self._prior_precision
        self._mu[:] = 0.0
        self._omegas = None
        self._error_history.clear()
