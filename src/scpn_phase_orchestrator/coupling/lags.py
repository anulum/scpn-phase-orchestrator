# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling lag functions

"""Phase-lag estimation and alpha-matrix construction.

`LagModel` converts physical distances or observed signal offsets into
antisymmetric phase-lag matrices consumed by the UPDE engine. The module is
kept dependency-light and deterministic while rejecting non-physical distance,
sample, carrier, and speed inputs before the resulting lags enter closed-loop
runs.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["LagModel"]

FloatArray: TypeAlias = NDArray[np.float64]


def _contains_boolean_alias(value: object) -> bool:
    raw = np.asarray(value, dtype=object)
    return any(isinstance(item, bool | np.bool_) for item in raw.ravel())


def _contains_complex_alias(value: object) -> bool:
    raw = np.asarray(value, dtype=object)
    return any(isinstance(item, complex | np.complexfloating) for item in raw.ravel())


def _validate_distances(value: object) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError("distances must not contain boolean values")
    if _contains_complex_alias(value):
        raise ValueError("distances must contain real-valued samples")
    try:
        distances = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "distances must be a finite non-negative square matrix"
        ) from exc
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a finite non-negative square matrix")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distances must contain only finite values")
    if np.any(distances < 0.0):
        raise ValueError("distances must be non-negative")
    if not np.allclose(np.diag(distances), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("distances must have a zero diagonal")
    if not np.allclose(distances, distances.T, rtol=1e-12, atol=1e-12):
        raise ValueError("distances must be symmetric physical pair distances")
    return np.ascontiguousarray(distances, dtype=np.float64)


def _validate_positive_real(value: object, *, name: str) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real")
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be a finite positive real")
    return resolved


def _validate_n_layers(value: object) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, Integral):
        raise ValueError("n_layers must be a positive integer")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError("n_layers must be a positive integer")
    return resolved


def _validate_signal(value: object, *, name: str) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if _contains_complex_alias(value):
        raise ValueError(f"{name} must contain real-valued samples")
    try:
        signal = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "signals must be the same finite one-dimensional arrays"
        ) from exc
    if signal.ndim != 1:
        raise ValueError("signals must be the same finite one-dimensional arrays")
    if signal.size == 0:
        raise ValueError("signals must be non-empty")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signals must contain only finite values")
    if float(np.var(signal)) <= 0.0:
        raise ValueError("signals must have non-zero variance")
    return np.ascontiguousarray(signal, dtype=np.float64)


def _validate_lag_entry(
    indices: tuple[int, int], lag: object, *, n_layers: int
) -> tuple[int, int, float]:
    if not isinstance(indices, tuple) or len(indices) != 2:
        raise ValueError("lag index must be a pair of layer indices")
    i, j = indices
    if (
        isinstance(i, bool | np.bool_)
        or isinstance(j, bool | np.bool_)
        or not isinstance(i, Integral)
        or not isinstance(j, Integral)
    ):
        raise ValueError("lag index must contain integer layer indices")
    i = int(i)
    j = int(j)
    if i == j:
        raise ValueError("lag index must not target the alpha diagonal")
    if not (0 <= i < n_layers and 0 <= j < n_layers):
        raise ValueError("lag index must be within n_layers")
    if isinstance(lag, bool | np.bool_) or not isinstance(lag, Real):
        raise ValueError("lag estimate must be a finite real")
    lag_value = float(lag)
    if not np.isfinite(lag_value):
        raise ValueError("lag estimate must be a finite real")
    return i, j, lag_value


class LagModel:
    """Phase-lag estimation and alpha matrix construction."""

    @staticmethod
    def estimate_from_distances(distances: FloatArray, speed: float) -> FloatArray:
        """Build antisymmetric alpha matrix from pairwise distances and speed.

        alpha[i,j] = 2*pi * distances[i,j] / speed.
        Matches the Rust ``LagModel::estimate_from_distances`` algorithm.

        Parameters
        ----------
        distances : FloatArray
            Pairwise distance matrix, shape ``(N, N)``.
        speed : float
            Signal propagation speed.

        Returns
        -------
        FloatArray
            The antisymmetric phase-lag matrix, shape ``(N, N)``.
        """
        distances = _validate_distances(distances)
        speed = _validate_positive_real(speed, name="speed")
        n = distances.shape[0]
        alpha: FloatArray = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                lag = 2.0 * np.pi * distances[i, j] / speed
                alpha[i, j] = lag
                alpha[j, i] = -lag
        return alpha

    def estimate_lag(
        self, signal_a: FloatArray, signal_b: FloatArray, sample_rate: float
    ) -> float:
        """Cross-correlation peak lag in seconds between two signals.

        Parameters
        ----------
        signal_a : FloatArray
            First signal, shape ``(T,)``.
        signal_b : FloatArray
            Second signal, shape ``(T,)``.
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        float
            The cross-correlation peak lag in seconds.

        Raises
        ------
        ValueError
            If the signals differ in length or the sample rate is invalid.
        """
        signal_a = _validate_signal(signal_a, name="signal_a")
        signal_b = _validate_signal(signal_b, name="signal_b")
        if signal_a.shape != signal_b.shape:
            raise ValueError("signals must be the same finite one-dimensional arrays")
        sample_rate = _validate_positive_real(sample_rate, name="sample_rate")
        corr = np.correlate(
            signal_a - signal_a.mean(), signal_b - signal_b.mean(), mode="full"
        )
        peak_idx = np.argmax(corr)
        lag_samples = peak_idx - (len(signal_a) - 1)
        return float(lag_samples / sample_rate)

    def build_alpha_matrix(
        self,
        lag_estimates: dict[tuple[int, int], float],
        n_layers: int,
        carrier_freq_hz: float = 1.0,
    ) -> FloatArray:
        """Pairwise lag estimates (seconds) to phase-offset matrix (radians).

        alpha[i,j] = 2*pi*carrier_freq_hz*lag[i,j], antisymmetric.
        ``carrier_freq_hz`` defaults to 1.0 for backward compatibility.

        Parameters
        ----------
        lag_estimates : dict[tuple[int, int], float]
            Mapping of oscillator pair to estimated lag in seconds.
        n_layers : int
            Number of SCPN hierarchy layers.
        carrier_freq_hz : float
            Carrier frequency in Hz used to convert lag to phase.

        Returns
        -------
        FloatArray
            The phase-offset matrix in radians, shape ``(N, N)``.
        """
        n_layers = _validate_n_layers(n_layers)
        carrier_freq_hz = _validate_positive_real(
            carrier_freq_hz, name="carrier_freq_hz"
        )
        alpha: FloatArray = np.zeros((n_layers, n_layers), dtype=np.float64)
        for indices, lag in lag_estimates.items():
            i, j, lag = _validate_lag_entry(indices, lag, n_layers=n_layers)
            offset = 2.0 * np.pi * carrier_freq_hz * lag
            alpha[i, j] = offset
            alpha[j, i] = -offset
        return alpha
