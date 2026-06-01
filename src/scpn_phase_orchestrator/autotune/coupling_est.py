# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling estimation from phase data

"""Least-squares coupling estimators for observed phase trajectories.

The primary estimator fits pairwise sinusoidal Kuramoto coupling from
phase-history derivatives and natural frequencies, returning a dense matrix
with zero diagonal. The harmonics variant expands the regression library with
higher Fourier sine and cosine terms. Both routines are offline inference
helpers: they estimate parameters from caller-provided arrays and perform no
runtime actuation or binding updates.
"""

from __future__ import annotations

import contextlib
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["estimate_coupling", "estimate_coupling_harmonics"]

FloatArray: TypeAlias = NDArray[np.float64]


def _contains_boolean_alias(value: object) -> bool:
    raw = np.asarray(value, dtype=object)
    return any(isinstance(item, (bool, np.bool_)) for item in raw.ravel())


def _contains_complex_alias(value: object) -> bool:
    raw = np.asarray(value, dtype=object)
    return any(isinstance(item, (complex, np.complexfloating)) for item in raw.ravel())


def _validate_inputs(
    phases: object,
    omegas: object,
    dt: object,
) -> tuple[FloatArray, FloatArray, float]:
    if isinstance(dt, (bool, np.bool_)) or not isinstance(dt, Real):
        raise ValueError("dt must be a finite positive real")
    dt_value = float(dt)
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError("dt must be a finite positive real")

    if _contains_boolean_alias(phases):
        raise ValueError("phases must not contain boolean values")
    raw_phases = np.asarray(phases)
    if raw_phases.dtype == np.bool_:
        raise ValueError("phases must not contain boolean values")
    if np.iscomplexobj(raw_phases) or _contains_complex_alias(raw_phases):
        raise ValueError("phases must be a finite 2-D trajectory matrix")
    try:
        phases_array = np.asarray(raw_phases, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("phases must be a finite 2-D trajectory matrix") from exc
    if phases_array.ndim != 2:
        raise ValueError("phases must be a finite 2-D trajectory matrix")
    if phases_array.shape[0] < 1:
        raise ValueError("phases must contain at least one oscillator")
    if not np.all(np.isfinite(phases_array)):
        raise ValueError("phases must contain only finite values")

    if _contains_boolean_alias(omegas):
        raise ValueError("omegas must not contain boolean values")
    raw_omegas = np.asarray(omegas)
    if raw_omegas.dtype == np.bool_:
        raise ValueError("omegas must not contain boolean values")
    if np.iscomplexobj(raw_omegas) or _contains_complex_alias(raw_omegas):
        raise ValueError("omegas must be a finite 1-D frequency vector")
    try:
        omegas_array = np.asarray(raw_omegas, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("omegas must be a finite 1-D frequency vector") from exc
    if omegas_array.ndim != 1:
        raise ValueError("omegas must be a finite 1-D frequency vector")
    if not np.all(np.isfinite(omegas_array)):
        raise ValueError("omegas must contain only finite values")
    if omegas_array.size != phases_array.shape[0]:
        raise ValueError("omegas length must match oscillator count")

    return phases_array, omegas_array, dt_value


def _validate_n_harmonics(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("n_harmonics must be a positive integer")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError("n_harmonics must be a positive integer")
    return resolved


def estimate_coupling(
    phases: FloatArray,
    omegas: FloatArray,
    dt: float,
) -> FloatArray:
    """Estimate K_ij coupling matrix from observed phase trajectories.

    Least-squares fit of dθ_i/dt - ω_i = Σ_j K_ij sin(θ_j - θ_i).
    Constructs the regression matrix from pairwise sin(Δθ) and solves
    for K_ij via pseudoinverse.

    Args:
        phases: (n_oscillators, n_timesteps) phase trajectories.
        omegas: (n_oscillators,) natural frequencies.
        dt: timestep between samples.

    Returns:
        (n_oscillators, n_oscillators) estimated coupling matrix K_ij.
    """
    phases, omegas, dt = _validate_inputs(phases, omegas, dt)
    n, T = phases.shape
    if T < 3:
        raise ValueError(f"Need >= 3 timesteps, got {T}")

    # Phase derivative: (dθ/dt)_i ≈ (θ_{t+1} - θ_{t-1}) / (2*dt)
    dphase = np.diff(np.unwrap(phases, axis=1), axis=1) / dt
    # Use interior points for derivative
    phases_mid = phases[:, :-1]
    T_eff = dphase.shape[1]

    knm = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        # Target: dθ_i/dt - ω_i at each timestep
        target = dphase[i, :] - omegas[i]

        # Regressor: sin(θ_j - θ_i) for each j, at each timestep
        regressors = np.sin(
            phases_mid[:, :T_eff] - phases_mid[i : i + 1, :T_eff]
        )  # (n, T_eff)

        # Least squares: target = K_i · regressors
        # K_i = target @ regressors^T @ (regressors @ regressors^T)^{-1}
        with contextlib.suppress(np.linalg.LinAlgError):
            coeffs = np.linalg.lstsq(regressors.T, target, rcond=None)[0]
            if np.all(np.isfinite(coeffs)):
                knm[i, :] = coeffs

    np.fill_diagonal(knm, 0.0)
    return knm


def estimate_coupling_harmonics(
    phases: FloatArray,
    omegas: FloatArray,
    dt: float,
    n_harmonics: int = 2,
) -> dict[str, FloatArray]:
    """Estimate coupling with higher Fourier harmonics.

    Fits: dθ_i/dt - ω_i = Σ_j Σ_k [a_jk sin(k·Δθ) + b_jk cos(k·Δθ)]
    for k = 1..n_harmonics.

    Real biological oscillators have non-sinusoidal coupling
    (Stankovski 2017, Rev. Mod. Phys.).

    Returns dict with keys 'sin_1', 'cos_1', 'sin_2', 'cos_2', ...
    each an (n, n) matrix of coefficients.
    """
    phases, omegas, dt = _validate_inputs(phases, omegas, dt)
    n_harmonics = _validate_n_harmonics(n_harmonics)
    n, T = phases.shape
    if T < 3:
        raise ValueError(f"Need >= 3 timesteps, got {T}")

    dphase = np.diff(np.unwrap(phases, axis=1), axis=1) / dt
    phases_mid = phases[:, :-1]
    T_eff = dphase.shape[1]

    result: dict[str, FloatArray] = {}
    for k in range(1, n_harmonics + 1):
        result[f"sin_{k}"] = np.zeros((n, n), dtype=np.float64)
        result[f"cos_{k}"] = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        target = dphase[i, :] - omegas[i]
        diff = phases_mid[:, :T_eff] - phases_mid[i : i + 1, :T_eff]

        # Build regressor matrix: [sin(Δθ), cos(Δθ), sin(2Δθ), cos(2Δθ), ...]
        blocks = []
        for k in range(1, n_harmonics + 1):
            blocks.append(np.sin(k * diff))
            blocks.append(np.cos(k * diff))
        regressors = np.vstack(blocks)  # (2*n_harmonics*n, T_eff)

        try:
            coeffs = np.linalg.lstsq(regressors.T, target, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(coeffs)):
            continue

        # Unpack coefficients
        idx = 0
        for k in range(1, n_harmonics + 1):
            result[f"sin_{k}"][i, :] = coeffs[idx : idx + n]
            idx += n
            result[f"cos_{k}"][i, :] = coeffs[idx : idx + n]
            idx += n

    for key in result:
        np.fill_diagonal(result[key], 0.0)

    return result
