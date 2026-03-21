# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling estimation from phase data

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["estimate_coupling"]


def estimate_coupling(
    phases: NDArray,
    omegas: NDArray,
    dt: float,
) -> NDArray:
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
    phases = np.atleast_2d(phases)
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
        try:
            knm[i, :] = np.linalg.lstsq(regressors.T, target, rcond=None)[0]
        except np.linalg.LinAlgError:  # pragma: no cover
            pass

    np.fill_diagonal(knm, 0.0)
    return knm
