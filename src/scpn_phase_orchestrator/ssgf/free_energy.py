# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SSGF free energy and Langevin dynamics

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["add_langevin_noise", "boltzmann_weight", "effective_temperature"]


def add_langevin_noise(
    z: NDArray,
    temperature: float,
    dt: float,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Add Langevin stochastic noise to a z-space vector.

    z_new = z + sqrt(2·T·dt) · η,  η ~ N(0, I)

    Models thermal fluctuations in the SSGF geometry space, enabling
    escape from local minima during stochastic optimisation.

    Gardiner 2009, Stochastic Methods, §4.3.
    """
    if temperature <= 0.0 or dt <= 0.0:
        return z.copy()
    if rng is None:
        rng = np.random.default_rng()
    sigma = np.sqrt(2.0 * temperature * dt)
    noise = rng.standard_normal(z.shape)
    return z + sigma * noise


def boltzmann_weight(u_total: float, temperature: float) -> float:
    """Boltzmann factor exp(-U/T) for a given total energy and temperature.

    Clamps exponent to [-700, 700] to avoid over/underflow.
    Returns 1.0 at T=0 if U=0, else 0.0 for U>0 at T=0.
    """
    if temperature <= 0.0:
        return 1.0 if u_total <= 0.0 else 0.0
    exponent = -u_total / temperature
    exponent = max(-700.0, min(700.0, exponent))
    return float(np.exp(exponent))


def effective_temperature(costs_history: NDArray) -> float:
    """Estimate effective temperature from cost fluctuations.

    T_eff = Var(U) / (2 · <U>)

    Based on the fluctuation-dissipation relation: in equilibrium,
    the variance of energy is proportional to T² C_V, and for a
    single degree of freedom <U> ~ T/2.

    Returns 0.0 if the cost series is constant or too short.
    """
    if len(costs_history) < 2:
        return 0.0
    var = float(np.var(costs_history, ddof=1))
    mean = float(np.mean(costs_history))
    if abs(mean) < 1e-30:
        return 0.0
    return var / (2.0 * abs(mean))
