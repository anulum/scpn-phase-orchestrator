# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Thermodynamic entropy production rate

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["entropy_production_rate"]


def entropy_production_rate(
    phases: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: float,
    dt: float,
) -> float:
    """Thermodynamic dissipation rate: Σ_i (dθ_i/dt)² · dt.

    dθ_i/dt = ω_i + (α/N) Σ_j K_ij sin(θ_j - θ_i)

    This measures the total power dissipated in the overdamped Kuramoto
    system. Zero at frequency-locked fixed points; positive otherwise.

    Acebrón et al. 2005, Rev. Mod. Phys. 77:137-185.
    """
    n = len(phases)
    if n == 0 or dt <= 0.0:
        return 0.0

    diff = phases[np.newaxis, :] - phases[:, np.newaxis]  # θ_j - θ_i
    coupling = np.sum(knm * np.sin(diff), axis=1)  # Σ_j K_ij sin(θ_j - θ_i)
    dtheta_dt = omegas + (alpha / n) * coupling
    return float(np.sum(dtheta_dt**2) * dt)
