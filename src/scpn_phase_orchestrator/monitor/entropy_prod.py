# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Thermodynamic entropy production rate

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import HAS_RUST as _HAS_RUST

__all__ = ["entropy_production_rate"]


def entropy_production_rate(
    phases: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: float,
    dt: float,
) -> float:
    """Thermodynamic dissipation rate: Sigma_i (dtheta_i/dt)^2 * dt.

    dtheta_i/dt = omega_i + (alpha/N) Sigma_j K_ij sin(theta_j - theta_i)

    This measures the total power dissipated in the overdamped Kuramoto
    system. Zero at frequency-locked fixed points; positive otherwise.

    Acebron et al. 2005, Rev. Mod. Phys. 77:137-185.
    """
    n = len(phases)
    if n == 0 or dt <= 0.0:
        return 0.0

    if _HAS_RUST:  # pragma: no cover
        from spo_kernel import entropy_production_rate as _rust_ep

        return float(
            _rust_ep(
                np.ascontiguousarray(phases.ravel()),
                np.ascontiguousarray(omegas.ravel()),
                np.ascontiguousarray(knm.ravel()),
                alpha,
                dt,
            )
        )

    diff = phases[np.newaxis, :] - phases[:, np.newaxis]
    coupling = np.sum(knm * np.sin(diff), axis=1)
    dtheta_dt = omegas + (alpha / n) * coupling
    return float(np.sum(dtheta_dt**2) * dt)
