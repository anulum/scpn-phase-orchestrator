# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Independent ethical-cost numerical reference

"""Evaluate the declared finite diagnostic with independent scalar reductions.

This is a comparison oracle, never a selectable production kernel. Eigenvalues
come from NumPy on an explicitly constructed reciprocal-magnitude Laplacian.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def reference_cost(
    phases: NDArray[np.float64],
    knm: NDArray[np.float64],
    **parameters: float,
) -> tuple[float, float, float, int]:
    """Return the SEC score, weighted penalty, total and violation count.

    Parameters
    ----------
    phases : numpy.typing.NDArray[numpy.float64]
        Finite raw phase vector for a mathematically representable fixture.
    knm : numpy.typing.NDArray[numpy.float64]
        Matching finite square real coupling matrix, including diagonal entries.
    **parameters : float
        Finite score weights and residual thresholds from the public API.

    Returns
    -------
    tuple[float, float, float, int]
        Independent score and squared-residual decomposition.
    """
    n = phases.size
    if n == 0:
        return 0.0, 0.0, 1.0, 0
    r = (
        math.hypot(
            math.fsum(math.cos(float(p)) for p in phases),
            math.fsum(math.sin(float(p)) for p in phases),
        )
        / n
    )
    adjacency = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            weight = (abs(float(knm[i, j])) + abs(float(knm[j, i]))) / 2.0
            adjacency[i, j] = adjacency[j, i] = weight
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    lam2 = max(0.0, float(np.linalg.eigvalsh(laplacian)[1])) if n > 1 else 0.0
    density = sum(float(v) != 0.0 for v in knm.flat) / (n * (n - 1)) if n > 1 else 0.0
    mean = math.fsum(float(p) for p in phases) / n
    dispersion = math.sqrt(math.fsum((float(p) - mean) ** 2 for p in phases) / n)
    score = (
        parameters.get("alpha_R", 0.4) * r
        + parameters.get("beta_K", 0.3) * lam2 / n
        + parameters.get("gamma_Q", 0.2) * density
        - parameters.get("nu_S", 0.1) * dispersion / math.pi
    )
    residuals = (
        parameters.get("R_min", 0.2) - r,
        parameters.get("connectivity_min", 0.1) - lam2,
        float(knm.max()) - parameters.get("max_coupling", 5.0)
        if np.any(knm > 0.0)
        else 0.0,
    )
    penalty = parameters.get("kappa", 1.0) * math.fsum(
        max(0.0, residual) ** 2 for residual in residuals
    )
    return score, penalty, 1.0 - score + penalty, sum(v > 0.0 for v in residuals)
