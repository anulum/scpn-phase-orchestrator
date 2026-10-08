# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Independent scalar chimera equation reference

"""Evaluate the original phase-difference equation independently of all owners."""

import cmath
import math

import numpy as np
from numpy.typing import NDArray


def scalar_local_order(
    phases: NDArray[np.float64], knm: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate the unweighted positive non-self neighbourhood phasor magnitude.

    Parameters
    ----------
    phases : numpy.ndarray
        Admitted N finite phases with finite pairwise differences, in radians.
        Comparison workloads use moderate angles; extreme-angle tests use
        analytic zero/single-neighbour identities instead of this oracle.
    knm : numpy.ndarray
        Admitted finite N by N target-row/source-column coupling.

    Returns
    -------
    numpy.ndarray
        Scalar complex-exponential sums, with zero for isolated oscillators.
    """
    n = int(phases.size)
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        neighbours = [j for j in range(n) if j != i and knm[i, j] > 0.0]
        if not neighbours:
            continue
        phasors = [
            cmath.exp(complex(0.0, float(phases[j]) - float(phases[i])))
            for j in neighbours
        ]
        degree = len(neighbours)
        result[i] = math.hypot(
            math.fsum(z.real for z in phasors) / degree,
            math.fsum(z.imag for z in phasors) / degree,
        )
    return result
