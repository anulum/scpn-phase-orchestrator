# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Three-factor Hebbian plasticity for coupling

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["compute_eligibility", "three_factor_update"]

FloatArray: TypeAlias = NDArray[np.float64]


def compute_eligibility(phases: FloatArray) -> FloatArray:
    """Pairwise Hebbian eligibility trace: cos(theta_j - theta_i).

    Returns shape (n, n) with zero diagonal.
    """
    phases = np.asarray(phases, dtype=np.float64)
    diffs = phases[np.newaxis, :] - phases[:, np.newaxis]
    elig = np.cos(diffs)
    np.fill_diagonal(elig, 0.0)
    result: FloatArray = elig
    return result


def three_factor_update(
    knm: FloatArray,
    eligibility: FloatArray,
    modulator: float,
    phase_gate: bool,
    lr: float = 0.01,
) -> FloatArray:
    """Three-factor plasticity rule: K_ij += lr * eligibility_ij * M * gate.

    Factors:
        1. eligibility — pairwise phase correlation (Hebbian trace)
        2. modulator — scalar reward/error signal from L16 director
        3. phase_gate — boolean from TCBO consciousness boundary

    Friston 2005, Philos. Trans. R. Soc. B
    360:815-836 (free energy & synaptic plasticity).

    Args:
        knm: current coupling matrix, shape (n, n).
        eligibility: Hebbian trace, shape (n, n).
        modulator: scalar neuromodulatory signal.
        phase_gate: if False, no update occurs (TCBO below consciousness threshold).
        lr: learning rate.

    Returns:
        Updated coupling matrix (new array, does not mutate input).
    """
    knm = np.asarray(knm, dtype=np.float64)
    if not phase_gate:
        return knm.copy()
    delta = lr * eligibility * modulator
    return knm + delta
