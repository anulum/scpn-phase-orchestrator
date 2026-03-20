# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anylum.li
# SCPN Phase Orchestrator — Excitatory/Inhibitory balance

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["EIBalance", "compute_ei_balance", "adjust_ei_ratio"]


@dataclass
class EIBalance:
    ratio: float
    excitatory_strength: float
    inhibitory_strength: float
    is_balanced: bool


def compute_ei_balance(
    knm: NDArray,
    excitatory_indices: list[int],
    inhibitory_indices: list[int],
) -> EIBalance:
    """Compute E/I balance from coupling matrix and layer typing.

    Kuroki & Mizuseki 2025, Neural Computation — E/I balance is the
    critical parameter for synchronization, not K or D.

    ratio > 1: excitation-dominated (hypersynchrony risk)
    ratio < 1: inhibition-dominated (desynchronization risk)
    ratio ≈ 1: balanced (optimal for metastability)
    """
    n = knm.shape[0]
    e_mask = np.zeros(n, dtype=bool)
    i_mask = np.zeros(n, dtype=bool)
    for idx in excitatory_indices:
        if idx < n:
            e_mask[idx] = True
    for idx in inhibitory_indices:
        if idx < n:
            i_mask[idx] = True

    # Excitatory strength: mean coupling FROM excitatory oscillators
    e_strength = float(np.mean(knm[e_mask, :])) if np.any(e_mask) else 0.0
    # Inhibitory strength: mean coupling FROM inhibitory oscillators
    i_strength = float(np.mean(knm[i_mask, :])) if np.any(i_mask) else 0.0

    if i_strength < 1e-15:
        ratio = float("inf") if e_strength > 0 else 1.0
    else:
        ratio = e_strength / i_strength

    return EIBalance(
        ratio=ratio,
        excitatory_strength=e_strength,
        inhibitory_strength=i_strength,
        is_balanced=0.8 <= ratio <= 1.2,
    )


def adjust_ei_ratio(
    knm: NDArray,
    excitatory_indices: list[int],
    inhibitory_indices: list[int],
    target_ratio: float = 1.0,
) -> NDArray:
    """Scale inhibitory coupling to achieve target E/I ratio.

    Returns modified knm with inhibitory rows scaled so that
    E_strength / I_strength ≈ target_ratio.
    """
    balance = compute_ei_balance(knm, excitatory_indices, inhibitory_indices)
    if balance.inhibitory_strength < 1e-15 or balance.excitatory_strength < 1e-15:
        return knm.copy()

    current_ratio = balance.ratio
    if abs(current_ratio - target_ratio) < 1e-10:
        return knm.copy()

    # Scale inhibitory rows: I_new = I_old * (current_ratio / target_ratio)
    scale = current_ratio / target_ratio
    result = knm.copy()
    n = knm.shape[0]
    for idx in inhibitory_indices:
        if idx < n:
            result[idx, :] *= scale
    return result
