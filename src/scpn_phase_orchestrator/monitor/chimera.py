# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Chimera state detection

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["ChimeraState", "detect_chimera"]

try:
    from spo_kernel import (
        detect_chimera_rust as _rust_dc,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

# Kuramoto & Battogtokh 2002, Nonlinear Phenom. Complex Syst. 5:380-385
_COHERENT_THRESHOLD = 0.7
_INCOHERENT_THRESHOLD = 0.3


@dataclass(frozen=True)
class ChimeraState:
    """Chimera detection result: coherent/incoherent oscillator partitions and index."""

    coherent_indices: list[int] = field(default_factory=list)
    incoherent_indices: list[int] = field(default_factory=list)
    chimera_index: float = 0.0


def _local_order_parameter(phases: NDArray, knm: NDArray) -> NDArray:
    """R_i = |mean(exp(i*(theta_j - theta_i))) for neighbors j|.

    Neighbors of oscillator i are those with K_ij > 0.
    """
    n = len(phases)
    r_local = np.zeros(n, dtype=np.float64)
    diffs = phases[np.newaxis, :] - phases[:, np.newaxis]  # (n, n): [i, j] = θ_j - θ_i
    unit_vectors = np.exp(1j * diffs)

    for i in range(n):
        neighbors = np.where(knm[i] > 0)[0]
        if len(neighbors) == 0:
            r_local[i] = 0.0
            continue
        r_local[i] = float(np.abs(np.mean(unit_vectors[i, neighbors])))

    return r_local


def detect_chimera(phases: NDArray, knm: NDArray) -> ChimeraState:
    """Detect chimera states in a Kuramoto oscillator network.

    Args:
        phases: oscillator phases, shape (n,).
        knm: coupling matrix, shape (n, n). Non-zero entries define neighbors.

    Returns:
        ChimeraState with coherent/incoherent partitions and chimera_index.
    """
    phases = np.asarray(phases, dtype=np.float64)
    knm = np.asarray(knm, dtype=np.float64)
    n = len(phases)

    if n == 0:
        return ChimeraState()

    if _HAS_RUST:
        coh, incoh, ci, _ = _rust_dc(
            np.ascontiguousarray(phases),
            np.ascontiguousarray(knm.ravel()),
            n,
        )
        return ChimeraState(
            coherent_indices=[int(i) for i in coh],
            incoherent_indices=[int(i) for i in incoh],
            chimera_index=float(ci),
        )

    r_local = _local_order_parameter(phases, knm)

    coherent = [int(i) for i in range(n) if r_local[i] > _COHERENT_THRESHOLD]
    incoherent = [int(i) for i in range(n) if r_local[i] < _INCOHERENT_THRESHOLD]
    boundary_count = n - len(coherent) - len(incoherent)
    chimera_index = boundary_count / n if n > 0 else 0.0

    return ChimeraState(
        coherent_indices=coherent,
        incoherent_indices=incoherent,
        chimera_index=chimera_index,
    )
