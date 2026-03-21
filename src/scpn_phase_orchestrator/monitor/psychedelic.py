# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Psychedelic simulation protocol
#
# Carhart-Harris et al. 2014, Front. Hum. Neurosci. 8:20
# ("The entropic brain: a theory of conscious states informed by
# neuroimaging research with psychedelic drugs")

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.chimera import detect_chimera
from scpn_phase_orchestrator.upde.engine import UPDEEngine

__all__ = [
    "entropy_from_phases",
    "reduce_coupling",
    "simulate_psychedelic_trajectory",
]


def reduce_coupling(knm: NDArray, reduction_factor: float) -> NDArray:
    """Scale coupling matrix by (1 - reduction_factor).

    Args:
        knm: coupling matrix, shape (n, n).
        reduction_factor: fraction to reduce, in [0, 1].

    Returns:
        Scaled copy. Zero when reduction_factor == 1.
    """
    return np.asarray(knm, dtype=np.float64) * (1.0 - reduction_factor)


def entropy_from_phases(phases: NDArray) -> float:
    """Circular entropy of phase distribution.

    Discretises phases into 36 bins (10-degree resolution) and computes
    Shannon entropy in nats.
    """
    phases = np.asarray(phases, dtype=np.float64)
    if phases.size == 0:
        return 0.0
    wrapped = phases % (2.0 * np.pi)
    n_bins = 36
    counts, _ = np.histogram(wrapped, bins=n_bins, range=(0, 2.0 * np.pi))
    probs = counts / counts.sum()
    # Filter zeros to avoid log(0)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def simulate_psychedelic_trajectory(
    engine: UPDEEngine,
    phases: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    reduction_schedule: list[float],
    n_steps_per_level: int = 100,
) -> list[dict]:
    """Progressively reduce coupling, recording observables at each level.

    Models the entropic brain hypothesis: reduced serotonergic gating
    (coupling reduction) increases neural entropy and breaks coherent
    states into chimera-like patterns.

    Args:
        engine: UPDE integrator instance.
        phases: initial oscillator phases, shape (n,).
        omegas: natural frequencies, shape (n,).
        knm: baseline coupling matrix, shape (n, n).
        alpha: phase-lag matrix, shape (n, n).
        reduction_schedule: list of reduction_factor values (0 to 1).
        n_steps_per_level: integration steps at each coupling level.

    Returns:
        List of dicts, one per level, with keys:
          reduction_factor, R, entropy, chimera_index, phases.
    """
    p = np.asarray(phases, dtype=np.float64).copy()
    results: list[dict] = []

    for rf in reduction_schedule:
        k_reduced = reduce_coupling(knm, rf)
        p = engine.run(p, omegas, k_reduced, zeta=0.0, psi=0.0, alpha=alpha, n_steps=n_steps_per_level)
        r_val, _ = engine.compute_order_parameter(p)
        ent = entropy_from_phases(p)
        chimera = detect_chimera(p, k_reduced)

        results.append({
            "reduction_factor": rf,
            "R": r_val,
            "entropy": ent,
            "chimera_index": chimera.chimera_index,
            "phases": p.copy(),
        })

    return results
