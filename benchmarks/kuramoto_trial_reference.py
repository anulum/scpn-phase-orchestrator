# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Independent scalar Euler measurement

"""Scalar Euler oracle independent of every production trial implementation."""

from __future__ import annotations

import math
from collections.abc import Sequence


def scalar_trial(
    phases: Sequence[float],
    omegas: Sequence[float],
    coupling: Sequence[Sequence[float]],
    alpha: Sequence[Sequence[float]],
    *,
    scale: float = 1.0,
    dt: float = 0.01,
    transient: int = 0,
    measure: int = 1,
) -> float:
    """Measure R with a separate scalar full-snapshot Euler calculation.

    Parameters
    ----------
    phases, omegas : sequence of float
        Initial radians and natural frequencies in radians per second.
    coupling, alpha : sequence of sequence of float
        Target-row/source-column rate coupling and radian lags.
    scale : float
        Coupling multiplier, without implicit population normalization.
    dt : float
        Positive integration timestep in seconds.
    transient, measure : int
        Discarded and post-step measurement counts; an empty window returns zero.

    Returns
    -------
    float
        Mean magnitude of the oscillator phasor average over the chosen window.

    Raises
    ------
    ValueError
        If dimensions, finite inputs, counts, or numerical intermediates fail.
    """
    n = len(phases)
    if n == 0 or len(omegas) != n or len(coupling) != n or len(alpha) != n:
        raise ValueError("oracle dimensions must match a positive population")
    if any(len(row) != n for row in (*coupling, *alpha)):
        raise ValueError("oracle matrices must be square")
    values = (*phases, *omegas, *(v for row in (*coupling, *alpha) for v in row))
    if not all(math.isfinite(value) for value in (*values, scale, dt)) or dt <= 0:
        raise ValueError("oracle inputs must be finite with positive dt")
    if transient < 0 or measure < 0:
        raise ValueError("oracle step counts must be nonnegative")
    if measure == 0:
        return 0.0
    current = list(phases)
    total = 0.0
    for step in range(transient + measure):
        updated: list[float] = []
        for i in range(n):
            force = 0.0
            for j in range(n):
                weight = coupling[i][j] * scale
                if not math.isfinite(weight):
                    raise ValueError("oracle scaled coupling overflow")
                if weight == 0.0:
                    continue
                angle = current[j] - current[i] - alpha[i][j]
                if not math.isfinite(angle):
                    raise ValueError("oracle phase difference overflow")
                force += weight * math.sin(angle)
            velocity = omegas[i] + force
            value = current[i] + dt * velocity
            if not math.isfinite(velocity) or not math.isfinite(value):
                raise ValueError("oracle Euler step overflow")
            updated.append(value)
        current = updated
        if step >= transient:
            cosine = sum(math.cos(value) for value in current) / n
            sine = sum(math.sin(value) for value in current) / n
            total += math.hypot(cosine, sine)
    return min(total / measure, 1.0)
