# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase winding number tracker

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["winding_numbers", "winding_vector"]

TWO_PI = 2.0 * np.pi


def winding_numbers(phases_history: NDArray) -> NDArray:
    """Cumulative winding number of each oscillator over a trajectory.

    winding_i = floor((θ_i(T) - θ_i(0)) / 2π)

    Counts how many full 2π rotations each oscillator completes.
    Positive = counterclockwise, negative = clockwise.

    Args:
        phases_history: (T, N) array — T timesteps, N oscillators.

    Returns:
        (N,) integer array of winding numbers.
    """
    if phases_history.ndim != 2 or phases_history.shape[0] < 2:
        return np.zeros(phases_history.shape[-1] if phases_history.ndim == 2 else 0, dtype=np.int64)

    # Unwrap via cumulative phase increments to handle wrap-around correctly
    dtheta = np.diff(phases_history, axis=0)
    # Wrap increments to [-π, π] to detect true direction
    dtheta_wrapped = (dtheta + np.pi) % TWO_PI - np.pi
    cumulative = np.sum(dtheta_wrapped, axis=0)
    return np.floor(cumulative / TWO_PI).astype(np.int64)


def winding_vector(phases_history: NDArray) -> NDArray:
    """N-dimensional integer classification vector from winding numbers.

    Same as winding_numbers but emphasises the vector interpretation:
    topologically distinct trajectories map to distinct integer lattice points.

    Args:
        phases_history: (T, N) array — T timesteps, N oscillators.

    Returns:
        (N,) integer array — the winding vector.
    """
    return winding_numbers(phases_history)
