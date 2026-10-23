# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stable dimension arithmetic

"""Floating-point arithmetic shared by dimension reference validators."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def row_distances(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute Euclidean row distances without squaring extreme coordinates.

    Parameters
    ----------
    left, right : numpy.ndarray
        Finite arrays broadcastable to the same row shape.

    Returns
    -------
    numpy.ndarray
        Distances, including infinity when the true distance exceeds float64.
    """
    with np.errstate(over="ignore"):
        differences = left - right
    return np.asarray(
        np.hypot.reduce(differences, axis=-1, initial=0.0), dtype=np.float64
    )


def kaplan_yorke_reference(exponents: NDArray[np.float64]) -> float:
    """Evaluate Kaplan–Yorke interpolation using scaled compensated prefixes.

    Parameters
    ----------
    exponents : numpy.ndarray
        Validated finite one-dimensional float64 spectrum.

    Returns
    -------
    float
        Dimension in the interval from zero to the spectrum length.
    """
    if exponents.size == 0:
        return 0.0
    scale = float(np.max(np.abs(exponents)))
    if scale == 0.0:
        return float(exponents.size)
    spectrum = np.sort(exponents)[::-1] / scale
    total = 0.0
    correction = 0.0
    accepted = 0.0
    for index, item in enumerate(spectrum):
        value = float(item)
        updated = total + value
        if abs(total) >= abs(value):
            correction += (total - updated) + value
        else:
            correction += (value - updated) + total
        total = updated
        prefix = total + correction
        if prefix < 0.0:
            return float(index) + accepted / abs(value)
        accepted = prefix
    return float(spectrum.size)
