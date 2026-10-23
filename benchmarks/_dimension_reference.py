# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Independent dimension benchmark oracles

"""Scalar geometric counts and rational Kaplan–Yorke benchmark references."""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def reference_outputs(
    trajectory: FloatArray, epsilons: FloatArray, spectrum: FloatArray
) -> tuple[FloatArray, float]:
    """Evaluate full-pair geometry and exact rational spectrum interpolation.

    Parameters
    ----------
    trajectory, epsilons, spectrum : numpy.ndarray
        Finite benchmark inputs, with sorted nonnegative radii and at least
        two trajectory samples. Every binary64 exponent is represented exactly.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Strict-radius scalar distance fractions and rounded rational KY value.
        Neither reference calls a production dimension solver or validator.
    """
    distances = [
        math.dist(left, right)
        for index, left in enumerate(trajectory)
        for right in trajectory[index + 1 :]
    ]
    fractions = np.array(
        [
            sum(distance < radius for distance in distances) / len(distances)
            for radius in epsilons
        ],
        dtype=np.float64,
    )
    ordered = sorted((Fraction(float(item)) for item in spectrum), reverse=True)
    total = Fraction(0)
    for index, value in enumerate(ordered):
        if total + value < 0:
            return fractions, float(index + total / abs(value))
        total += value
    return fractions, float(len(ordered))


def output_errors(
    actual: tuple[FloatArray, float], expected: tuple[FloatArray, float]
) -> dict[str, float | bool]:
    """Describe shape, finite, monotonic and scalar reference agreement.

    Parameters
    ----------
    actual, expected : tuple[numpy.ndarray, float]
        Public CI/KY result and independently evaluated reference.

    Returns
    -------
    dict[str, float | bool]
        Error magnitudes and contracts. Shape or finite refusal has infinite
        error; callers must refuse the result before serializing success data.
    """
    ci, ky = actual
    ref_ci, ref_ky = expected
    valid = ci.shape == ref_ci.shape and bool(np.all(np.isfinite(ci)))
    ci_error = float(np.max(np.abs(ci - ref_ci))) if valid else math.inf
    ky_error = abs(ky - ref_ky) if math.isfinite(ky) else math.inf
    return {
        "ci_max_abs_error": ci_error,
        "ky_abs_error": ky_error,
        "max_abs_error": max(ci_error, ky_error),
        "ci_monotonic": valid and bool(np.all(np.diff(ci) >= 0.0)),
        "ci_unit_interval": valid and bool(np.all((ci >= 0.0) & (ci <= 1.0))),
    }


def array_sha256(value: FloatArray) -> str:
    """Hash a canonical float64 result including its shape.

    Parameters
    ----------
    value : numpy.ndarray
        Array to serialize in contiguous little-endian float64 form.

    Returns
    -------
    str
        SHA-256 of the shape and numerical bytes.
    """
    contiguous = np.ascontiguousarray(value, dtype="<f8")
    payload = json.dumps(contiguous.shape, separators=(",", ":")).encode()
    return hashlib.sha256(payload + contiguous.tobytes()).hexdigest()


def payload_sha256(value: object) -> str:
    """Hash finite JSON content independently of formatting.

    Parameters
    ----------
    value : object
        JSON-serializable numerical evidence excluding wall-clock measurements.

    Returns
    -------
    str
        Canonical content SHA-256.
    """
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()
