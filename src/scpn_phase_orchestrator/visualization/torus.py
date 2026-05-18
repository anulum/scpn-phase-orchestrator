# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Three.js torus visualization data

from __future__ import annotations

import json
from math import isfinite
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["torus_points_json", "phase_wheel_json"]

FloatArray: TypeAlias = NDArray[np.float64]


def _validate_phase_array(value: object, *, name: str) -> FloatArray:
    array = np.asarray(value)
    dtype = array.dtype
    if (
        np.issubdtype(dtype, np.bool_)
        or np.issubdtype(dtype, np.complexfloating)
        or not np.issubdtype(dtype, np.number)
    ):
        raise ValueError(f"{name} must be finite")
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    parsed = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(parsed)):
        raise ValueError(f"{name} must be finite")
    return parsed


def _validate_metric_values(
    value: object,
    *,
    name: str,
    expected_length: int,
) -> FloatArray:
    values = _validate_phase_array(value, name=name)
    if len(values) != expected_length:
        raise ValueError(
            f"{name} length must match phases length {expected_length}, "
            f"got {len(values)}"
        )
    return values


def _validate_layer_names(
    value: object,
    *,
    expected_length: int,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("layer_names must be a list of non-empty strings")
    if len(value) != expected_length:
        raise ValueError(
            f"layer_names length must match phases length {expected_length}, "
            f"got {len(value)}"
        )
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError("layer_names must be non-empty strings")
    return value


def _validate_positive_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def torus_points_json(
    phases: FloatArray,
    R_values: list[float] | None = None,
    major_radius: float = 2.0,
    minor_radius: float = 0.5,
) -> str:
    """Map oscillator phases to 3D torus coordinates for Three.js.

    Each oscillator at phase θ_i maps to a point on the torus:
      x = (R + r·cos(φ_i)) · cos(θ_i)
      y = (R + r·cos(φ_i)) · sin(θ_i)
      z = r · sin(φ_i)
    where φ_i = 2π·i/N distributes oscillators around the minor circle.
    """
    phases = _validate_phase_array(phases, name="phases")
    n = len(phases)
    if R_values is None:
        r_values = np.ones(n, dtype=np.float64)
    else:
        r_values = _validate_metric_values(
            R_values,
            name="R_values",
            expected_length=n,
        )

    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    R = _validate_positive_real(major_radius, name="major_radius")
    r = _validate_positive_real(minor_radius, name="minor_radius")

    points = []
    for i in range(n):
        theta = float(phases[i])
        p = float(phi[i])
        x = (R + r * np.cos(p)) * np.cos(theta)
        y = (R + r * np.cos(p)) * np.sin(theta)
        z = r * np.sin(p)
        points.append(
            {
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "z": round(float(z), 4),
                "phase": round(theta, 4),
                "R": round(float(r_values[i]), 4),
                "id": i,
            }
        )

    return json.dumps({"points": points}, indent=2)


def phase_wheel_json(phases: FloatArray, layer_names: list[str] | None = None) -> str:
    """Phase wheel data: oscillator phases as polar coordinates.

    Each oscillator is a point at angle=θ_i, radius=1 on the unit circle.
    """
    phases = _validate_phase_array(phases, name="phases")
    n = len(phases)
    if layer_names is None:
        layer_names = [f"L{i}" for i in range(n)]
    else:
        layer_names = _validate_layer_names(layer_names, expected_length=n)

    entries = []
    for i in range(n):
        theta = float(phases[i])
        entries.append(
            {
                "name": layer_names[i],
                "phase": round(theta, 4),
                "x": round(float(np.cos(theta)), 4),
                "y": round(float(np.sin(theta)), 4),
            }
        )

    return json.dumps({"oscillators": entries}, indent=2)
