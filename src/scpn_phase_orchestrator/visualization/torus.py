# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Three.js torus visualization data

from __future__ import annotations

import json
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["torus_points_json", "phase_wheel_json"]

FloatArray: TypeAlias = NDArray[np.float64]


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
    n = len(phases)
    if R_values is None:
        R_values = [1.0] * n

    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    R = major_radius
    r = minor_radius

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
                "R": round(float(R_values[i]), 4),
                "id": i,
            }
        )

    return json.dumps({"points": points}, indent=2)


def phase_wheel_json(phases: FloatArray, layer_names: list[str] | None = None) -> str:
    """Phase wheel data: oscillator phases as polar coordinates.

    Each oscillator is a point at angle=θ_i, radius=1 on the unit circle.
    """
    n = len(phases)
    if layer_names is None:
        layer_names = [f"L{i}" for i in range(n)]

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
