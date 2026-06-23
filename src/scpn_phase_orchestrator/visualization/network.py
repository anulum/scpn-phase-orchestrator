# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — D3 network graph visualization

"""D3-compatible JSON encoders for validated coupling network views.

The module turns finite square coupling matrices into force-graph and heatmap
payloads with optional layer names and per-node coherence metrics. Inputs reject
boolean, complex, non-numeric, non-finite, non-square, and length-mismatched
values before serialization. The functions are pure presentation encoders:
they return JSON strings and do not change coupling matrices or runtime state.
"""

from __future__ import annotations

import json
from math import isfinite
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["network_graph_json", "coupling_heatmap_json"]

FloatArray: TypeAlias = NDArray[np.float64]


def _validate_coupling_matrix(value: object, *, name: str) -> FloatArray:
    """Return the coupling as a validated finite square matrix, else raise."""
    matrix = np.asarray(value)
    dtype = matrix.dtype
    if (
        np.issubdtype(dtype, np.bool_)
        or np.issubdtype(dtype, np.complexfloating)
        or not np.issubdtype(dtype, np.number)
    ):
        raise ValueError(f"{name} must be finite")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    parsed = matrix.astype(np.float64, copy=False)
    if not np.all(np.isfinite(parsed)):
        raise ValueError(f"{name} must be finite")
    return parsed


def _validate_metric_values(
    value: object,
    *,
    name: str,
    expected_length: int,
) -> FloatArray:
    """Return the per-node metric values as a validated finite array, else raise."""
    values = np.asarray(value)
    dtype = values.dtype
    if (
        np.issubdtype(dtype, np.bool_)
        or np.issubdtype(dtype, np.complexfloating)
        or not np.issubdtype(dtype, np.number)
    ):
        raise ValueError(f"{name} must be finite")
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {values.shape}")
    parsed = values.astype(np.float64, copy=False)
    if len(parsed) != expected_length:
        raise ValueError(
            f"{name} length must match node count {expected_length}, got {len(parsed)}"
        )
    if not np.all(np.isfinite(parsed)):
        raise ValueError(f"{name} must be finite")
    return parsed


def _validate_layer_names(
    value: object,
    *,
    expected_length: int,
) -> list[str]:
    """Return the validated non-empty layer names, else raise."""
    if not isinstance(value, list):
        raise ValueError("layer_names must be a list of non-empty strings")
    if len(value) != expected_length:
        raise ValueError(
            f"layer_names length must match node count {expected_length}, "
            f"got {len(value)}"
        )
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError("layer_names must be non-empty strings")
    return value


def _validate_non_negative_real(value: object, *, name: str) -> float:
    """Return ``value`` as a non-negative finite real, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and non-negative")
    parsed = float(value)
    if not isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def network_graph_json(
    knm: FloatArray,
    layer_names: list[str] | None = None,
    R_values: list[float] | None = None,
    threshold: float = 0.01,
) -> str:
    """Generate D3 force-directed graph JSON from coupling matrix.

    Returns JSON with {nodes: [...], links: [...]} for D3.js consumption.
    Edges below threshold are omitted.

    Parameters
    ----------
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    layer_names : list[str] | None
        Per-layer names.
    R_values : list[float] | None
        Order-parameter values.
    threshold : float
        Decision threshold.

    Returns
    -------
    str
        D3 force-directed graph JSON from coupling matrix.
    """
    knm = _validate_coupling_matrix(knm, name="knm")
    threshold = _validate_non_negative_real(threshold, name="threshold")
    n = knm.shape[0]
    if layer_names is None:
        layer_names = [f"L{i}" for i in range(n)]
    else:
        layer_names = _validate_layer_names(layer_names, expected_length=n)
    if R_values is None:
        r_values: FloatArray = np.zeros(n, dtype=np.float64)
    else:
        r_values = _validate_metric_values(
            R_values,
            name="R_values",
            expected_length=n,
        )

    nodes = []
    for i in range(n):
        nodes.append(
            {
                "id": i,
                "name": layer_names[i],
                "R": round(float(r_values[i]), 4),
            }
        )

    links = []
    for i in range(n):
        for j in range(i + 1, n):
            w = float(knm[i, j])
            if abs(w) > threshold:
                links.append(
                    {
                        "source": i,
                        "target": j,
                        "weight": round(w, 4),
                    }
                )

    return json.dumps({"nodes": nodes, "links": links}, indent=2)


def coupling_heatmap_json(
    knm: FloatArray,
    layer_names: list[str] | None = None,
) -> str:
    """Generate heatmap data JSON from coupling matrix.

    Returns JSON with {labels: [...], matrix: [[...]]} for rendering.

    Parameters
    ----------
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    layer_names : list[str] | None
        Per-layer names.

    Returns
    -------
    str
        Heatmap data JSON from coupling matrix.
    """
    knm = _validate_coupling_matrix(knm, name="knm")
    n = knm.shape[0]
    if layer_names is None:
        layer_names = [f"L{i}" for i in range(n)]
    else:
        layer_names = _validate_layer_names(layer_names, expected_length=n)

    matrix = []
    for i in range(n):
        row = [round(float(knm[i, j]), 4) for j in range(n)]
        matrix.append(row)

    return json.dumps(
        {
            "labels": layer_names,
            "matrix": matrix,
            "min": round(float(np.min(knm)), 4),
            "max": round(float(np.max(knm)), 4),
        },
        indent=2,
    )
