# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — D3 network graph visualization

from __future__ import annotations

import json

import numpy as np
from numpy.typing import NDArray

__all__ = ["network_graph_json", "coupling_heatmap_json"]


def network_graph_json(
    knm: NDArray,
    layer_names: list[str] | None = None,
    R_values: list[float] | None = None,
    threshold: float = 0.01,
) -> str:
    """Generate D3 force-directed graph JSON from coupling matrix.

    Returns JSON with {nodes: [...], links: [...]} for D3.js consumption.
    Edges below threshold are omitted.
    """
    n = knm.shape[0]
    if layer_names is None:
        layer_names = [f"L{i}" for i in range(n)]
    if R_values is None:
        R_values = [0.0] * n

    nodes = []
    for i in range(n):
        nodes.append({
            "id": i,
            "name": layer_names[i],
            "R": round(float(R_values[i]), 4),
        })

    links = []
    for i in range(n):
        for j in range(i + 1, n):
            w = float(knm[i, j])
            if abs(w) > threshold:
                links.append({
                    "source": i,
                    "target": j,
                    "weight": round(w, 4),
                })

    return json.dumps({"nodes": nodes, "links": links}, indent=2)


def coupling_heatmap_json(
    knm: NDArray,
    layer_names: list[str] | None = None,
) -> str:
    """Generate heatmap data JSON from coupling matrix.

    Returns JSON with {labels: [...], matrix: [[...]]} for rendering.
    """
    n = knm.shape[0]
    if layer_names is None:
        layer_names = [f"L{i}" for i in range(n)]

    matrix = []
    for i in range(n):
        row = [round(float(knm[i, j]), 4) for j in range(n)]
        matrix.append(row)

    return json.dumps({
        "labels": layer_names,
        "matrix": matrix,
        "min": round(float(np.min(knm)), 4),
        "max": round(float(np.max(knm)), 4),
    }, indent=2)
