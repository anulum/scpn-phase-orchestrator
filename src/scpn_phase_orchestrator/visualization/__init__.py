# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Visualization components

"""Public visualization facade for JSON encoders and streaming helpers.

The package exports deterministic JSON builders for coupling graphs, heatmaps,
phase wheels, torus coordinates, and an optional WebSocket streamer for live
visual clients. Visualization helpers validate presentation inputs and serialize
state for external renderers only; they do not mutate solver state, activate
control paths, or perform scientific inference.
"""

from __future__ import annotations

from scpn_phase_orchestrator.visualization.network import (
    coupling_heatmap_json,
    network_graph_json,
)
from scpn_phase_orchestrator.visualization.streamer import VisualizerStreamer
from scpn_phase_orchestrator.visualization.torus import (
    phase_wheel_json,
    torus_points_json,
)

__all__ = [
    "VisualizerStreamer",
    "coupling_heatmap_json",
    "network_graph_json",
    "phase_wheel_json",
    "torus_points_json",
]
