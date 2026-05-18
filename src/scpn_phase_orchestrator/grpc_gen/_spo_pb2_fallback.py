# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hand-written protobuf message stubs
#
# Mirrors proto/spo.proto. Re-generate real stubs with:
#     bash tools/generate_grpc.sh

"""Dataclass fallback messages mirroring the Phase Orchestrator protobuf API.

These classes provide a lightweight import-time substitute for generated
protobuf message classes when ``google.protobuf`` or generated artifacts are
unavailable. They preserve field names and default values used by local tests
and server fallback paths, but they do not implement protobuf serialization,
wire compatibility, descriptors, or validation beyond dataclass construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "ConfigRequest",
    "ConfigResponse",
    "LayerState",
    "ResetRequest",
    "StateRequest",
    "StateResponse",
    "StepRequest",
    "StreamRequest",
]


@dataclass
class StateRequest:
    """Fallback request for retrieving the current orchestrator state."""

    pass


@dataclass
class StepRequest:
    """Fallback request to advance the orchestrator by ``n_steps``."""

    n_steps: int = 1


@dataclass
class ResetRequest:
    """Fallback request to reset orchestrator state."""

    pass


@dataclass
class StreamRequest:
    """Fallback request for bounded streaming of phase-state snapshots."""

    max_steps: int = 100
    interval_s: float = 0.05


@dataclass
class ConfigRequest:
    """Fallback request for static orchestrator configuration."""

    pass


@dataclass
class LayerState:
    """Fallback per-layer coherence state returned by gRPC responses."""

    name: str = ""
    R: float = 0.0
    psi: float = 0.0


@dataclass
class StateResponse:
    """Fallback response containing runtime state and layer summaries."""

    step: int = 0
    R_global: float = 0.0
    regime: str = ""
    amplitude_mode: bool = False
    mean_amplitude: float = 0.0
    layers: list[LayerState] = field(default_factory=list)


@dataclass
class ConfigResponse:
    """Fallback response containing static binding and timing configuration."""

    name: str = ""
    n_oscillators: int = 0
    n_layers: int = 0
    amplitude_mode: bool = False
    sample_period_s: float = 0.0
    control_period_s: float = 0.0
