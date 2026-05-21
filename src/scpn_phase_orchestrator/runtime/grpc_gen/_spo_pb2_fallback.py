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
from math import isfinite
from numbers import Real

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

    def __post_init__(self) -> None:
        if (
            not isinstance(self.n_steps, int)
            or isinstance(self.n_steps, bool)
            or self.n_steps <= 0
        ):
            raise ValueError("n_steps must be a positive integer")


@dataclass
class ResetRequest:
    """Fallback request to reset orchestrator state."""

    pass


@dataclass
class StreamRequest:
    """Fallback request for bounded streaming of phase-state snapshots."""

    max_steps: int = 100
    interval_s: float = 0.05

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_steps, int)
            or isinstance(self.max_steps, bool)
            or self.max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer")
        if (
            not isinstance(self.interval_s, Real)
            or isinstance(self.interval_s, bool)
            or not isfinite(float(self.interval_s))
            or float(self.interval_s) < 0.0
        ):
            raise ValueError("interval_s must be a finite non-negative real")


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

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError("name must be a string")
        if any(ord(char) < 32 for char in self.name):
            raise ValueError("name must not contain control characters")
        if (
            not isinstance(self.R, Real)
            or isinstance(self.R, bool)
            or not isfinite(float(self.R))
        ):
            raise ValueError("R must be a finite real")
        if (
            not isinstance(self.psi, Real)
            or isinstance(self.psi, bool)
            or not isfinite(float(self.psi))
        ):
            raise ValueError("psi must be a finite real")


@dataclass
class StateResponse:
    """Fallback response containing runtime state and layer summaries."""

    step: int = 0
    R_global: float = 0.0
    regime: str = ""
    amplitude_mode: bool = False
    mean_amplitude: float = 0.0
    layers: list[LayerState] = field(default_factory=list)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.step, int)
            or isinstance(self.step, bool)
            or self.step < 0
        ):
            raise ValueError("step must be a non-negative integer")
        if (
            not isinstance(self.R_global, Real)
            or isinstance(self.R_global, bool)
            or not isfinite(float(self.R_global))
        ):
            raise ValueError("R_global must be a finite real")
        if not isinstance(self.regime, str):
            raise ValueError("regime must be a string")
        if any(ord(char) < 32 for char in self.regime):
            raise ValueError("regime must not contain control characters")
        if not isinstance(self.amplitude_mode, bool):
            raise ValueError("amplitude_mode must be a bool")
        if (
            not isinstance(self.mean_amplitude, Real)
            or isinstance(self.mean_amplitude, bool)
            or not isfinite(float(self.mean_amplitude))
        ):
            raise ValueError("mean_amplitude must be a finite real")
        if not isinstance(self.layers, list):
            raise ValueError("layers must be a list of LayerState")
        for item in self.layers:
            if not isinstance(item, LayerState):
                raise ValueError("layers must be a list of LayerState")


@dataclass
class ConfigResponse:
    """Fallback response containing static binding and timing configuration."""

    name: str = ""
    n_oscillators: int = 0
    n_layers: int = 0
    amplitude_mode: bool = False
    sample_period_s: float = 0.0
    control_period_s: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError("name must be a string")
        if any(ord(char) < 32 for char in self.name):
            raise ValueError("name must not contain control characters")
        for field_name in ("n_oscillators", "n_layers"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if not isinstance(self.amplitude_mode, bool):
            raise ValueError("amplitude_mode must be a bool")
        for field_name in ("sample_period_s", "control_period_s"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{field_name} must be a finite non-negative real")
