# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hand-written protobuf message stubs
#
# Mirrors proto/spo.proto. Re-generate real stubs with:
#     bash tools/generate_grpc.sh

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
    pass


@dataclass
class StepRequest:
    n_steps: int = 1


@dataclass
class ResetRequest:
    pass


@dataclass
class StreamRequest:
    max_steps: int = 100
    interval_s: float = 0.05


@dataclass
class ConfigRequest:
    pass


@dataclass
class LayerState:
    name: str = ""
    R: float = 0.0
    psi: float = 0.0


@dataclass
class StateResponse:
    step: int = 0
    R_global: float = 0.0
    regime: str = ""
    amplitude_mode: bool = False
    mean_amplitude: float = 0.0
    layers: list[LayerState] = field(default_factory=list)


@dataclass
class ConfigResponse:
    name: str = ""
    n_oscillators: int = 0
    n_layers: int = 0
    amplitude_mode: bool = False
    sample_period_s: float = 0.0
    control_period_s: float = 0.0
