# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — gRPC PhaseOrchestrator service

"""gRPC service implementing the full PhaseOrchestrator API.

Install grpcio to run a live server::

    pip install grpcio grpcio-tools

Without grpcio the servicer still works for in-process testing
with a mocked context.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

from scpn_phase_orchestrator.grpc_gen import (
    ConfigResponse,
    LayerState,
    PhaseOrchestratorServicer,
    StateResponse,
)
from scpn_phase_orchestrator.server import SimulationState

__all__ = ["PhaseStreamServicer", "HAS_GRPC"]

try:
    import grpc  # pragma: no cover

    HAS_GRPC = True  # pragma: no cover
except ModuleNotFoundError:
    grpc = None
    HAS_GRPC = False


def _snap_to_response(snap: dict) -> StateResponse:
    """Convert a SimulationState.snapshot() dict to a StateResponse."""
    layers = [
        LayerState(
            name=ly.get("name", ""),
            R=ly.get("R", 0.0),
            psi=ly.get("psi", 0.0),
        )
        for ly in snap.get("layers", [])
    ]
    return StateResponse(
        step=snap.get("step", 0),
        R_global=snap.get("R_global", 0.0),
        regime=snap.get("regime", ""),
        amplitude_mode=snap.get("amplitude_mode", False),
        mean_amplitude=snap.get("mean_amplitude", 0.0),
        layers=layers,
    )


class PhaseStreamServicer(PhaseOrchestratorServicer):
    """gRPC servicer that exposes state, step, reset, streaming, and config.

    Wraps a ``SimulationState`` (from server.py) and translates its
    dict snapshots into proto-compatible ``StateResponse`` messages.
    """

    def __init__(self, sim: SimulationState) -> None:
        self._sim = sim
        self._lock = threading.Lock()

    # -- unary RPCs -----------------------------------------------------------

    def GetState(self, request: Any, context: Any) -> StateResponse:
        with self._lock:
            return _snap_to_response(self._sim.snapshot())

    def Step(self, request: Any, context: Any) -> StateResponse:
        n = getattr(request, "n_steps", 1) or 1
        with self._lock:
            for _ in range(n):
                self._sim.step()
            return _snap_to_response(self._sim.snapshot())

    def Reset(self, request: Any, context: Any) -> StateResponse:
        with self._lock:
            self._sim.reset()
            return _snap_to_response(self._sim.snapshot())

    def GetConfig(self, request: Any, context: Any) -> ConfigResponse:
        spec = self._sim.spec
        return ConfigResponse(
            name=spec.name,
            n_oscillators=self._sim.n_osc,
            n_layers=len(spec.layers),
            amplitude_mode=self._sim.amplitude_mode,
            sample_period_s=spec.sample_period_s,
            control_period_s=spec.control_period_s,
        )

    # -- server-streaming RPC -------------------------------------------------

    def StreamPhases(self, request: Any, context: Any) -> Iterator[StateResponse]:
        """Read-only observer: streams snapshots without advancing simulation."""
        max_steps = getattr(request, "max_steps", 100) or 100
        interval = getattr(request, "interval_s", 0.05) or 0.05
        for _ in range(max_steps):
            if (
                context is not None
                and hasattr(context, "is_active")
                and not context.is_active()
            ):
                return
            with self._lock:
                snap = self._sim.snapshot()
            yield _snap_to_response(snap)
            time.sleep(interval)
