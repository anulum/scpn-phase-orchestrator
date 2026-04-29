# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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

import logging
import os
import time
from collections.abc import Iterator
from typing import Any

from scpn_phase_orchestrator.grpc_gen import (
    ConfigResponse,
    LayerState,
    PhaseOrchestratorServicer,
    StateResponse,
)
from scpn_phase_orchestrator.network_security import (
    FixedWindowRateLimiter,
    env_int,
    is_production_mode,
)
from scpn_phase_orchestrator.server import SimulationState

logger = logging.getLogger(__name__)

__all__ = ["PhaseStreamServicer", "HAS_GRPC"]

try:
    # type ignore: grpcio ships without complete typing in the supported range.
    import grpc  # type: ignore[import-untyped]  # pragma: no cover

    HAS_GRPC = True  # pragma: no cover
except ModuleNotFoundError:  # pragma: no cover
    grpc = None  # pragma: no cover
    HAS_GRPC = False  # pragma: no cover


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
        # Share the SimulationState mutex so gRPC threads and FastAPI async
        # handlers serialise against the same lock. A servicer-local lock
        # would leave the HTTP side free to race on shared engine state.
        self._lock = sim._lock
        self._api_key = os.environ.get("SPO_GRPC_API_KEY") or os.environ.get(
            "SPO_API_KEY"
        )
        self._production = is_production_mode("SPO_GRPC") or is_production_mode("SPO")
        if self._production and not self._api_key:
            raise RuntimeError(
                "SPO_GRPC_API_KEY or SPO_API_KEY is required in production"
            )
        rate_limit = env_int(
            "SPO_GRPC_RATE_LIMIT_PER_MINUTE", 120 if self._production else 0
        )
        self._limiter = FixedWindowRateLimiter(rate_limit) if rate_limit > 0 else None

    def _abort(self, context: Any, code: Any, detail: str) -> None:
        if context is not None and hasattr(context, "abort"):
            context.abort(code, detail)
        raise PermissionError(detail)

    def _authorise(self, context: Any) -> None:
        metadata = {}
        if context is not None and hasattr(context, "invocation_metadata"):
            metadata = dict(context.invocation_metadata())
        supplied = metadata.get("x-api-key")
        if self._api_key is not None and supplied != self._api_key:
            code = grpc.StatusCode.UNAUTHENTICATED if grpc is not None else None
            self._abort(context, code, "Invalid or missing x-api-key")
        identity = supplied or "anonymous"
        if self._limiter is not None and not self._limiter.allow(identity):
            code = grpc.StatusCode.RESOURCE_EXHAUSTED if grpc is not None else None
            self._abort(context, code, "Rate limit exceeded")

    # -- unary RPCs -----------------------------------------------------------

    def GetState(self, request: Any, context: Any) -> StateResponse:
        """gRPC unary RPC: return current simulation state."""
        self._authorise(context)
        with self._lock:
            return _snap_to_response(self._sim.snapshot())

    def Step(self, request: Any, context: Any) -> StateResponse:
        """gRPC unary RPC: advance simulation by n_steps and return state."""
        self._authorise(context)
        n = getattr(request, "n_steps", 1) or 1
        with self._lock:
            for _ in range(n):
                self._sim.step()
            response = _snap_to_response(self._sim.snapshot())
        logger.debug(
            "grpc.Step: n_steps=%d step=%d R_global=%.4f regime=%s",
            n,
            response.step,
            response.R_global,
            response.regime,
        )
        return response

    def Reset(self, request: Any, context: Any) -> StateResponse:
        """gRPC unary RPC: reset simulation and return fresh state."""
        self._authorise(context)
        with self._lock:
            self._sim.reset()
            response = _snap_to_response(self._sim.snapshot())
        logger.info("grpc.Reset: step=%d regime=%s", response.step, response.regime)
        return response

    def GetConfig(self, request: Any, context: Any) -> ConfigResponse:
        """gRPC unary RPC: return engine configuration."""
        self._authorise(context)
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
        self._authorise(context)
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
