# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hand-written gRPC servicer stubs
#
# Mirrors proto/spo.proto. Re-generate real stubs with:
#     bash tools/generate_grpc.sh

from __future__ import annotations

import abc
from collections.abc import Iterator
from typing import Any

from scpn_phase_orchestrator.grpc_gen._spo_pb2_fallback import (
    ConfigRequest,
    ConfigResponse,
    StateRequest,
    StateResponse,
    StepRequest,
    StreamRequest,
)

__all__ = [
    "PhaseOrchestratorServicer",
    "add_PhaseOrchestratorServicer_to_server",
]


class PhaseOrchestratorServicer(abc.ABC):
    """Abstract servicer matching the PhaseOrchestrator service in spo.proto."""

    @abc.abstractmethod
    def GetState(self, request: StateRequest, context: Any) -> StateResponse:
        raise NotImplementedError

    @abc.abstractmethod
    def Step(self, request: StepRequest, context: Any) -> StateResponse:
        raise NotImplementedError

    @abc.abstractmethod
    def Reset(self, request: Any, context: Any) -> StateResponse:
        raise NotImplementedError

    @abc.abstractmethod
    def StreamPhases(
        self, request: StreamRequest, context: Any
    ) -> Iterator[StateResponse]:
        raise NotImplementedError

    @abc.abstractmethod
    def GetConfig(self, request: ConfigRequest, context: Any) -> ConfigResponse:
        raise NotImplementedError


def add_PhaseOrchestratorServicer_to_server(
    servicer: PhaseOrchestratorServicer, server: Any
) -> None:
    """Register *servicer* on a grpc.Server.

    With real generated stubs this binds the service descriptors.
    This fallback version requires grpcio at runtime.
    """
    try:
        import grpc  # type: ignore[import-untyped]
    except ImportError as exc:
        msg = "grpcio required to register servicer on a live server"
        raise ImportError(msg) from exc

    from grpc import unary_stream_rpc_method_handler, unary_unary_rpc_method_handler

    method_handlers = {
        "GetState": unary_unary_rpc_method_handler(servicer.GetState),
        "Step": unary_unary_rpc_method_handler(servicer.Step),
        "Reset": unary_unary_rpc_method_handler(servicer.Reset),
        "StreamPhases": unary_stream_rpc_method_handler(servicer.StreamPhases),
        "GetConfig": unary_unary_rpc_method_handler(servicer.GetConfig),
    }
    generic_handler = grpc.method_service_handler(
        "spo.PhaseOrchestrator", method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))
