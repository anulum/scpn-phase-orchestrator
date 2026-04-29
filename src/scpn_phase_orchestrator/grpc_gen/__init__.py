# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — gRPC stub loader
#
# Generated stubs live in spo_pb2.py / spo_pb2_grpc.py (from proto/spo.proto).
# Hand-written dataclass fallbacks in _spo_pb2_fallback.py / _spo_pb2_grpc_fallback.py.
#
# Import order:
#   1. Try generated protobuf messages (requires google.protobuf).
#   2. Fall back to hand-written dataclass stubs.
#   3. Try generated gRPC servicer (requires grpcio).
#   4. Fall back to hand-written ABC servicer.

from __future__ import annotations

USING_GENERATED_PB2: bool
USING_GENERATED_GRPC: bool

try:
    # type ignore: generated protobuf modules expose runtime attributes mypy cannot see.
    from scpn_phase_orchestrator.grpc_gen.spo_pb2 import (  # type: ignore[attr-defined]
        ConfigRequest,
        ConfigResponse,
        LayerState,
        ResetRequest,
        StateRequest,
        StateResponse,
        StepRequest,
        StreamRequest,
    )

    USING_GENERATED_PB2 = True
except Exception:  # pragma: no cover
    from scpn_phase_orchestrator.grpc_gen._spo_pb2_fallback import (
        ConfigRequest,
        ConfigResponse,
        LayerState,
        ResetRequest,
        StateRequest,
        StateResponse,
        StepRequest,
        StreamRequest,
    )

    USING_GENERATED_PB2 = False

try:
    from scpn_phase_orchestrator.grpc_gen.spo_pb2_grpc import (
        PhaseOrchestratorServicer,
        add_PhaseOrchestratorServicer_to_server,
    )

    USING_GENERATED_GRPC = True
except Exception:  # pragma: no cover
    # type ignore: fallback servicer intentionally substitutes generated grpc types.
    from scpn_phase_orchestrator.grpc_gen._spo_pb2_grpc_fallback import (  # type: ignore[assignment]
        PhaseOrchestratorServicer,
        add_PhaseOrchestratorServicer_to_server,
    )

    USING_GENERATED_GRPC = False

__all__ = [
    "ConfigRequest",
    "ConfigResponse",
    "LayerState",
    "ResetRequest",
    "StateRequest",
    "StateResponse",
    "StepRequest",
    "StreamRequest",
    "PhaseOrchestratorServicer",
    "add_PhaseOrchestratorServicer_to_server",
    "USING_GENERATED_PB2",
    "USING_GENERATED_GRPC",
]
