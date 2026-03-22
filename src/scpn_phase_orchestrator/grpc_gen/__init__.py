# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Generated gRPC stubs
#
# Files in this package are generated from proto/spo.proto.
# Re-generate with: bash tools/generate_grpc.sh
# Hand-written fallback stubs are provided for environments
# where grpc_tools is not installed.

from scpn_phase_orchestrator.grpc_gen.spo_pb2 import (
    ConfigRequest,
    ConfigResponse,
    LayerState,
    ResetRequest,
    StateRequest,
    StateResponse,
    StepRequest,
    StreamRequest,
)
from scpn_phase_orchestrator.grpc_gen.spo_pb2_grpc import (
    PhaseOrchestratorServicer,
    add_PhaseOrchestratorServicer_to_server,
)

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
]
