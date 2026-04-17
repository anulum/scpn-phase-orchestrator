# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Adapter registry

from __future__ import annotations

from scpn_phase_orchestrator.adapters.fusion_core_bridge import FusionCoreBridge
from scpn_phase_orchestrator.adapters.gaian_mesh_bridge import GaianMeshNode
from scpn_phase_orchestrator.adapters.hardware_io import (
    HAS_BRAINFLOW,
    HAS_MODBUS,
    BrainFlowAdapter,
    ModbusAdapter,
    SimulatedBoardAdapter,
)
from scpn_phase_orchestrator.adapters.lsl_bci_bridge import LSLBCIBridge
from scpn_phase_orchestrator.adapters.metrics_exporter import MetricsExporter
from scpn_phase_orchestrator.adapters.modbus_tls import (
    HAS_PYMODBUS,
    SecureModbusAdapter,
)
from scpn_phase_orchestrator.adapters.neurocore_bridge import NeurocoreBridge
from scpn_phase_orchestrator.adapters.opentelemetry import OTelExporter
from scpn_phase_orchestrator.adapters.plasma_control_bridge import PlasmaControlBridge
from scpn_phase_orchestrator.adapters.prometheus import PrometheusAdapter
from scpn_phase_orchestrator.adapters.quantum_control_bridge import QuantumControlBridge
from scpn_phase_orchestrator.adapters.redis_store import RedisStateStore
from scpn_phase_orchestrator.adapters.remanentia_bridge import (
    CoherenceMemorySnapshot,
    RemanentiaBridge,
)
from scpn_phase_orchestrator.adapters.scpn_control_bridge import SCPNControlBridge
from scpn_phase_orchestrator.adapters.snn_bridge import SNNControllerBridge
from scpn_phase_orchestrator.adapters.synapse_channel_bridge import (
    AgentState,
    SynapseChannelBridge,
)
from scpn_phase_orchestrator.adapters.synapse_coupling_bridge import (
    SynapseCouplingBridge,
    SynapseSnapshot,
)

__all__ = [
    "AgentState",
    "BrainFlowAdapter",
    "CoherenceMemorySnapshot",
    "FusionCoreBridge",
    "GaianMeshNode",
    "HAS_BRAINFLOW",
    "HAS_MODBUS",
    "HAS_PYMODBUS",
    "LSLBCIBridge",
    "MetricsExporter",
    "ModbusAdapter",
    "NeurocoreBridge",
    "OTelExporter",
    "PlasmaControlBridge",
    "PrometheusAdapter",
    "QuantumControlBridge",
    "RedisStateStore",
    "RemanentiaBridge",
    "SCPNControlBridge",
    "SNNControllerBridge",
    "SecureModbusAdapter",
    "SimulatedBoardAdapter",
    "SynapseChannelBridge",
    "SynapseCouplingBridge",
    "SynapseSnapshot",
]
