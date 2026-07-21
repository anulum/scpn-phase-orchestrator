# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Adapter registry

"""Public adapter registry for optional external-system bridge surfaces.

The adapters package re-exports bridges for fusion, plasma, quantum, SNN,
Prometheus, Redis, hardware I/O, NeuroCore, Remanentia, Synapse, and related
integration points. Runtime metrics and OpenTelemetry exporters live in
``scpn_phase_orchestrator.runtime.observability`` so the integration boundary
does not depend on runtime serving code.
"""

from __future__ import annotations

from scpn_phase_orchestrator.adapters.fmi_cosimulation import (
    CoSimulationSlave,
    FMIVariable,
    cosimulate,
    generate_model_description,
    write_fmu,
)
from scpn_phase_orchestrator.adapters.fusion_core_bridge import FusionCoreBridge
from scpn_phase_orchestrator.adapters.gaian_mesh_bridge import GaianMeshNode
from scpn_phase_orchestrator.adapters.hardware_io import (
    HAS_BRAINFLOW,
    HAS_MODBUS,
    BrainFlowAdapter,
    ModbusAdapter,
    SampleBuffer,
    SimulatedBoardAdapter,
)
from scpn_phase_orchestrator.adapters.hybrid_cocompiler import (
    audit_hybrid_target_readiness,
    build_hybrid_cocompiler_manifest,
    build_hybrid_operator_handoff_package,
)
from scpn_phase_orchestrator.adapters.lsl_bci_bridge import LSLBCIBridge
from scpn_phase_orchestrator.adapters.modbus_tls import (
    HAS_PYMODBUS,
    SecureModbusAdapter,
)
from scpn_phase_orchestrator.adapters.mqtt_bridge import (
    HAS_PAHO_MQTT,
    MqttBridgeConfig,
    MqttPhaseBridge,
    MqttTag,
)
from scpn_phase_orchestrator.adapters.neurocore_bridge import NeurocoreBridge
from scpn_phase_orchestrator.adapters.neuromorphic_ir_export import (
    NeuromorphicIRGraph,
    to_nir_graph,
)
from scpn_phase_orchestrator.adapters.opcua_bridge import (
    HAS_ASYNCUA,
    OpcUaBridgeConfig,
    OpcUaPhaseBridge,
    OpcUaTag,
)
from scpn_phase_orchestrator.adapters.openqasm_conformance import (
    OpenQasm3ConformanceReport,
    check_openqasm3,
)
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
from scpn_phase_orchestrator.adapters.synchrophasor_c37118 import (
    ConfigurationFrame2,
    DataFrame,
    FrameChecksumError,
    FrameTruncationError,
    PhasorUnit,
    PmuConfiguration,
    PmuMeasurement,
    SynchrophasorFrameCodec,
    SynchrophasorFrameError,
    SynchrophasorHeader,
    UnsupportedFrameError,
    compute_crc_ccitt,
    data_frames_to_frequency_series,
)
from scpn_phase_orchestrator.adapters.synchrophasor_client import (
    COMMAND_DATA_OFF,
    COMMAND_DATA_ON,
    COMMAND_SEND_CONFIG1,
    COMMAND_SEND_CONFIG2,
    COMMAND_SEND_CONFIG3,
    COMMAND_SEND_HEADER,
    C37118SessionClient,
    build_command_frame,
    read_frame,
)
from scpn_phase_orchestrator.adapters.synchrophasor_phase_bridge import (
    C37118PhaseBridge,
    PhasorBinding,
)

__all__ = [
    "AgentState",
    "BrainFlowAdapter",
    "C37118PhaseBridge",
    "C37118SessionClient",
    "COMMAND_DATA_OFF",
    "COMMAND_DATA_ON",
    "COMMAND_SEND_CONFIG1",
    "COMMAND_SEND_CONFIG2",
    "COMMAND_SEND_CONFIG3",
    "COMMAND_SEND_HEADER",
    "CoSimulationSlave",
    "CoherenceMemorySnapshot",
    "ConfigurationFrame2",
    "DataFrame",
    "FMIVariable",
    "FrameChecksumError",
    "FrameTruncationError",
    "FusionCoreBridge",
    "GaianMeshNode",
    "HAS_ASYNCUA",
    "HAS_BRAINFLOW",
    "HAS_MODBUS",
    "HAS_PYMODBUS",
    "HAS_PAHO_MQTT",
    "LSLBCIBridge",
    "ModbusAdapter",
    "MqttBridgeConfig",
    "MqttPhaseBridge",
    "MqttTag",
    "NeurocoreBridge",
    "NeuromorphicIRGraph",
    "OpcUaBridgeConfig",
    "OpcUaPhaseBridge",
    "OpcUaTag",
    "OpenQasm3ConformanceReport",
    "PhasorBinding",
    "PhasorUnit",
    "PlasmaControlBridge",
    "PmuConfiguration",
    "PmuMeasurement",
    "PrometheusAdapter",
    "QuantumControlBridge",
    "RedisStateStore",
    "RemanentiaBridge",
    "SCPNControlBridge",
    "SampleBuffer",
    "SNNControllerBridge",
    "SecureModbusAdapter",
    "SimulatedBoardAdapter",
    "SynapseChannelBridge",
    "SynapseCouplingBridge",
    "SynapseSnapshot",
    "SynchrophasorFrameCodec",
    "SynchrophasorFrameError",
    "SynchrophasorHeader",
    "UnsupportedFrameError",
    "audit_hybrid_target_readiness",
    "build_command_frame",
    "build_hybrid_cocompiler_manifest",
    "build_hybrid_operator_handoff_package",
    "check_openqasm3",
    "compute_crc_ccitt",
    "cosimulate",
    "data_frames_to_frequency_series",
    "generate_model_description",
    "read_frame",
    "to_nir_graph",
    "write_fmu",
]
