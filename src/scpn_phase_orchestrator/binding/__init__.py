# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Binding subsystem

from __future__ import annotations

from scpn_phase_orchestrator.binding.channel_algebra import (
    ChannelAlgebraReport,
    ChannelCouplingEdge,
    ChannelRuntimePolicy,
    build_channel_algebra_report,
)
from scpn_phase_orchestrator.binding.channel_runtime import (
    ChannelLayerRuntimeEvidence,
    ChannelRuntimeExecution,
    ChannelRuntimeExecutor,
)
from scpn_phase_orchestrator.binding.digital_twin import (
    DigitalTwinBindingContract,
    DigitalTwinLayerContract,
    DigitalTwinSyncCapability,
    DigitalTwinSyncEnvelope,
    DigitalTwinSyncJsonlReport,
    DigitalTwinSyncMemoryAdapter,
    DigitalTwinTransportValidation,
    build_digital_twin_binding_contract,
    build_digital_twin_sync_envelope,
    read_digital_twin_sync_jsonl,
    validate_digital_twin_sync_envelope,
    write_digital_twin_sync_jsonl,
)
from scpn_phase_orchestrator.binding.loader import BindingLoadError, load_binding_spec
from scpn_phase_orchestrator.binding.resolved import (
    format_resolved_binding_config,
    resolved_binding_config,
)
from scpn_phase_orchestrator.binding.semantic import (
    GeneratedBindingArtifacts,
    RetrievalEvidence,
    SemanticDomainCompiler,
    compile_symbolic_binding,
)
from scpn_phase_orchestrator.binding.types import BindingSpec
from scpn_phase_orchestrator.binding.validator import validate_binding_spec

__all__ = [
    "BindingLoadError",
    "BindingSpec",
    "ChannelAlgebraReport",
    "ChannelCouplingEdge",
    "ChannelLayerRuntimeEvidence",
    "ChannelRuntimeExecution",
    "ChannelRuntimeExecutor",
    "ChannelRuntimePolicy",
    "DigitalTwinBindingContract",
    "DigitalTwinLayerContract",
    "DigitalTwinSyncCapability",
    "DigitalTwinSyncEnvelope",
    "DigitalTwinSyncJsonlReport",
    "DigitalTwinSyncMemoryAdapter",
    "DigitalTwinTransportValidation",
    "GeneratedBindingArtifacts",
    "RetrievalEvidence",
    "SemanticDomainCompiler",
    "build_channel_algebra_report",
    "build_digital_twin_binding_contract",
    "build_digital_twin_sync_envelope",
    "compile_symbolic_binding",
    "format_resolved_binding_config",
    "load_binding_spec",
    "read_digital_twin_sync_jsonl",
    "resolved_binding_config",
    "validate_binding_spec",
    "validate_digital_twin_sync_envelope",
    "write_digital_twin_sync_jsonl",
]
