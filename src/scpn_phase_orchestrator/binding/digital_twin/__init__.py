# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Digital-twin binding contract

"""Transport-neutral digital-twin contracts derived from bindings.

This package turns a validated `BindingSpec` into deterministic contract hashes,
adapter manifests, sync capabilities, and envelope validation records for
simulators, services, and hardware twins, split into responsibility modules
(contract, envelope, evidence, and per-transport adapters) behind a stable
re-export surface. REST, gRPC, Kafka, JSONL, hardware, and in-memory helpers
validate decoded payloads only; they do not open sockets, spawn servers, or
apply live control actions.
"""

from __future__ import annotations

from .adapter_grpc import (
    DigitalTwinSyncGrpcAdapter,
    DigitalTwinSyncGrpcResponse,
)
from .adapter_hardware import (
    DigitalTwinSyncHardwareAdapter,
    DigitalTwinSyncHardwareResponse,
)
from .adapter_kafka import (
    DigitalTwinSyncKafkaAdapter,
    DigitalTwinSyncKafkaResponse,
)
from .adapter_memory import DigitalTwinSyncMemoryAdapter
from .adapter_rest import (
    DigitalTwinSyncRestAdapter,
    DigitalTwinSyncRestResponse,
)
from .contract import (
    DigitalTwinAdapterCompatibility,
    DigitalTwinAdapterManifest,
    DigitalTwinBindingContract,
    DigitalTwinLayerContract,
    DigitalTwinSyncCapability,
    build_digital_twin_adapter_manifest,
    build_digital_twin_binding_contract,
)
from .envelope import (
    DigitalTwinSyncEnvelope,
    DigitalTwinSyncJsonlReport,
    DigitalTwinTransportValidation,
    build_digital_twin_sync_envelope,
    read_digital_twin_sync_jsonl,
    validate_digital_twin_sync_envelope,
    write_digital_twin_sync_jsonl,
)
from .evidence import (
    DigitalTwinOperatorEvidence,
    build_digital_twin_operator_evidence,
)

__all__ = [
    "DigitalTwinBindingContract",
    "DigitalTwinAdapterManifest",
    "DigitalTwinAdapterCompatibility",
    "DigitalTwinLayerContract",
    "DigitalTwinSyncCapability",
    "DigitalTwinSyncEnvelope",
    "DigitalTwinSyncGrpcAdapter",
    "DigitalTwinSyncGrpcResponse",
    "DigitalTwinSyncHardwareAdapter",
    "DigitalTwinSyncHardwareResponse",
    "DigitalTwinSyncJsonlReport",
    "DigitalTwinSyncKafkaAdapter",
    "DigitalTwinSyncKafkaResponse",
    "DigitalTwinSyncMemoryAdapter",
    "DigitalTwinOperatorEvidence",
    "DigitalTwinSyncRestAdapter",
    "DigitalTwinSyncRestResponse",
    "DigitalTwinTransportValidation",
    "build_digital_twin_adapter_manifest",
    "build_digital_twin_binding_contract",
    "build_digital_twin_operator_evidence",
    "build_digital_twin_sync_envelope",
    "read_digital_twin_sync_jsonl",
    "validate_digital_twin_sync_envelope",
    "write_digital_twin_sync_jsonl",
]
