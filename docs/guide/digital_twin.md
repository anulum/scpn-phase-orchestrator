# Digital Twin Pattern

SPO fits naturally into digital twin architectures as the **coherence
control layer** — the component that monitors synchronisation health
and intervenes when the physical system drifts.

## Architecture

```
Physical System ──► Sensors ──► Phase Extraction (P/I/S)
                                      │
                                      ▼
                              ┌── SPO Engine ──┐
                              │  UPDE dynamics  │
                              │  15+ monitors   │
                              │  Regime FSM     │
                              │  Supervisor     │
                              └───────┬─────────┘
                                      │
                              ┌───────┴─────────┐
                              │   Actuation      │
                              │   (K, α, ζ, Ψ)  │
                              └───────┬─────────┘
                                      │
                                      ▼
Physical System ◄── Actuators ◄── Control Commands
```

The digital twin runs SPO in parallel with the physical system.
Sensor data feeds the oscillator extractors; control commands flow
back through the actuation mapper.

## Binding Contract

Before a simulator, service twin, or hardware twin exchanges data with SPO,
export the binding as a versioned contract:

```python
from scpn_phase_orchestrator.binding import (
    build_digital_twin_binding_contract,
    load_binding_spec,
)

spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
contract = build_digital_twin_binding_contract(spec)
contract_payload = contract.to_audit_record()
```

The contract records binding name/version, timing, layer oscillator IDs,
actuator limits, N-channel algebra, default sync capabilities, and a stable
`contract_hash`. It is a compatibility artefact: consumers can compare hashes
and capability names before accepting telemetry or proposed actions. It does
not open a network connection or apply control.

Adapters should validate every transport payload against that contract before
runtime use:

```python
from scpn_phase_orchestrator.binding import (
    build_digital_twin_sync_envelope,
    validate_digital_twin_sync_envelope,
)

envelope = build_digital_twin_sync_envelope(
    contract,
    capability="state_snapshot",
    direction="twin_to_spo",
    sequence=1,
    payload={"layer": "machine_cells", "R": 0.91},
)
validation = validate_digital_twin_sync_envelope(contract, envelope)
if not validation.accepted:
    raise ValueError(validation.reason)
```

This gives REST, gRPC, Kafka, file, and hardware adapters the same compatibility
gate without coupling the binding layer to any specific transport.

For deterministic file replay, use the JSONL adapter:

```python
from scpn_phase_orchestrator.binding import (
    read_digital_twin_sync_jsonl,
    write_digital_twin_sync_jsonl,
)

write_digital_twin_sync_jsonl("sync.jsonl", [envelope])
replay_report = read_digital_twin_sync_jsonl(contract, "sync.jsonl")
```

The replay report separates accepted validations from malformed JSON, invalid
envelope shapes, and contract-validation rejections. This is useful for adapter
smoke tests and audit replay before introducing a live network transport.

The same accepted/rejected validation records can be reduced to a stable
operator evidence payload:

```python
from scpn_phase_orchestrator.binding import build_digital_twin_operator_evidence

evidence = build_digital_twin_operator_evidence(
    contract,
    replay_report.accepted,
    rejected=replay_report.rejected,
)
```

The evidence record reports accepted/rejected counts, capability and direction
counts, latest sequence, maximum absolute twin residual, mismatch reasons,
adapter health, and an operator status (`healthy`, `warning`, `degraded`, or
`critical`). Live REST/gRPC/Kafka/hardware adapters and replayed JSONL files
therefore expose the same dashboard fields.

For runtime-facing tests that should not touch disk, use the in-memory adapter:

```python
from scpn_phase_orchestrator.binding import DigitalTwinSyncMemoryAdapter

adapter = DigitalTwinSyncMemoryAdapter.for_contract(contract)
validation = adapter.submit(envelope)
accepted_batch = adapter.drain()
```

The adapter queues accepted envelopes in order and returns rejection reasons
for invalid submissions. For HTTP integrations, use the dependency-free REST
boundary adapter inside a framework route:

```python
from scpn_phase_orchestrator.binding import DigitalTwinSyncRestAdapter

adapter = DigitalTwinSyncRestAdapter.for_contract(
    contract,
    sync_capabilities=("state_snapshot", "audit_replay"),
)
response = adapter.handle_post(
    envelope.to_audit_record(),
    headers={"authorization": "Bearer ..."},
)
accepted_batch = adapter.drain()
```

This helper does not start a web server. It enforces the adapter manifest,
authentication posture, JSON envelope shape, contract hash, declared capability,
direction, and non-empty payload before a FastAPI, Flask, or gateway endpoint
hands data to SPO runtime code.

For gRPC integrations, use the same boundary pattern after decoding protobuf
fields inside a servicer:

```python
from scpn_phase_orchestrator.binding import DigitalTwinSyncGrpcAdapter

adapter = DigitalTwinSyncGrpcAdapter.for_contract(contract)
response = adapter.handle_unary(
    envelope.to_audit_record(),
    metadata={"authorization": "Bearer ..."},
)
accepted_batch = adapter.drain()
```

The gRPC helper returns status names such as `OK`, `UNAUTHENTICATED`,
`INVALID_ARGUMENT`, and `FAILED_PRECONDITION` so a real servicer can map the
result to framework-native status handling without duplicating contract logic.

For Kafka or compatible message buses, pass decoded message records through the
broker-neutral boundary:

```python
from scpn_phase_orchestrator.binding import DigitalTwinSyncKafkaAdapter

adapter = DigitalTwinSyncKafkaAdapter.for_contract(
    contract,
    topic="spo.digital_twin.sync",
)
response = adapter.handle_message(
    {"topic": "spo.digital_twin.sync", "value": envelope.to_audit_record()},
    headers={"authorization": "Bearer ..."},
)
accepted_batch = adapter.drain()
```

The Kafka helper does not import a broker client, commit offsets, or open
network connections. It checks the topic, auth header, decoded value shape,
contract, direction, and payload before caller-controlled offset handling.

For hardware integrations, validate decoded device frames before any driver or
actuator layer can see them:

```python
from scpn_phase_orchestrator.binding import DigitalTwinSyncHardwareAdapter

adapter = DigitalTwinSyncHardwareAdapter.for_contract(
    contract,
    device_ids=("pynq-loopback-0",),
)
response = adapter.handle_frame(
    {
        "device_id": "pynq-loopback-0",
        "safety_interlock": True,
        "value": envelope.to_audit_record(),
    },
    headers={"authorization": "Bearer ..."},
)
accepted_batch = adapter.drain()
```

The hardware helper never opens device files, writes registers, toggles GPIO, or
applies actuation. It checks the registered device ID, explicit safety
interlock, auth header, decoded frame value, contract, direction, and payload;
`hardware_write_permitted` is always `False` in its response and audit record.

Before enabling a concrete adapter, publish a reviewable manifest:

```python
from scpn_phase_orchestrator.binding import build_digital_twin_adapter_manifest

compatibility = build_digital_twin_adapter_manifest(
    contract,
    name="grpc-live",
    transport="grpc",
    sync_capabilities=("state_snapshot", "audit_replay"),
    supports_replay=True,
    requires_auth=True,
)
```

The manifest check rejects undeclared capabilities, live transports without
authentication, and offline transports that cannot replay payloads.

## Implementation

```python
from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.runtime.server import SimulationState

# 1. Load domain binding
spec = load_binding_spec("domainpacks/power_grid/binding_spec.yaml")
twin = SimulationState(spec)

# 2. Feed sensor data each tick
while True:
    sensor_data = read_sensors()       # your data source
    state = twin.step()                # advance twin
    R = state["R_global"]
    regime = state["regime"]

    # 3. Act on supervisor decisions
    if regime == "critical":
        increase_coupling()            # your actuator
    elif regime == "degraded":
        log_warning(R)

    # 4. Audit trail for compliance
    audit_logger.log_step(state)
```

## Deployment Options

| Mode | Latency | Use case |
|------|---------|----------|
| **Embedded** (Python library) | <1ms/step | Edge devices, PLCs |
| **REST API** (FastAPI server) | ~10ms | Cloud dashboards |
| **gRPC streaming** | ~5ms | High-frequency telemetry |
| **WASM** (browser) | ~1ms | Monitoring dashboards |
| **Docker** (full stack) | N/A | Production deployment |

## Real-World Examples

- **Tokamak plasma**: MHD mode coupling → SPO detects mode locking precursors
- **Power grid**: generator swing equations → SPO predicts cascade failures
- **Manufacturing**: SPC quality oscillations → SPO tunes process parameters
- **Traffic**: signal timing drift → SPO maintains green wave coherence

Each maps onto the same pattern: extract phases → run engine → monitor → act.
