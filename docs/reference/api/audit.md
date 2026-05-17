<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — audit API reference -->

# Audit

SHA256-chained audit logging, protobuf event streaming, and deterministic
replay for regulatory compliance, debugging, and formal verification. Every
supervisor decision, regime transition, and actuation command can be recorded
with tamper-evident JSONL records and a parallel event-sourced protobuf stream.

## Motivation

SPO is designed for safety-critical applications (power grids, plasma
control, medical devices). These domains require:

1. **Traceability** — every control decision must be attributable to
   a specific input state and policy rule
2. **Tamper evidence** — audit records must detect insertion, deletion,
   or modification after the fact
3. **Reproducibility** — given the same inputs and code version, the
   system must produce identical outputs

The audit subsystem provides all three via hash-chained logging,
event-sourced streaming, and deterministic replay.

## Hash Chain Structure

Each audit record contains:

| Field | Description |
|-------|-------------|
| `step` | Monotonic step counter |
| `timestamp` | ISO 8601 UTC timestamp |
| `event_type` | `regime_transition`, `actuation`, `boundary_breach`, etc. |
| `payload` | Event-specific data (JSON-serialisable) |
| `prev_hash` | SHA256 of the previous record |
| `hash` | SHA256 of this record (step + event_type + payload + prev_hash) |

The first record uses `prev_hash = "0" * 64`. To verify the chain,
recompute each hash and check it matches the stored value and the
next record's `prev_hash`.

The `spo run --audit` header includes the resolved binding summary under
`binding_config` and `binding_summary`. For N-channel domainpacks this summary
includes `channel_algebra`, covering required and optional channels, derived
channels, runtime evidence channels, group membership, coupling participants,
and missing required channel evidence.

## Keyed Audit Signatures

Set `SPO_AUDIT_KEY` to enable HMAC-SHA256 signatures on JSONL audit records.
Signed records add audit metadata for replay verification:

| Field | Description |
|-------|-------------|
| `_audit_schema_version` | Signature metadata schema version |
| `_audit_stream_id` | Logical JSONL stream id |
| `_audit_sequence` | Monotonic record sequence |
| `_audit_timestamp_unix_ns` | Signing timestamp in Unix nanoseconds |
| `_previous_hash` | Previous JSONL record hash used by the signature |
| `_payload_hash` | SHA256 of the canonical payload without audit metadata |
| `_signature` | HMAC algorithm, key id, and signature value |

The raw key is never written to the audit file. The stored key id is
`sha256(key)[:16]`, which lets replay choose the right verification key without
logging secret material.

When `SPO_AUDIT_KEY` is configured, `ReplayEngine.verify_integrity()` and
`spo replay --verify` fail closed if a record is unsigned, malformed, signed by
an unknown key, or modified after signing. Without `SPO_AUDIT_KEY`, legacy
unsigned development logs remain readable and hash-chain verification keeps its
previous behaviour.

For key rotation, keep historical keys only in the operator environment and pass
them as a JSON object through `SPO_AUDIT_KEYRING`:

```bash
export SPO_AUDIT_KEY="new secret"
export SPO_AUDIT_KEYRING='{
  "old_key_id": "old secret",
  "new_key_id": "new secret"
}'
spo replay audit.jsonl --verify
```

Each keyring object key must match `sha256(secret)[:16]`; mismatches fail
closed. Do not commit these environment values, include them in diagnostics, or
store them in audit artefacts.

## Audit Logger

Appends timestamped, SHA256-chained records to a JSONL audit trail. When
`event_stream` is supplied, the same stored records are also appended to a
length-delimited protobuf stream.

```python
from scpn_phase_orchestrator.audit.logger import AuditLogger

logger = AuditLogger("audit.jsonl", event_stream="audit.spoa")
logger.log_event("regime_transition", {"from": "nominal", "to": "degraded"})
logger.close()
```

::: scpn_phase_orchestrator.audit.logger

## Deterministic Replay

Replays an audit trail against a fresh SPO instance to reproduce
the exact sequence of states. Replay verifies that the same inputs
produce the same outputs — detecting non-determinism, floating-point
platform differences, or code regressions.

**Replay guarantees:**

- Same binding spec + same audit trail → identical phase trajectories
- Any divergence is flagged with the step number and magnitude
- Platform-specific float differences (x86 vs ARM extended precision)
  are handled via configurable tolerance

```python
from scpn_phase_orchestrator.audit.replay import ReplayEngine

engine = ReplayEngine("audit.jsonl", binding_spec)
divergences = engine.replay()
if divergences:
    for d in divergences:
        print(f"Step {d.step}: Δ={d.magnitude:.2e}")
```

::: scpn_phase_orchestrator.audit.replay

## Protobuf Event Stream

`scpn_phase_orchestrator.audit.stream` provides the event-sourced stream layer.
The schema is tracked in `proto/audit.proto` and packaged as
`scpn_phase_orchestrator/audit/audit.proto`.

```python
from scpn_phase_orchestrator.audit.stream import (
    read_event_stream,
    verify_event_stream_integrity,
)

events = read_event_stream("audit.spoa")
ok, verified = verify_event_stream_integrity(events)
```

The stream is not a replacement for deterministic replay; it is the live
transport for the same audit records. The JSONL file remains the compatibility
format for existing reports and replay tooling.

::: scpn_phase_orchestrator.audit.stream

## Pipeline integration

The audit logger sits at the output of the supervisor loop:

```
SupervisorPolicy.decide() ──→ list[ControlAction]
                                      │
                               ┌──────┼──────┐
                               ↓      ↓      ↓
                          Actuator  Audit   EventBus
                                   Logger
                                     │
                              audit.jsonl (append)
                              audit.spoa (append)
                                     │
                              SHA-256 chain
```

Every regime transition, actuation command, and boundary violation
is recorded. The audit trail is the authoritative record of what
the system did and why.

## AuditLogger API

```python
AuditLogger(log_path: str | Path, *, event_stream: str | Path | None = None)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `log_header` | `(n_oscillators, dt, method, seed, amplitude_mode)` | Engine config record |
| `log_step` | `(step, upde_state, actions, *, phases, omegas, knm, alpha, zeta, psi_drive, amplitudes, mu, knm_r, epsilon)` | Full simulation step |
| `log_event` | `(event_type: str, data: dict)` | Named event with arbitrary data |
| `close` | `()` | Flush and close file handle |

Supports context manager (`with AuditLogger(...) as logger:`).

`log_step` optionally records full engine state (phases, omegas, knm, alpha)
for deterministic replay. When `phases` is provided, `omegas`, `knm`, and
`alpha` are required (raises `AuditError` otherwise). Stuart-Landau fields
(`amplitudes`, `mu`, `knm_r`, `epsilon`) are optional.

## ReplayEngine API

```python
ReplayEngine(log_path: str | Path)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `load` | `() → list[dict]` | Parse all JSONL entries |
| `load_header` | `(entries) → dict \| None` | Extract engine config |
| `step_entries` | `(entries) → list[dict]` | Filter to replayable steps |
| `build_engine` | `(header) → UPDEEngine \| StuartLandauEngine` | Reconstruct engine from header |
| `verify_integrity` | `(entries) → (bool, int)` | Verify SHA-256 chain |
| `verify_determinism` | `(engine, steps) → bool` | Compare replayed R to logged R |
| `verify_determinism_chained` | `(engine, entries, atol) → (bool, int)` | Multi-step replay: output N = input N+1 |
| `verify_determinism_sl_chained` | `(engine, entries, atol) → (bool, int)` | Stuart-Landau chained replay |

### Hash chain verification algorithm

```python
prev_hash = "0" * 64  # genesis hash
for record in records:
    stored_hash = record.pop("_hash")
    content = json.dumps(record, separators=(",", ":"))
    expected = sha256((prev_hash + content).encode()).hexdigest()
    assert stored_hash == expected  # tamper detection
    prev_hash = stored_hash
```

## Compliance references

- NIST SP 800-92: Guide to Computer Security Log Management
- IEC 62443: Industrial communication networks security
- ISO 27001 A.12.4: Logging and monitoring
