# Audit

SHA256-chained audit logging and deterministic replay for regulatory
compliance, debugging, and formal verification. Every supervisor
decision, regime transition, and actuation command is recorded with
a cryptographic hash chain that detects post-hoc tampering.

## Motivation

SPO is designed for safety-critical applications (power grids, plasma
control, medical devices). These domains require:

1. **Traceability** — every control decision must be attributable to
   a specific input state and policy rule
2. **Tamper evidence** — audit records must detect insertion, deletion,
   or modification after the fact
3. **Reproducibility** — given the same inputs and code version, the
   system must produce identical outputs

The audit subsystem provides all three via hash-chained logging and
deterministic replay.

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

Design follows NIST SP 800-92 (Guide to Computer Security Log Management)
adapted for real-time control systems.

## Audit Logger

Appends timestamped, SHA256-chained records to a JSONL audit trail.
Thread-safe (uses a lock for concurrent writes). Flushes after each
record to prevent data loss on crash.

```python
from scpn_phase_orchestrator.audit.logger import AuditLogger

logger = AuditLogger("audit.jsonl")
logger.log("regime_transition", {"from": "nominal", "to": "degraded", "R": 0.55})
logger.log("actuation", {"knob": "K", "value": 0.3, "ttl_s": 5.0})

# Verify chain integrity
assert logger.verify_chain()
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
                                     │
                              SHA-256 chain
```

Every regime transition, actuation command, and boundary violation
is recorded. The audit trail is the authoritative record of what
the system did and why.

## AuditLogger API

```python
AuditLogger(log_path: str | Path)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `log` | `(event_type: str, payload: dict)` | Append record |
| `verify_chain` | `() → bool` | Validate hash integrity |
| `close` | `()` | Flush and close file handle |

### Thread safety

AuditLogger uses a threading.Lock for all write operations.
Multiple threads can call `log()` concurrently without corruption.

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
