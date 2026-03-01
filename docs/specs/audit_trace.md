# Audit Trace

## Format

Append-only JSONL (one JSON object per line).

## Step Record

```json
{
  "ts": 1709272800.123,
  "step": 42,
  "regime": "nominal",
  "stability": 0.87,
  "layers": [
    {"R": 0.91, "psi": 1.23},
    {"R": 0.85, "psi": 4.56}
  ],
  "actions": [
    {
      "knob": "K",
      "scope": "global",
      "value": 0.05,
      "ttl_s": 10.0,
      "justification": "degraded: boost global coupling"
    }
  ]
}
```

## Event Record

```json
{
  "ts": 1709272800.456,
  "event": "regime_transition",
  "from": "nominal",
  "to": "degraded"
}
```

## Fields

| Field | Type | Present in |
|-------|------|------------|
| `ts` | float | All records. Unix epoch seconds. |
| `step` | int | Step records only. |
| `regime` | str | Step records. Current regime. |
| `stability` | float | Step records. Stability proxy. |
| `layers` | list[{R, psi}] | Step records. Per-layer state. |
| `actions` | list[ControlAction] | Step records. Actions taken. |
| `event` | str | Event records. Event type. |

## Deterministic Replay Contract

Given the same binding spec and initial seed, replaying the sequence of actions from the audit log must reproduce the same layer R values within floating-point tolerance (< 1e-12 absolute difference).

`ReplayEngine` in `audit.replay` loads JSONL and provides iteration over entries.

## Log Management

- Logs are never truncated during a run.
- `AuditLogger.close()` flushes and closes the file handle.
- Log rotation is the caller's responsibility.

## References

Deterministic replay requirements are specified in [eval_protocol.md](eval_protocol.md) § Deterministic Replay. The `stability` field corresponds to the Kuramoto order parameter R — see [lock_metrics.md](lock_metrics.md).
