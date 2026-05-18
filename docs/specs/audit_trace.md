<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — audit trace specification -->

# Audit Trace

## Format

SPO audit has two compatible encodings:

- append-only JSONL, one JSON object per line, retained for existing replay,
  report, and plotting tools;
- append-only length-delimited protobuf event stream, using
  `proto/audit.proto` / `scpn_phase_orchestrator.audit.audit.proto`.

The protobuf stream is event-sourced: every header, step, and named audit event
is an immutable `AuditEnvelope` with sequence number, event type, timestamp,
payload digest, previous event hash, and event hash. The JSON payload is still
stored inside the envelope so existing deterministic replay semantics remain
unchanged while the stream becomes tail-safe and schema-versioned.

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

`ReplayEngine` in `runtime.replay` loads JSONL and provides iteration over entries.

## Protobuf Event Stream

The protobuf stream starts with the magic bytes `SPOA1\n`, followed by repeated
varint-length-delimited `spo.audit.AuditEnvelope` messages.

Envelope fields:

| Field | Type | Meaning |
|-------|------|---------|
| `schema_version` | uint32 | Audit envelope schema version. |
| `stream_id` | string | Logical stream identity. |
| `sequence` | uint64 | Monotonic event number, starting at 1. |
| `event_type` | string | `header`, `step`, or the named audit event. |
| `recorded_at` | Timestamp | Wall-clock capture time. |
| `source` | string | Producing subsystem. |
| `previous_hash` | string | Previous event hash, or 64 zeroes. |
| `payload_json` | string | Canonical JSON audit record. |
| `payload_sha256` | string | SHA-256 of `payload_json`. |
| `event_hash` | string | SHA-256 over envelope metadata and payload digest. |

`verify_event_stream_integrity()` checks sequence continuity, previous-hash
continuity, payload digests, and event hashes. Any mutation to a payload or
event envelope breaks the chain at the first affected event.

## Live Watch

`spo watch` tails the protobuf stream and prints replay summaries as events
arrive:

```bash
spo run domainpacks/minimal_domain/binding_spec.yaml \
  --steps 120 \
  --audit audit.jsonl \
  --audit-stream audit.spoa

spo watch audit.spoa --from-start
```

For bounded automation and tests, pass `--max-events N`. Without
`--from-start`, watch mode follows newly appended events.

## Log Management

- Logs are never truncated during a run.
- `AuditLogger.close()` flushes and closes the file handle.
- JSONL and protobuf stream rotation are the caller's responsibility.

## References

Deterministic replay requirements are specified in [eval_protocol.md](eval_protocol.md) § Deterministic Replay. The `stability` field corresponds to the Kuramoto order parameter R — see [lock_metrics.md](lock_metrics.md).
