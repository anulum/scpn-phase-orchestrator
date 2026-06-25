# Subsystem: `runtime` / `actuation` / `assurance` / `audit` / `meta` / `artifacts` — trust & execution spine

The concrete execution loop and the tamper-evidence layer. `runtime` 56 files
(~17.4k LOC), `actuation` 9, `assurance` 6, `audit` 2 (protobuf schema only),
`meta` 2, `artifacts` 2.

## Inputs

`UPDEState` and `BoundaryState` per step; `ControlAction` proposals from the
supervisor; for audit, prior JSONL records (to resume the hash chain) and
environment keys (`SPO_AUDIT_KEY`, `SPO_AUDIT_KEYRING`).

## Outputs

- Projected `ControlAction`s (clamped, rate-limited).
- A SHA-256 hash-chained **JSONL audit log** and a protobuf **event stream**,
  optionally HMAC-signed.
- An `AssuranceCaseBundle` mapping evidence to EU AI Act 2024/1689, ISO/IEC
  42001:2023, and ANSI/UL 4600 clauses, with a recomputed bundle hash.
- `QPUDataArtifact` (canonical JSON + per-component SHA-256) for the quantum
  sibling.

## Processing model

- `runtime/simulation.simulate` — the master step loop (audit record → supervisor
  decision → action projection → engine step). When a protobuf audit stream is
  attached, the run-end `SimulationResult` includes a flushed whole-stream
  integrity summary from `verify_event_stream_integrity()`.
- `actuation/ActionProjector.project` — clamp to bounds, then per-step rate-limit,
  then re-clamp. Deterministic; always closes the loop. An optional neural
  Control Barrier Function filter (interval-bound-propagation verified) and a
  Koopman MPC are available.
- `audit` — hash chain `event_hash = SHA-256(canonical_json(record))`; optional
  HMAC over the chain; deterministic `ReplayEngine`.
- `assurance` — content-addressed evidence items and a frozen, review-only bundle
  (`actuation_permitted=False`).
- `runtime/deterministic` — a bounded-jitter, WCET-budgeted hard-deadline loop
  with a deadline-miss policy.

## Backends

Pure Python/NumPy. No Rust in the runtime loop; the `spo-supervisor` Rust crate
is not bound here.

## Wiring

`monitor → supervisor → actuation (projector) → audit → back to the engine`. The
CLI `audit`, `assurance`, and `verification` commands consume the audit log,
build bundles, and run replay.

## Scope boundaries (verified by execution)

- **Per-step tamper evidence is append-time hash chaining.** `simulate()` does
  not rescan the full stream before each action; a protobuf audit stream is
  verified once at run end and surfaced on `SimulationResult`.
- **Audit signing is environment-gated** (`SPO_AUDIT_KEY`), not structural;
  unsigned writes are permitted.
- **`actuation_permitted=False` is documentary** — it is a frozen-dataclass
  assertion never consulted by the actuation flow. Runtime gating remains in
  the safety-tier checks and `ActionProjector` clamp/rate-limit path.
- The hard-deadline loop defaults to `miss_policy="observe"` (records and
  continues); `"abort"` must be requested explicitly.
- The Koopman MPC is implemented but **not wired into the default `simulate()`
  loop**; `meta`-transfer extracts policy examples but is not consumed by
  actuation.
- The `audit.proto` schema omits the signature fields, which the writer adds
  dynamically (schema/implementation drift).
