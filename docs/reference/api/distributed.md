# Distributed sync API reference

The distributed sync API defines the transport-neutral phase-vector gossip surface
used by the runtime when multiple oscillator workers must keep their local UPDE
state close enough to exchange control decisions safely.

The module is intentionally narrow.
It does not open sockets.
It does not run a background worker.
It does not decide cluster membership.
It defines deterministic message records, validation boundaries, replay helpers,
and bounded phase-correction logic that a caller can embed in its own transport.

The implementation lives in
`scpn_phase_orchestrator.runtime.distributed.sync`.
The public package re-exports the same symbols from
`scpn_phase_orchestrator.runtime.distributed`.

::: scpn_phase_orchestrator.runtime.distributed.sync
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Primary use cases

- Multi-process simulation workers that integrate different oscillator shards and
  periodically exchange phase vectors.
- Laboratory control services that must ingest signed or authenticated transport
  payloads only after deterministic message-level validation has passed.
- Replayable distributed experiments where every accepted, rejected, and dropped
  phase message must be auditable after the run.
- Deterministic failure-injection tests for lossy peer-to-peer sync topologies.
- Conservative phase alignment in safety-aware loops where a peer update must not
  produce an unbounded local phase jump.

## Design boundary

This API is a synchronization contract, not a networking stack.

A caller is responsible for:

- selecting the transport,
- authenticating the transport session,
- assigning peer names,
- deciding when to call `observe_local`,
- deciding when to call `ingest`,
- deciding when to call `synchronise`,
- persisting the audit records if persistence is required,
- enforcing deployment-specific peer admission policy,
- enforcing deployment-specific rate limits outside this module.

The module is responsible for:

- serializing phase messages in canonical JSON,
- rejecting non-finite phase payloads and duplicate object keys,
- preserving a protocol kind string,
- checking protocol version compatibility,
- checking payload digests,
- rejecting self-originated messages,
- rejecting stale or duplicate sequence numbers,
- storing the latest valid peer phases,
- computing bounded circular corrections,
- reporting deterministic replay statistics.

## Exported symbols

| Symbol | Kind | Purpose |
| --- | --- | --- |
| `DistributedSyncConfig` | dataclass | Validated local-node configuration for message limits and correction bounds. |
| `PhaseSyncMessage` | dataclass | Canonical wire record for one peer phase-vector observation. |
| `GossipIngestResult` | dataclass | Structured ingest decision with reason, sequence, peer, and digest metadata. |
| `PhaseGossipNode` | class | Stateful local gossip participant with sequence watermarks and bounded sync. |
| `LossyGossipReplayResult` | dataclass | Deterministic summary for synthetic lossy all-to-all replay. |
| `simulate_lossy_phase_gossip` | function | Deterministic replay helper for transport-independent topology tests. |

## Protocol overview

Every phase gossip message carries the protocol kind `spo.phase_sync`.
The kind prevents accidental acceptance of unrelated JSON payloads.
The version number prevents silent mixing of incompatible encodings.
The sequence number provides a per-peer monotonic ordering boundary.
The digest binds the canonical message content to the transmitted record.

The wire flow is:

1. A node calls `observe_local(phases, timestamp)`.
2. The node validates the local phase vector length and finiteness.
3. The node increments its local sequence number.
4. The node constructs a `PhaseSyncMessage`.
5. The message is converted to a canonical record with `to_record()`.
6. The record receives a deterministic SHA-256 digest.
7. The record is encoded with `to_wire()`.
8. A caller transmits the wire bytes through its chosen transport.
9. A receiving caller passes bytes to `PhaseGossipNode.ingest()`.
10. The receiver decodes with `PhaseSyncMessage.from_wire()`.
11. The receiver validates kind, version, finiteness, duplicate-key absence,
    size, and digest.
12. The receiver applies peer-specific sequence watermarks.
13. The receiver records acceptance or rejection as `GossipIngestResult`.
14. The receiver may later call `synchronise(local_phases)`.
15. The returned phase vector applies bounded circular correction.

## DistributedSyncConfig

`DistributedSyncConfig` is the local contract object for a gossip node.

Fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `node_id` | `str` | Stable local peer identifier. |
| `n_oscillators` | `int` | Required length of every phase vector. |
| `protocol_version` | `int` | Accepted message protocol version. |
| `peer_timeout` | `float` | Maximum tolerated age for peer records in seconds. |
| `max_phase_step_rad` | `float` | Maximum absolute phase correction per oscillator per sync step. |

Validation invariants:

- `node_id` must be non-empty after trimming whitespace.
- `n_oscillators` must be positive.
- `protocol_version` must be positive.
- `peer_timeout` must be finite and positive.
- `max_phase_step_rad` must be finite and positive.
- `max_phase_step_rad` is normalized to a float for deterministic arithmetic.
- `peer_timeout` is normalized to a float for deterministic arithmetic.

Configuration errors are raised immediately at construction time.
The module does not defer invalid runtime settings to the first message.
That makes configuration failures easier to diagnose before a distributed run
starts.

### Choosing `node_id`

A `node_id` should identify the logical gossip participant.
It should not encode volatile process IDs when the peer identity should survive a
restart.
It should not reuse the same value for two live participants.
If a receiver ingests a message whose `node_id` matches its own configuration,
the message is rejected as a self message.

### Choosing `n_oscillators`

`n_oscillators` is part of the safety boundary.
A receiver rejects a phase vector with any other length.
This prevents a peer with a stale model layout from being blended into the local
state.
The caller should derive this value from the same model manifest that defines
the UPDE state dimension.

### Choosing `protocol_version`

The current implementation uses a positive integer version.
A receiver rejects any message whose `protocol_version` differs from the local
configuration.
A deployment that must roll forward in stages should run a compatibility layer
outside this module and only pass normalized messages into `ingest()`.

### Choosing `peer_timeout`

`peer_timeout` is evaluated when `synchronise()` is called.
Peers older than the timeout do not contribute to the circular mean.
This prevents an old but previously valid phase vector from pulling the local
node after the peer has stopped reporting.

### Choosing `max_phase_step_rad`

`max_phase_step_rad` caps each oscillator correction independently.
The correction is computed on the circle and then clipped into the interval
`[-max_phase_step_rad, max_phase_step_rad]`.
This creates a bounded circular correction rather than an unbounded assignment to
the peer mean.

## PhaseSyncMessage

`PhaseSyncMessage` is the immutable value object for one local phase observation.

Fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `node_id` | `str` | Sender peer identity. |
| `sequence` | `int` | Sender-local monotonic sequence number. |
| `timestamp` | `float` | Sender observation time. |
| `phases` | `tuple[float, ...]` | Phase vector in radians. |
| `protocol_version` | `int` | Encoding version. |
| `digest` | `str | None` | SHA-256 digest over the canonical message record. |

Validation invariants:

- `node_id` must be non-empty after trimming whitespace.
- `sequence` must be non-negative.
- `timestamp` must be finite.
- `protocol_version` must be positive.
- `phases` must be a sequence of finite numeric values.
- `phases` are stored as a tuple of floats.
- A digest, when present, must match the canonical record.

A message object may be created without a digest.
`to_record()` computes the canonical digest if it is missing.
`to_wire()` always emits a record with a digest.

## Canonical record

`PhaseSyncMessage.to_record()` returns a JSON-compatible dictionary.
The record includes:

- `kind`, with value `spo.phase_sync`,
- `node_id`,
- `sequence`,
- `timestamp`,
- `phases`,
- `protocol_version`,
- `digest`.

The digest is computed over the canonical content without the digest field.
The content is serialized with stable key ordering and compact separators.
The digest algorithm is SHA-256.
The resulting lowercase hexadecimal digest is embedded in the outgoing record.

The digest protects deterministic integrity at the message-record layer.
It is not a replacement for transport authentication.
It does catch accidental mutation, non-canonical record assembly, and many forms
of malformed replay payload before the payload enters node state.

## Wire encoding

`PhaseSyncMessage.to_wire()` encodes the canonical record as UTF-8 JSON bytes.
The encoder rejects non-finite floating point values by using finite JSON rules.
That means `NaN`, positive infinity, and negative infinity are never emitted by
this API.

`PhaseSyncMessage.from_wire()` decodes the UTF-8 JSON bytes and validates the
record before returning a message instance.
The decoder also rejects non-finite constants if they appear in inbound JSON.
This is important because some JSON libraries accept non-standard constants by
default.

## finite JSON boundary

The finite JSON boundary is explicit.
Phase gossip must not carry non-finite numbers.
A non-finite phase has no well-defined circular correction and cannot be safely
included in a deterministic replay.

Rejected examples include:

```json
{"kind":"spo.phase_sync","phases":[NaN]}
```

```json
{"kind":"spo.phase_sync","phases":[Infinity]}
```

```json
{"kind":"spo.phase_sync","phases":[-Infinity]}
```

The rejection happens before a peer record is stored.
The local node remains unchanged after rejection.

## from_wire rejection modes

`PhaseSyncMessage.from_wire()` rejects malformed records before node-level
watermarks are evaluated.

The rejection classes include:

- invalid UTF-8 JSON bytes,
- JSON that is not an object,
- `kind` missing or not equal to `spo.phase_sync`,
- missing required fields,
- non-string `node_id`,
- empty `node_id`,
- negative `sequence`,
- non-integer `sequence`,
- non-finite `timestamp`,
- non-positive `protocol_version`,
- non-sequence `phases`,
- non-finite phase values,
- digest mismatch,
- malformed digest field.

The node-level `ingest()` method converts these failures into a rejected
`GossipIngestResult`.
Callers can persist that record without parsing exception text.

## GossipIngestResult

`GossipIngestResult` is the structured ingest audit result.

Fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `accepted` | `bool` | Whether the message changed peer state. |
| `reason` | `str` | Stable human-readable decision reason. |
| `node_id` | `str | None` | Sender identity if available. |
| `sequence` | `int | None` | Sender sequence if available. |
| `digest` | `str | None` | Message digest if available. |

Accepted messages use a positive reason.
Rejected messages identify the failed boundary.
The result is intentionally simple so it can be embedded in audit logs, replay
reports, and run diagnostics without a custom serializer.

## PhaseGossipNode

`PhaseGossipNode` owns local gossip state for one configured participant.

The node stores:

- the validated `DistributedSyncConfig`,
- the local outgoing sequence counter,
- the latest accepted phase vector for each peer,
- the latest accepted timestamp for each peer,
- the latest accepted digest for each peer,
- per-peer sequence watermarks.

The node does not store transport handles.
The node does not sleep.
The node does not retry.
Those behaviours belong to the deployment layer.

## observe_local

`observe_local(phases, timestamp)` creates a local outbound message.

Contract:

- `phases` must contain exactly `n_oscillators` finite values.
- `timestamp` must be finite.
- the node increments its local sequence number once per call.
- the returned `PhaseSyncMessage` has the local `node_id`.
- the returned message uses the local `protocol_version`.
- the returned message is ready for `to_wire()`.

A caller should call `observe_local()` after it has advanced its local state and
before it transmits the observation to peers.

## ingest

`ingest(wire)` validates and stores a peer message.

The method applies these boundaries in order:

1. decode the wire record,
2. validate message kind and digest,
3. compare `protocol_version`,
4. reject self-originated messages,
5. compare phase-vector length with `n_oscillators`,
6. compare sender sequence with the stored sequence watermarks,
7. store peer phases, timestamp, digest, and sequence,
8. return an accepted `GossipIngestResult`.

A protocol_version mismatch is rejected before sequence state is updated.
A self message is rejected before sequence state is updated.
A stale or duplicate sequence is rejected before peer phases are replaced.
A digest mismatch is rejected before the message reaches the node-level checks.

## sequence watermarks

Sequence watermarks enforce monotonic progress per sender.
For each accepted `node_id`, the node stores the highest accepted sequence.
The next accepted message from that sender must have a strictly greater
sequence.

This rejects:

- duplicate delivery,
- out-of-order replay,
- stale retransmission,
- accidental reuse of a sender sequence counter.

The watermark is local to the receiver.
Another receiver may accept a different highest sequence depending on its own
transport history.
That is expected in lossy gossip.
The replay helper reports this behaviour explicitly.

## Peer timeout handling

Peer timeout is applied during `synchronise()`.
A peer is eligible only when `current_time - peer_timestamp <= peer_timeout`.
The local node does not delete timed-out peers during this calculation.
It simply ignores them for the current correction.
That makes audit records stable and avoids hidden state mutation during a pure
synchronisation pass.

If no peers are eligible, `synchronise()` returns the local phases normalized by
the module's numeric path and applies no correction.

## synchronise

`synchronise(local_phases, current_time=None)` computes a corrected local phase
vector.

Contract:

- `local_phases` must contain exactly `n_oscillators` finite values.
- eligible peer phases are selected by timeout.
- the local phases and eligible peer phases are combined on the unit circle.
- the circular mean is converted into per-oscillator phase differences.
- each difference is clipped by `max_phase_step_rad`.
- the clipped corrections are added to the local phases.
- the returned array has the same oscillator count.

The result is a bounded circular correction.
It is not a hard overwrite by a peer vector.
This matters because phase angles wrap at `2*pi` and linear subtraction can take
the long path around the circle.

## Circular correction example

Assume a local phase is close to `+pi` and a peer phase is close to `-pi`.
A linear difference appears near `-2*pi`.
The circular difference is small because both angles are near the same point on
the circle.
The implementation follows the circular path.
The configured step bound is then applied to that circular difference.

## LossyGossipReplayResult

`LossyGossipReplayResult` summarizes a deterministic replay from
`simulate_lossy_phase_gossip`.

Fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `accepted` | `int` | Number of accepted peer messages. |
| `rejected` | `int` | Number of rejected peer messages. |
| `dropped` | `int` | Number of intentionally dropped directed edges. |
| `peer_counts` | `dict[str, int]` | Accepted peer count by receiver node. |
| `final_phases` | `dict[str, list[float]]` | Final synchronized phase vector by node. |

The result is deterministic for the same inputs.
It is designed for adapter tests, examples, and benchmark harnesses that need a
transport-independent distributed sync surface.

## simulate_lossy_phase_gossip

`simulate_lossy_phase_gossip(configs, initial_phases, timestamp, drop_edges)`
constructs a node for each config, emits local observations, drops selected
directed edges, ingests the remaining messages, and returns synchronized final
phases.

The helper is deliberately small.
It does not model network latency.
It does not model retries.
It does not randomize delivery.
It provides a deterministic all-to-all replay with explicit drop edges.

Inputs:

- `configs`: iterable of `DistributedSyncConfig` values,
- `initial_phases`: mapping from `node_id` to phase vectors,
- `timestamp`: replay observation time,
- `drop_edges`: optional directed edge set of `(sender, receiver)` pairs.

Output:

- `LossyGossipReplayResult` with accept, reject, drop, peer-count, and final
  phase-vector data.

## Replay safety checks

The replay helper inherits the same safety checks as live node ingestion.
It validates local observations through `observe_local()`.
It serializes through `to_wire()`.
It ingests through `ingest()`.
It synchronizes through `synchronise()`.
It therefore exercises the same protocol contract as a caller using a real
transport.

## Minimal send/receive example

```python
from scpn_phase_orchestrator.runtime.distributed import (
    DistributedSyncConfig,
    PhaseGossipNode,
)

sender = PhaseGossipNode(
    DistributedSyncConfig(node_id="node-a", n_oscillators=3)
)
receiver = PhaseGossipNode(
    DistributedSyncConfig(node_id="node-b", n_oscillators=3)
)

message = sender.observe_local([0.0, 0.5, 1.0], timestamp=1.0)
wire = message.to_wire()
result = receiver.ingest(wire)

assert result.accepted
```

## Minimal synchronisation example

```python
corrected = receiver.synchronise([0.1, 0.4, 1.2], current_time=1.1)
assert len(corrected) == 3
```

The returned vector is numerically corrected but bounded by the receiver's
`max_phase_step_rad`.

## Deterministic lossy replay example

```python
from scpn_phase_orchestrator.runtime.distributed import (
    DistributedSyncConfig,
    simulate_lossy_phase_gossip,
)

configs = [
    DistributedSyncConfig(node_id="a", n_oscillators=2),
    DistributedSyncConfig(node_id="b", n_oscillators=2),
    DistributedSyncConfig(node_id="c", n_oscillators=2),
]

result = simulate_lossy_phase_gossip(
    configs,
    initial_phases={
        "a": [0.0, 0.2],
        "b": [0.1, 0.3],
        "c": [0.2, 0.4],
    },
    timestamp=10.0,
    drop_edges={("c", "a")},
)

assert result.dropped == 1
assert result.peer_counts["a"] == 1
```

## Error-path examples

A digest mismatch is rejected:

```python
wire = message.to_wire().replace(b"0", b"1", 1)
result = receiver.ingest(wire)
assert not result.accepted
```

A protocol version mismatch is rejected:

```python
old_peer = PhaseGossipNode(
    DistributedSyncConfig(node_id="old", n_oscillators=3, protocol_version=1)
)
new_peer = PhaseGossipNode(
    DistributedSyncConfig(node_id="new", n_oscillators=3, protocol_version=2)
)
wire = old_peer.observe_local([0.0, 0.1, 0.2], timestamp=1.0).to_wire()
result = new_peer.ingest(wire)
assert not result.accepted
```

A stale or duplicate sequence is rejected:

```python
wire = sender.observe_local([0.0, 0.5, 1.0], timestamp=1.0).to_wire()
assert receiver.ingest(wire).accepted
assert not receiver.ingest(wire).accepted
```

## Audit record guidance

`PhaseGossipNode.to_audit_record()` reports the node's sync state without
requiring callers to inspect private attributes.
A typical audit record includes local identity, peer watermarks, peer digests,
and peer timestamps.

Use audit records to answer:

- which peers have contributed to this node,
- which sender sequence is the latest accepted for each peer,
- which digest was accepted most recently,
- whether a peer became stale relative to the current run time,
- whether a replay reproduced the same peer acceptance structure.

## Security model

The message digest is a deterministic integrity check for the canonical record.
It is not a deployment security boundary by itself.
A production transport should still provide authenticated channels,
peer admission, replay policy, and deployment-level request shaping.

This module deliberately avoids embedding those decisions because they vary by
runtime environment.
The stable contract is that malformed records never enter peer state and that
accepted records can be reconstructed deterministically.

## Numerical model

The sync operation treats phases as angles, not ordinary real-valued features.
For each oscillator index, eligible phases are mapped onto the unit circle.
The mean direction is computed from the circular representation.
The local phase is moved toward that direction by a clipped amount.

This preserves wraparound behaviour.
It also prevents one peer from forcing an arbitrary jump in a single sync step.
The correction bound is local configuration, so a conservative receiver can
accept messages while still limiting actuation magnitude.

## Determinism model

Determinism depends on:

- canonical JSON ordering,
- finite numeric inputs,
- explicit sequence watermarks,
- deterministic drop-edge selection in replay,
- stable iteration over the supplied configs and phase mappings,
- deterministic circular arithmetic for the same floating point backend.

The module does not depend on wall-clock time unless the caller omits
`current_time` in `synchronise()`.
For reproducible experiments, pass `current_time` explicitly.

## Compatibility notes

This reference documents the Python distributed sync surface.
There is no separate polyglot counterpart for this transport-neutral helper in
this repository at the time of this update.
No benchmark contract changes are required because this update does not change
production runtime behaviour.
If a future transport or polyglot binding is added, it should preserve the same
canonical record and rejection semantics before being treated as compatible.

## Behavioural tests guarding this reference

The following module-specific tests guard the distributed sync API surface:

- `tests/test_distributed_sync.py` checks accepted gossip, correction bounds,
  deterministic replay, and stale peer handling.
- `tests/test_distributed_sync_validation.py` checks validation failures,
  digest mismatch, protocol mismatch, and duplicate sequence rejection.
- `tests/test_reference_api_distributed.py` checks that this reference remains
  deep enough and continues to document the public protocol contract.

The docs guard is not a coverage bucket.
It protects the public API reference from regressing back to a shallow index page
that omits the semantics required by users and maintainers.

## Operational checklist

Use this checklist before connecting the module to a real deployment:

- Confirm every participant uses the same oscillator count.
- Confirm every participant uses the same protocol version.
- Choose stable node identifiers before the run starts.
- Reject duplicate live participants with the same node identifier.
- Persist accepted ingest results when replay evidence is required.
- Persist rejected ingest results when failure diagnosis is required.
- Pass explicit current_time values in deterministic experiments.
- Keep peer_timeout shorter than the maximum tolerated stale-state window.
- Set max_phase_step_rad from the physical actuation budget.
- Avoid deriving max_phase_step_rad from arbitrary UI preferences.
- Treat digest mismatch as a malformed message boundary.
- Treat protocol mismatch as an incompatible deployment boundary.
- Treat duplicate sequence rejection as expected under repeated delivery.
- Treat out-of-order sequence rejection as expected under stale transport buffers.
- Do not bypass from_wire validation in production adapters.
- Do not store peer phases from rejected messages.
- Do not blend peers that exceed the timeout boundary.
- Do not overwrite local phases with peer phases directly.
- Use circular differences for every phase correction.
- Clip each oscillator correction independently.
- Record drop_edges explicitly in deterministic replay tests.
- Keep replay topology deterministic when comparing runs.
- Use the audit record instead of private attributes for diagnostics.
- Handle transport authentication outside this module.
- Handle deployment-level request shaping outside this module.
- Handle peer admission outside this module.
- Keep non-finite values outside the phase pipeline.
- Fail fast on invalid local configuration.
- Fail closed on malformed inbound wire records.
- Document any future protocol version change with migration notes.
- Add adapter tests before binding a new transport.
- Add compatibility tests before adding a polyglot binding.
- Keep examples deterministic and small.
- Prefer named nodes in examples over numeric positions.
- Validate replay results by accepted, rejected, dropped, and peer_counts.
- Validate final phases by physical invariants, not only by shape.
- Use bounded correction tests near the wraparound boundary.
- Use stale peer tests with explicit timestamps.
- Use digest mutation tests for wire integrity.
- Use duplicate delivery tests for sequence watermarks.
- Use protocol mismatch tests for version boundaries.
- Use wrong-size phase tests for model layout boundaries.
- Use non-finite phase tests for numeric safety boundaries.
- Keep documentation examples aligned with exported symbol names.
- Keep public references free of internal planning notes.
- Keep transport examples separate from the core sync contract.
- Do not introduce sockets into the sync module.
- Do not introduce background scheduling into the sync module.
- Do not introduce random drops into simulate_lossy_phase_gossip.
- Do not treat digest checks as a replacement for deployment security.
- Do not call synchronise with a different oscillator count.
- Do not call observe_local with non-finite timestamps.
- Do not accept a message whose kind is not spo.phase_sync.
- Do not accept a message whose digest does not match its canonical content.
- Do not accept a self message.
- Do not accept a stale or duplicate sequence.
- Do not silently coerce invalid configuration.
- Do not publish a new transport without audit-record coverage.
- Do not publish a new transport without malformed-payload coverage.
- Do not publish a new transport without replay evidence.
- Review peer_timeout and max_phase_step_rad together.
- Review model dimension changes before accepting old peer traffic.
- Review protocol version changes before mixed deployments.
- Review audit retention needs before long experiments.
- Review deterministic replay needs before enabling lossy links.
- Review phase wraparound cases near positive and negative pi.
- Review sender sequence reset behaviour after process restart.
- Review receiver watermark reset behaviour after state restoration.
- Review how transport-level retries interact with duplicate sequence rejection.
- Review how peer admission interacts with node_id naming.
- Review how deployment monitoring surfaces accepted and rejected counts.
- Review how dashboards label dropped replay edges.
- Review how runbooks explain protocol_version mismatch.
- Review how runbooks explain stale or duplicate sequence rejection.
- Review how runbooks explain digest mismatch.
- Review how runbooks explain finite JSON rejection.
- Review how runbooks explain bounded circular correction.
- Review how operators reproduce a lossy replay.
- Review whether peer digests are stored in audit output.
- Review whether peer timestamps are stored in audit output.
- Review whether peer sequence watermarks are stored in audit output.
- Review whether accepted peer counts match expected topology.
- Review whether dropped edge counts match test setup.
- Review whether rejected counts are expected under the transport.
- Review whether no-peer synchronisation returns a finite vector.
- Review whether timeout-only synchronisation leaves local phases bounded.
- Review whether all examples import from the public distributed package.
- Review whether downstream docs link to this reference.
- Review whether future changes require benchmark notes.
- Review whether future changes require polyglot compatibility notes.
- Review whether future changes require migration notes.
- Review whether future changes require changelog entries.
- Review whether future changes require module-specific tests.
- Review whether future changes preserve transport neutrality.
- Review whether future changes preserve deterministic replay.
- Review whether future changes preserve bounded correction.
- Review whether future changes preserve finite JSON rejection.
- Review whether future changes preserve SHA-256 digest semantics.
- Review whether future changes preserve sequence watermarks.
- Review whether future changes preserve protocol kind spo.phase_sync.
- Review whether future changes preserve protocol_version mismatch rejection.
- Review whether future changes preserve stale or duplicate sequence rejection.
- Review whether future changes preserve self-message rejection.
- Review whether future changes preserve wrong-size phase rejection.
- Review whether future changes preserve node audit records.
- Review whether future changes preserve LossyGossipReplayResult fields.
- Review whether future changes preserve simulate_lossy_phase_gossip determinism.
- Review whether future changes preserve DistributedSyncConfig validation.
- Review whether future changes preserve PhaseSyncMessage validation.
- Review whether future changes preserve GossipIngestResult fields.
- Review whether future changes preserve PhaseGossipNode state boundaries.
- Review whether future changes preserve examples as runnable documentation.
- Review whether future changes preserve docs guard expectations.
- Review whether future changes preserve public roadmap accuracy.
- Review whether future changes preserve internal TODO accuracy.
- Review whether future changes preserve changelog traceability.
- Review whether future changes preserve user-facing terminology.
- Review whether future changes avoid transport-specific assumptions.
- Review whether future changes avoid shallow shape-only validation.
- Review whether future changes keep physics invariants visible.
- Review whether future changes keep distributed safety boundaries visible.
- Review whether future changes keep replay contracts visible.
- Review whether future changes keep audit contracts visible.

## Release-note summary

The distributed sync surface provides a deterministic phase gossip contract for
runtime integrations that need peer phase exchange without coupling the core
library to a specific network transport.
The API is safe by construction at the message boundary: wrong kind, wrong
version, wrong size, non-finite data, digest mismatch, self messages, and stale
or duplicate sender sequences are rejected before they can influence local
state.
The synchronization operation then applies bounded circular correction so that
accepted peer data can guide phase alignment without creating an unbounded jump.
