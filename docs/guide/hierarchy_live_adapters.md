<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Hierarchy adapter guide -->

# Hierarchy Adapter Boundaries

## What These Boundaries Are For

These adapters are non-networked normalization points. Their role is to take
already-decoded runtime records and transform them into the strict internal audit
shape expected by `HierarchyTransportRuntime`.

From a product perspective, they keep replay and policy review logic separated
from transport plumbing. A transport error should become a structured decode
failure at the boundary; it should not leak ambiguous data into a shared ledger.

Hierarchy adapter helpers are decoded, non-socket boundaries over
`HierarchyTransportRuntime`. They validate caller-supplied records and return
deterministic audit payloads without owning HTTP servers, sockets, threads,
brokers, or event loops.

Available boundaries:

- `replay_hierarchy_jsonl(lines)`: deterministic JSONL or decoded-record replay.
- `handle_hierarchy_rest_payload(payload, headers=...)`: REST route helper for
  already-decoded request bodies and headers.
- `handle_hierarchy_frame(frame)`: WebSocket-style frame helper for already
  decoded frames.

JSONL replay parses string records with canonical finite JSON semantics before
runtime ingestion. Non-finite constants such as `NaN`, duplicate object keys,
blank lines, arrays, malformed JSON, and non-mapping decoded records fail
closed, so replay files cannot smuggle ambiguous edge summaries into the parent
watermark ledger.

Each returns a `HierarchyAdapterResult` with accepted/rejected counts, runtime
watermarks, parent plan summary, and the underlying sync ledger.

```python
from scpn_phase_orchestrator.supervisor import (
    HierarchyTransportRuntime,
    handle_hierarchy_rest_payload,
)

runtime = HierarchyTransportRuntime()
result = handle_hierarchy_rest_payload(
    {"envelope": envelope.to_audit_record()},
    headers={"content-type": "application/json"},
    runtime=runtime,
)
print(result.to_audit_record()["watermarks"])
```

The runtime accepts reduced child summaries only. Raw time-series, graph
payloads, actuator handles, raw coupling matrices, and raw evidence aliases are
rejected before audit serialisation.

That constraint is the design point for safe composability: each boundary does
not promise transport correctness or trust; it guarantees that once a payload is
inside the boundary contract, downstream replay and policy logic can proceed on
one coherent record format.

## Operational rationale

In deployment, these boundaries are the "shape firewall" between dynamic runtime
systems and deterministic audit workflows.

By keeping these adapters minimal and closed-form, the runtime gains two practical
benefits:
- fewer acceptance paths for malformed runtime payloads, and
- higher confidence that replay systems are evaluating the same payload schema
  that triggered decisions.

This lets teams diagnose issues at the right layer: parsing behavior at the
boundary, policy behavior in the supervisor, and integration behavior in
connectors.

## Byzantine Meta-Orchestrator Review

The Byzantine meta-orchestrator helper is also offline and non-networked. It
builds a deterministic review manifest from signed policy proposals and
hash-linked audit parents.

```python
from scpn_phase_orchestrator.supervisor import (
    build_bft_meta_orchestrator_manifest,
    sign_policy_proposal,
)

proposal = sign_policy_proposal(
    "node-a",
    {"policy": "hold", "knobs": {"K": 0.3}, "actuation": False},
    previous_audit_hash,
    signing_key,
)
manifest = build_bft_meta_orchestrator_manifest(
    [proposal, proposal_b, proposal_c],
    keyring,
    quorum=2,
)
```

The manifest verifies proposal signatures, payload hashes, parent audit hashes,
and quorum agreement. It reports the winning policy hash, accepted nodes,
rejected nodes, blocked reasons, and a hash-linked audit-chain digest.
`actuation_permitted` and `network_opened` remain false; accepted proposals still
have to pass the normal supervisor review gate before use.
