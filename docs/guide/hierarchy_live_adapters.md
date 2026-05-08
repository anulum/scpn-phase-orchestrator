<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Hierarchy adapter guide -->

# Hierarchy Adapter Boundaries

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
