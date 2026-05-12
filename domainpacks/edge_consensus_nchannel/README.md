# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Edge Consensus N-Channel Domainpack

# Edge Consensus N-Channel

This domainpack demonstrates a six-channel edge orchestration profile:
`P`, `I`, `S`, `Load`, `Trust`, and the derived `ConsensusHealth`
channel. It models local nodes that phase-lock through gossip while
load pressure and trust scores alter the coupling path.

The example is intentionally compact. Its purpose is to show how
N-channel bindings declare named channels, groups, derived channels, and
cross-channel coupling in one runnable binding spec.

Compared with a P/I/S-only binding, the extra channels expose why an edge
cluster loses coherence: `Load` separates congestion pressure from raw
phase drift, `Trust` separates peer-quality evidence from event cadence,
and `ConsensusHealth` gives replay/report tooling one derived safety
signal to track across the run.

The binding spec includes a `value_alignment` template for review-time edge
coupling and signed gateway-lag correction guards.

## Sheaf Obstruction Demo

`sheaf_obstruction_demo.py` demonstrates the sheaf-coherence supervisor on this
heterogeneous six-channel binding. It compares a nominal edge/gateway/parent
section against a gateway-stressed section where `Load`, `Trust`, and derived
`ConsensusHealth` no longer agree across restriction maps.

Run the replay with:

```bash
PYTHONPATH=src python domainpacks/edge_consensus_nchannel/sheaf_obstruction_demo.py
```

The replay is non-actuating. It validates the binding spec, evaluates the
directed sheaf Laplacian, reports nominal and stressed obstruction metrics, and
emits the obstruction delta as audit evidence.
