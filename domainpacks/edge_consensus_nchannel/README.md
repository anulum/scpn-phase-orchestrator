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
