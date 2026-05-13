<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Distributed Sync API -->

# Distributed Sync API

The distributed sync module defines a transport-neutral phase-vector gossip
protocol for multiple UPDE nodes. Each node publishes canonical JSON wire
messages containing its current phase vector, protocol version, node id,
sequence number, wall-clock timestamp, and SHA-256 digest.

Receivers reject malformed, tampered, stale, duplicate, wrong-version, and
wrong-dimension messages before they can affect the local phase state.
Accepted peer states are merged through a circular mean and applied as a
bounded phase correction, so one lossy or delayed peer cannot force an
unbounded jump in the local oscillator state.

The module does not open sockets. Production deployments can carry the wire
messages over UDP, QUIC, WebSocket, Kafka, REST, or another owned transport
while keeping the protocol validation and bounded synchronisation logic shared.

::: scpn_phase_orchestrator.distributed.sync
