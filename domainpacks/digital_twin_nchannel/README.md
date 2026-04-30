# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Digital Twin N-Channel Domainpack

# Digital Twin N-Channel

This domainpack models a plant and its live digital twin with six
channels: `P`, `I`, `S`, `Thermal`, `Quality`, and derived
`TwinResidual`. The residual channel shows how measured phase,
temperature, and quality streams can be folded into a supervisor-visible
coherence signal without adding another physical oscillator layer.

Compared with a P/I/S-only binding, `Thermal` and `Quality` make the
cause of plant/twin divergence explicit, while `TwinResidual` gives
audit, replay, and policy dry-runs a named derived channel instead of
burying the residual inside a domain-specific script.
