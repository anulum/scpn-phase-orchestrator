# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Power Safety N-Channel Domainpack

# Power Safety N-Channel

This domainpack extends the power-grid profile beyond `P`, `I`, and `S`
with `Reserve`, `Weather`, and derived `Risk` channels. It is a compact
example of cross-channel safety evidence where weather forcing perturbs
grid phase, reserve margin suppresses risk, and risk gates symbolic grid
state transitions.

Compared with a P/I/S-only binding, the extra channels keep forecast
forcing, reserve headroom, and derived safety pressure separate. That
lets policy rules distinguish a phase-lock loss caused by exogenous
weather from one caused by insufficient reserve.
