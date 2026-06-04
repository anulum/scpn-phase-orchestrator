# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — spatial modulator quickstart

"""Minimal moving-agent coupling example."""

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.coupling import SpatialCouplingModulator

knm = np.array(
    [
        [0.0, 1.0, 0.5],
        [1.0, 0.0, 0.8],
        [0.5, 0.8, 0.0],
    ],
    dtype=np.float64,
)
positions = np.array([[0.0], [0.25], [2.0]], dtype=np.float64)
modulated = SpatialCouplingModulator(K_base=0.75).modulate(knm, positions)
print(modulated)
