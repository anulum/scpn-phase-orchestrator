# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Simplicial engine order-parameter workflow contract

"""
Workflow contract for SimplicialEngine finite phase output feeding the order-
parameter monitor.
"""

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine


def test_simplicial_engine_phases_finite_and_valid_r():
    n = 4
    eng = SimplicialEngine(n, dt=0.01, sigma2=0.1)
    phases = np.array([0.0, 0.5, 1.0, 1.5])
    omegas = np.ones(n)
    knm = np.full((n, n), 0.5)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    result = eng.step(phases, omegas, knm, 0.5, 1.0, alpha)
    assert np.all(np.isfinite(result))
    r, _ = compute_order_parameter(result)
    assert 0.0 <= r <= 1.0
