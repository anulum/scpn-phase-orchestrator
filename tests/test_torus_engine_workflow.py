# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Torus engine driven-dynamics workflow contract

"""
Workflow contract for TorusEngine driven phase dynamics feeding the order-parameter
monitor.
"""

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.geometric import TorusEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter


def test_torus_engine_zeta_changes_dynamics():
    """Torus engine with zeta must differ from pure Euler advance."""
    n = 4
    eng = TorusEngine(n, dt=0.01)
    phases = np.array([0.0, 0.5, 1.0, 1.5])
    omegas = np.ones(n)
    knm = np.zeros((n, n))
    alpha = np.zeros((n, n))
    result = eng.step(phases, omegas, knm, 1.0, 1.0, alpha)
    naive = (phases + 0.01 * omegas) % (2 * np.pi)
    assert not np.allclose(result, naive), "Zeta must alter dynamics beyond pure Euler"
    r, _ = compute_order_parameter(result)
    assert 0.0 <= r <= 1.0
