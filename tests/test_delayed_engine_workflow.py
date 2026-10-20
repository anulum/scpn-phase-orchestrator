# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Delayed engine order-parameter workflow contract

"""Qualify delayed phase evolution feeding the real order-parameter monitor."""

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.delay import DelayedEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter


class TestEngineVariantsPipelineWiring:
    """Delayed phase evolution feeds the real order-parameter monitor."""

    def test_delayed_engine_phases_finite_and_advance(self) -> None:
        """Actual delayed steps advance finite phases consumed by the order monitor."""
        n = 4
        eng = DelayedEngine(n, dt=0.01, delay_steps=2)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        omegas = np.ones(n)
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        initial = phases.copy()
        for _ in range(10):
            phases = eng.step(phases, omegas, knm, 0.5, 1.0, alpha)
        assert np.all(np.isfinite(phases))
        assert not np.allclose(phases, initial), "Phases must advance"
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0, "R from delayed engine must be valid"
