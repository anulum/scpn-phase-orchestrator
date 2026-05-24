# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stochastic self-consistency contracts

"""
Numerical contracts for Kuramoto stochastic self-consistency and optimal-noise
search.
"""

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.stochastic import (
    _self_consistency_R,
    find_optimal_noise,
)


class TestStochasticSelfConsistency:
    """Verify the Kuramoto self-consistency equation R = f(K/D)
    and optimal noise search."""

    def test_self_consistency_r_bounded(self):
        """R from self-consistency must be in (0, 1) for finite K/D."""
        R = _self_consistency_R(2.1, 1.0)
        assert 0.0 < R < 1.0

    def test_higher_coupling_higher_r(self):
        """Stronger coupling → higher R (monotonicity)."""
        R_weak = _self_consistency_R(1.5, 1.0)
        R_strong = _self_consistency_R(5.0, 1.0)
        assert R_strong > R_weak, f"K↑ → R↑: weak={R_weak:.3f}, strong={R_strong:.3f}"

    def test_find_optimal_noise_returns_nonnegative_d(self):
        n = 4
        engine = UPDEEngine(n, dt=0.01)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        omegas = np.ones(n)
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = find_optimal_noise(
            engine,
            phases,
            omegas,
            knm,
            alpha,
            D_range=None,
            n_steps=20,
        )
        assert result.D >= 0.0
        assert hasattr(result, "R_achieved")


# ---------------------------------------------------------------------------
# Prediction model
# ---------------------------------------------------------------------------
