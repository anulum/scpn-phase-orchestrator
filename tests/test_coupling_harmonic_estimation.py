# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling harmonic estimation contracts

"""Numerical contracts for harmonic coupling estimation outputs and shapes."""

from __future__ import annotations

import numpy as np


class TestCouplingHarmonicEstimation:
    """Verify harmonic coupling estimation returns structured results."""

    def test_returns_harmonic_components(self):
        from scpn_phase_orchestrator.autotune.coupling_est import (
            estimate_coupling_harmonics,
        )

        rng = np.random.default_rng(42)
        N, T = 4, 200
        phases = rng.uniform(0, 2 * np.pi, (N, T))
        omegas = rng.normal(0, 0.5, N)
        result = estimate_coupling_harmonics(phases, omegas, dt=0.01, n_harmonics=2)
        assert isinstance(result, dict)
        assert "sin_1" in result
        assert "cos_1" in result

    def test_harmonic_shapes_match_n(self):
        from scpn_phase_orchestrator.autotune.coupling_est import (
            estimate_coupling_harmonics,
        )

        N = 6
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, (N, 100))
        omegas = rng.normal(0, 0.5, N)
        result = estimate_coupling_harmonics(phases, omegas, dt=0.01, n_harmonics=1)
        assert result["sin_1"].shape == (N, N)
