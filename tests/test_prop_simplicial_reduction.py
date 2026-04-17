# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Property-based simplicial reduction proofs

"""Hypothesis-driven invariant proofs for the simplicial Kuramoto model.

The key mathematical identity: σ₂=0 reduces to standard pairwise Kuramoto.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine

TWO_PI = 2.0 * np.pi


def _connected_knm(n: int, strength: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 1.0, (n, n)) * strength
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    return knm


class TestSimplicialReductionInvariants:
    @given(
        n=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_sigma2_zero_matches_upde(self, n: int, seed: int) -> None:
        """σ₂=0 must produce identical output to standard Kuramoto."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        knm = _connected_knm(n, seed=seed)
        alpha = np.zeros((n, n))
        upde = UPDEEngine(n, dt=0.01)
        simp = SimplicialEngine(n, dt=0.01, sigma2=0.0)
        out_upde = upde.step(phases, omegas, knm, 0.0, 0.0, alpha)
        out_simp = simp.step(phases, omegas, knm, 0.0, 0.0, alpha)
        np.testing.assert_allclose(out_simp, out_upde, atol=1e-10)

    @given(
        n=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=40, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_output_finite(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        knm = _connected_knm(n, seed=seed)
        alpha = np.zeros((n, n))
        simp = SimplicialEngine(n, dt=0.01, sigma2=1.0)
        out = simp.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(np.isfinite(out))

    @given(
        n=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=40, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_output_length_n(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        knm = _connected_knm(n, seed=seed)
        alpha = np.zeros((n, n))
        simp = SimplicialEngine(n, dt=0.01, sigma2=0.5)
        out = simp.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert len(out) == n

    @pytest.mark.parametrize("sigma2", [0.01, 0.1, 0.5, 1.0, 5.0])
    def test_nonzero_sigma2_differs(self, sigma2: float) -> None:
        """σ₂ ≠ 0 → different from standard Kuramoto."""
        n = 5
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        knm = _connected_knm(n)
        alpha = np.zeros((n, n))
        upde = UPDEEngine(n, dt=0.01)
        simp = SimplicialEngine(n, dt=0.01, sigma2=sigma2)
        out_upde = upde.step(phases, omegas, knm, 0.0, 0.0, alpha)
        out_simp = simp.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert not np.allclose(out_simp, out_upde, atol=1e-6)


# Pipeline wiring: simplicial reduction tests use SimplicialEngine directly.
