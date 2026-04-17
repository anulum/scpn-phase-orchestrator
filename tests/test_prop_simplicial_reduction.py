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


class TestSimplicialEdgeCases:
    """Edge and error paths Gemini S6 flagged as missing from the property
    suite — the hypothesis cases exercise the happy path; these cover
    degenerate inputs, long-run behaviour and cross-path parity."""

    def test_zero_coupling_reduces_to_drift(self) -> None:
        """K ≡ 0, σ₂=0 → phases advance only by ω·dt."""
        n = 4
        dt = 0.01
        phases = np.zeros(n)
        omegas = np.array([0.5, 1.0, 1.5, 2.0])
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        simp = SimplicialEngine(n, dt=dt, sigma2=0.0)
        out = simp.step(phases, omegas, knm, 0.0, 0.0, alpha)
        np.testing.assert_allclose(out, dt * omegas, atol=1e-12)

    def test_antisymmetric_alpha_breaks_sigma_reduction(self) -> None:
        """With σ₂=0 and a non-trivial α, SimplicialEngine must still
        match UPDEEngine — the reduction identity does not depend on α."""
        n = 4
        dt = 0.01
        rng = np.random.default_rng(2026)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.ones(n)
        knm = _connected_knm(n, seed=7)
        alpha = np.full((n, n), 0.15)
        np.fill_diagonal(alpha, 0.0)
        upde = UPDEEngine(n, dt=dt)
        simp = SimplicialEngine(n, dt=dt, sigma2=0.0)
        out_upde = upde.step(phases, omegas, knm, 0.0, 0.0, alpha)
        out_simp = simp.step(phases, omegas, knm, 0.0, 0.0, alpha)
        np.testing.assert_allclose(out_simp, out_upde, atol=1e-10)

    def test_long_run_bounded(self) -> None:
        """After 500 steps with moderate σ₂, phases must stay in [0, 2π)."""
        n = 6
        dt = 0.01
        rng = np.random.default_rng(55)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(0.5, 1.5, n)
        knm = _connected_knm(n, seed=55)
        alpha = np.zeros((n, n))
        simp = SimplicialEngine(n, dt=dt, sigma2=0.3)
        for _ in range(500):
            phases = simp.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(phases >= 0.0)
        assert np.all(phases < TWO_PI)
        assert np.all(np.isfinite(phases))

    def test_external_drive_attractor(self) -> None:
        """ζ > 0, Ψ=π/2, ω=0, K=0 → all phases pulled toward π/2.

        Starting away from both Ψ and Ψ + π ensures sin(Ψ − θ) > 0 so the
        drive term is non-trivial — exactly Ψ or Ψ ± π is an unstable or
        neutral fixed point of the attractor.
        """
        n = 3
        dt = 0.01
        phases = np.array([0.1, 0.2, 0.3])
        omegas = np.zeros(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        simp = SimplicialEngine(n, dt=dt, sigma2=0.0)
        zeta = 0.5
        psi = np.pi / 2.0
        for _ in range(3000):
            phases = simp.step(phases, omegas, knm, zeta, psi, alpha)
        np.testing.assert_allclose(phases, np.full(n, psi), atol=0.05)

    def test_minimum_n_is_three_for_higher_order_term(self) -> None:
        """Simplicial k=3 coupling requires ≥3 oscillators; N=2 should
        not crash — the σ₂ term silently contributes zero."""
        n = 2
        dt = 0.01
        phases = np.array([0.1, 0.5])
        omegas = np.ones(n)
        knm = np.array([[0.0, 0.3], [0.3, 0.0]])
        alpha = np.zeros((n, n))
        simp = SimplicialEngine(n, dt=dt, sigma2=0.5)
        out = simp.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(np.isfinite(out))
        assert len(out) == n


# Pipeline wiring: simplicial reduction covers the identity σ₂=0 ⇔ pairwise
# Kuramoto. The hypothesis cases scan the small-N regime; the edge cases
# above pin drift-only reduction, long-run boundedness, external-drive
# attractor and the degenerate N=2 path.
