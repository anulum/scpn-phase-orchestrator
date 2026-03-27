# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Extreme-scale stress tests

"""Stress tests verifying correctness and stability at production scale.

These tests use large oscillator counts (N=1000+) and long integration
runs (T=10000+) to catch numerical drift, memory leaks, and
scaling-dependent bugs that small-N unit tests miss.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.spectral import fiedler_value, graph_laplacian
from scpn_phase_orchestrator.monitor.npe import compute_npe
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.reduction import OttAntonsenReduction

TWO_PI = 2.0 * np.pi


def _all_to_all_knm(n: int, strength: float = 1.0) -> np.ndarray:
    knm = np.full((n, n), strength / n)
    np.fill_diagonal(knm, 0.0)
    return knm


# ── Large-N correctness ─────────────────────────────────────────────────


class TestLargeN:
    @pytest.mark.slow
    def test_n1000_identical_sync(self) -> None:
        """N=1000 identical oscillators must synchronise."""
        n = 1000
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.zeros(n)
        knm = _all_to_all_knm(n, strength=2.0)
        alpha = np.zeros((n, n))
        for _ in range(1000):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        R, _ = compute_order_parameter(phases)
        assert R > 0.90
        assert np.all(np.isfinite(phases))

    @pytest.mark.slow
    def test_n1000_random_r_bounded(self) -> None:
        """N=1000 random phases → R near 1/√N ≈ 0.032."""
        n = 1000
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        R, _ = compute_order_parameter(phases)
        assert R < 0.15  # 1/√1000 ≈ 0.032, with fluctuation margin

    @pytest.mark.slow
    def test_n1000_npe_finite(self) -> None:
        """NPE computable at N=1000 without OOM or NaN."""
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, 1000)
        npe = compute_npe(phases)
        assert np.isfinite(npe)
        assert 0.0 <= npe <= 1.0

    @pytest.mark.slow
    def test_n512_laplacian_psd(self) -> None:
        """Graph Laplacian PSD at N=512."""
        knm = _all_to_all_knm(512)
        L = graph_laplacian(knm)
        eigs = np.linalg.eigvalsh(L)
        assert eigs[0] > -1e-8
        assert fiedler_value(knm) > 0


# ── Long-run stability ──────────────────────────────────────────────────


class TestLongRun:
    @pytest.mark.slow
    def test_10k_steps_no_drift(self) -> None:
        """10,000 Euler steps: R must not drift to NaN or diverge."""
        n = 16
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-1, 1, n)
        knm = _all_to_all_knm(n, strength=3.0)
        alpha = np.zeros((n, n))
        for _ in range(10_000):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(np.isfinite(phases))
        R, _ = compute_order_parameter(phases)
        assert 0.0 <= R <= 1.0

    @pytest.mark.slow
    def test_50k_steps_r_stable(self) -> None:
        """50,000 steps: R should stabilise, not oscillate wildly."""
        n = 8
        eng = UPDEEngine(n, dt=0.005)
        phases = np.zeros(n)
        omegas = np.linspace(-0.5, 0.5, n)
        knm = _all_to_all_knm(n, strength=5.0)
        alpha = np.zeros((n, n))
        r_history = []
        for step in range(50_000):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            if step % 1000 == 0:
                R, _ = compute_order_parameter(phases)
                r_history.append(R)
        # Last 10 samples should have low variance (settled)
        assert np.std(r_history[-10:]) < 0.1


# ── OA analytical validation ────────────────────────────────────────────


class TestOttAntonsenAnalytical:
    """Compare OA reduction against UPDE simulation on Lorentzian g(ω).

    Analytical: K_c = 2Δ, R_ss = √(1 - 2Δ/K) for K > K_c.
    Acebrón et al. 2005, Rev. Mod. Phys. 77(1):137-185.
    """

    def test_kc_lorentzian(self) -> None:
        """K_c = 2Δ for Lorentzian frequency distribution."""
        delta = 0.5
        oa = OttAntonsenReduction(omega_0=0.0, delta=delta, K=1.0)
        assert abs(oa.K_c - 2 * delta) < 1e-12

    def test_rss_above_kc(self) -> None:
        """R_ss = √(1 - 2Δ/K) for K > K_c."""
        delta, K = 0.5, 4.0
        oa = OttAntonsenReduction(omega_0=0.0, delta=delta, K=K)
        R_ss = oa.steady_state_R()
        expected = np.sqrt(1.0 - 2 * delta / K)
        assert abs(R_ss - expected) < 1e-12

    @pytest.mark.slow
    def test_oa_vs_upde_lorentzian(self) -> None:
        """OA prediction matches UPDE simulation on Lorentzian ω."""
        n = 200
        delta = 0.5
        K = 4.0  # K > K_c = 1.0
        rng = np.random.default_rng(42)
        # Lorentzian(0, Δ) via inverse CDF
        u = rng.uniform(0, 1, n)
        omegas = delta * np.tan(np.pi * (u - 0.5))
        omegas = np.clip(omegas, -20, 20)

        # UPDE simulation
        eng = UPDEEngine(n, dt=0.01)
        phases = rng.uniform(0, TWO_PI, n)
        knm = np.full((n, n), K / n)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(5000):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        R_sim, _ = compute_order_parameter(phases)

        # OA analytical
        oa = OttAntonsenReduction(omega_0=0.0, delta=delta, K=K)
        R_oa = oa.steady_state_R()

        # Should agree within finite-N fluctuations
        assert abs(R_sim - R_oa) < 0.15, (
            f"OA R_ss={R_oa:.3f}, UPDE R={R_sim:.3f}, gap={abs(R_sim - R_oa):.3f}"
        )

    @pytest.mark.slow
    def test_oa_vs_upde_below_kc(self) -> None:
        """Below K_c: both OA and UPDE should give R ≈ 0."""
        n = 200
        delta = 1.0
        K = 1.0  # K_c = 2.0, so K < K_c
        rng = np.random.default_rng(0)
        u = rng.uniform(0, 1, n)
        omegas = delta * np.tan(np.pi * (u - 0.5))
        omegas = np.clip(omegas, -20, 20)

        eng = UPDEEngine(n, dt=0.01)
        phases = rng.uniform(0, TWO_PI, n)
        knm = np.full((n, n), K / n)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(3000):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        R_sim, _ = compute_order_parameter(phases)

        oa = OttAntonsenReduction(omega_0=0.0, delta=delta, K=K)
        R_oa = oa.steady_state_R()  # should be 0.0

        assert R_oa == 0.0
        assert R_sim < 0.3  # finite-N fluctuations, but no bulk sync
