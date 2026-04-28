# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Physics benchmark tests

"""Verify that UPDE and Stuart-Landau engines reproduce known analytical results.

References:
  - Strogatz (2000) "From Kuramoto to Crawford": K_c = 2/(pi*g(0)) for
    Lorentzian frequency distribution g(w) with half-width gamma.
  - Pikovsky et al. (2001) "Synchronization": Stuart-Landau limit cycle
    r → sqrt(mu) for mu > 0, r → 0 for mu < 0.
  - Acebrón et al. (2005) Rev. Mod. Phys. 77(1): Kuramoto order parameter
    R → 0 for K < K_c, R > 0 for K > K_c.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

TWO_PI = 2.0 * np.pi


class TestKuramotoSyncThreshold:
    """Strogatz (2000): identical oscillators synchronise above K_c."""

    def test_identical_oscillators_sync(self):
        """N identical oscillators with K > 0 converge to R ≈ 1."""
        n = 16
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.ones(n)
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        engine = UPDEEngine(n, dt=0.01, method="rk4")
        for _ in range(2000):
            phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)

        r, _ = compute_order_parameter(phases)
        assert r > 0.95, f"R={r:.4f}, expected > 0.95 for identical omegas"

    def test_spread_frequencies_weak_coupling_desync(self):
        """Spread frequencies with weak coupling → low R (below threshold)."""
        n = 32
        rng = np.random.default_rng(1)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2.0, 2.0, n)
        knm = np.full((n, n), 0.01)  # very weak
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        engine = UPDEEngine(n, dt=0.01, method="rk4")
        for _ in range(2000):
            phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)

        r, _ = compute_order_parameter(phases)
        assert r < 0.5, f"R={r:.4f}, expected < 0.5 for weak coupling"

    def test_increasing_coupling_increases_r(self):
        """R monotonically increases with K for fixed frequency spread."""
        n = 16
        rng = np.random.default_rng(2)
        init_phases = rng.uniform(0, TWO_PI, n)
        omegas = 1.0 + rng.normal(0, 0.3, n)
        alpha = np.zeros((n, n))

        r_values = []
        for k_val in [0.01, 0.1, 0.5, 1.0, 3.0]:
            knm = np.full((n, n), k_val)
            np.fill_diagonal(knm, 0.0)
            engine = UPDEEngine(n, dt=0.01, method="rk4")
            phases = init_phases.copy()
            for _ in range(3000):
                phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
            r, _ = compute_order_parameter(phases)
            r_values.append(r)

        for i in range(1, len(r_values)):
            assert r_values[i] >= r_values[i - 1] - 0.05, (
                f"R not monotonic: K sequence R={r_values}"
            )

    def test_external_drive_entrains(self):
        """Strong external drive (zeta) entrains oscillators to psi."""
        n = 8
        rng = np.random.default_rng(3)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.ones(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        psi_target = 1.5

        engine = UPDEEngine(n, dt=0.01, method="rk4")
        for _ in range(5000):
            phases = engine.step(phases, omegas, knm, 2.0, psi_target, alpha)

        # All phases should cluster near psi_target (mod 2pi drift from omega)
        r, _ = compute_order_parameter(phases)
        assert r > 0.9, f"R={r:.4f}, expected > 0.9 under strong drive"


class TestStuartLandauPhysics:
    """Pikovsky (2001): Stuart-Landau bifurcation dynamics."""

    @pytest.mark.parametrize("mu", [0.5, 1.0, 2.0, 4.0])
    def test_supercritical_limit_cycle(self, mu):
        """mu > 0 → r converges to sqrt(mu)."""
        n = 1
        state = np.array([0.0, 0.1])
        omegas = np.array([1.0])
        mu_arr = np.array([mu])

        eng = StuartLandauEngine(n, dt=0.01, method="rk4")
        for _ in range(5000):
            state = eng.step(
                state,
                omegas,
                mu_arr,
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                0.0,
                0.0,
                np.zeros((1, 1)),
                epsilon=0.0,
            )

        expected = np.sqrt(mu)
        assert abs(state[1] - expected) < 0.02, (
            f"r={state[1]:.4f}, expected sqrt({mu})={expected:.4f}"
        )

    @pytest.mark.parametrize("mu", [-0.5, -1.0, -2.0])
    def test_subcritical_decay(self, mu):
        """mu < 0 → r decays toward 0."""
        n = 1
        state = np.array([0.0, 1.0])
        omegas = np.array([1.0])
        mu_arr = np.array([mu])

        eng = StuartLandauEngine(n, dt=0.01, method="rk4")
        for _ in range(3000):
            state = eng.step(
                state,
                omegas,
                mu_arr,
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                0.0,
                0.0,
                np.zeros((1, 1)),
                epsilon=0.0,
            )

        assert state[1] < 0.05, f"r={state[1]:.4f}, expected ~0 for mu={mu}"

    def test_amplitude_coupling_sync(self):
        """Amplitude coupling (epsilon > 0) drives amplitudes toward consensus."""
        n = 4
        rng = np.random.default_rng(10)
        theta = rng.uniform(0, 0.3, n)  # nearly synchronised phases
        r = np.array([0.5, 1.0, 1.5, 2.0])  # spread amplitudes
        state = np.concatenate([theta, r])
        omegas = np.ones(n)
        mu = np.ones(n)
        knm = np.full((n, n), 0.3)
        np.fill_diagonal(knm, 0.0)
        knm_r = np.full((n, n), 0.5)
        np.fill_diagonal(knm_r, 0.0)
        alpha = np.zeros((n, n))

        eng = StuartLandauEngine(n, dt=0.01, method="rk4")
        for _ in range(5000):
            state = eng.step(
                state,
                omegas,
                mu,
                knm,
                knm_r,
                0.0,
                0.0,
                alpha,
                epsilon=1.0,
            )

        amps = state[n:]
        spread = amps.max() - amps.min()
        assert spread < 0.3, f"amplitude spread={spread:.4f}, expected < 0.3"

    def test_phase_frequency_relation(self):
        """Uncoupled: phase advances at rate omega per unit time."""
        n = 1
        omega = 3.0
        state = np.array([0.0, 1.0])  # theta=0, r=1 (on limit cycle)
        omegas = np.array([omega])
        mu = np.array([1.0])

        eng = StuartLandauEngine(n, dt=0.001, method="rk4")
        n_steps = 1000  # 1.0 second
        for _ in range(n_steps):
            state = eng.step(
                state,
                omegas,
                mu,
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                0.0,
                0.0,
                np.zeros((1, 1)),
                epsilon=0.0,
            )

        # theta should have advanced by omega * dt * n_steps = 3.0 rad
        expected = (omega * 0.001 * n_steps) % TWO_PI
        actual = state[0]
        diff = min(abs(actual - expected), TWO_PI - abs(actual - expected))
        assert diff < 0.01, f"theta={actual:.4f}, expected={expected:.4f}"

    def test_zero_coupling_independence(self):
        """Zero coupling: each oscillator evolves independently."""
        n = 4
        rng = np.random.default_rng(20)
        state = np.concatenate([rng.uniform(0, TWO_PI, n), np.ones(n)])

        eng_coupled = StuartLandauEngine(n, dt=0.01, method="rk4")
        eng_single = StuartLandauEngine(1, dt=0.01, method="rk4")

        result_coupled = state.copy()
        for _ in range(100):
            result_coupled = eng_coupled.step(
                result_coupled,
                np.ones(n),
                np.ones(n),
                np.zeros((n, n)),
                np.zeros((n, n)),
                0.0,
                0.0,
                np.zeros((n, n)),
                epsilon=0.0,
            )

        for i in range(n):
            single_state = np.array([state[i], state[n + i]])
            for _ in range(100):
                single_state = eng_single.step(
                    single_state,
                    np.array([1.0]),
                    np.array([1.0]),
                    np.zeros((1, 1)),
                    np.zeros((1, 1)),
                    0.0,
                    0.0,
                    np.zeros((1, 1)),
                    epsilon=0.0,
                )
            np.testing.assert_allclose(
                result_coupled[i],
                single_state[0],
                atol=1e-10,
                err_msg=f"oscillator {i} phase differs",
            )
            np.testing.assert_allclose(
                result_coupled[n + i],
                single_state[1],
                atol=1e-10,
                err_msg=f"oscillator {i} amplitude differs",
            )


class TestCouplingBuilderPhysics:
    """Verify coupling matrix properties from Kuramoto theory."""

    def test_distance_decay(self):
        """Coupling decays with distance: K[i,j] decreases as |i-j| grows."""
        n = 8
        cs = CouplingBuilder().build(n, 0.45, 0.3)
        for i in range(n):
            for j in range(i + 2, n):
                assert cs.knm[i, j] <= cs.knm[i, j - 1] + 1e-12

    def test_symmetry_and_positivity(self):
        """Knm is symmetric, non-negative, zero diagonal."""
        n = 16
        cs = CouplingBuilder().build(n, 0.45, 0.3)
        np.testing.assert_allclose(cs.knm, cs.knm.T, atol=1e-12)
        assert np.all(cs.knm >= 0.0)
        np.testing.assert_allclose(np.diag(cs.knm), 0.0, atol=1e-15)

    def test_stronger_base_means_larger_entries(self):
        """Higher base_strength → uniformly larger coupling entries."""
        n = 8
        weak = CouplingBuilder().build(n, 0.1, 0.3)
        strong = CouplingBuilder().build(n, 1.0, 0.3)
        assert np.all(strong.knm >= weak.knm - 1e-12)


class TestChimeraState:
    """Abrams & Strogatz (2004): partial synchronisation in nonlocally
    coupled identical oscillators on a ring."""

    def test_partial_sync_emerges(self):
        """Nonlocal coupling on a ring: some oscillators sync, others drift."""
        n = 32
        rng = np.random.default_rng(99)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.ones(n)

        # Nonlocal ring coupling: each oscillator couples to R nearest
        r_range = n // 4
        knm = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist = min(abs(i - j), n - abs(i - j))
                if 0 < dist <= r_range:
                    knm[i, j] = 0.5
        alpha = np.full((n, n), 1.46)  # Abrams-Strogatz α ≈ π/2 - 0.1
        np.fill_diagonal(alpha, 0.0)

        engine = UPDEEngine(n, dt=0.01, method="rk4")
        for _ in range(5000):
            phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)

        r_global, _ = compute_order_parameter(phases)
        # Chimera: partial sync means 0 < R < 1 (not fully synced)
        assert 0.1 < r_global < 0.95, (
            f"R={r_global:.4f}, expected partial sync (chimera range)"
        )
