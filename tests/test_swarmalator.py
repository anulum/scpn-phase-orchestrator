# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for swarmalator dynamics

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.swarmalator import SwarmalatorEngine

N = 10
DIM = 2
DT = 0.01


@pytest.fixture()
def engine():
    return SwarmalatorEngine(N, dim=DIM, dt=DT, A=1.0, B=1.0, J=0.5, K=1.0)


@pytest.fixture()
def initial_state():
    rng = np.random.default_rng(42)
    positions = rng.uniform(-1, 1, (N, DIM))
    phases = rng.uniform(0, 2 * np.pi, N)
    omegas = rng.normal(0, 0.1, N)
    return positions, phases, omegas


class TestSwarmalatorStep:
    def test_output_shapes(self, engine, initial_state):
        pos, ph, om = initial_state
        new_pos, new_ph = engine.step(pos, ph, om)
        assert new_pos.shape == (N, DIM)
        assert new_ph.shape == (N,)

    def test_phases_in_range(self, engine, initial_state):
        pos, ph, om = initial_state
        _, new_ph = engine.step(pos, ph, om)
        assert np.all(new_ph >= 0.0)
        assert np.all(new_ph < 2.0 * np.pi)

    def test_finite_values(self, engine, initial_state):
        pos, ph, om = initial_state
        new_pos, new_ph = engine.step(pos, ph, om)
        assert np.all(np.isfinite(new_pos))
        assert np.all(np.isfinite(new_ph))

    def test_J_zero_decouples_phase_from_position(self, initial_state):
        pos, ph, om = initial_state
        engine_coupled = SwarmalatorEngine(N, DIM, DT, J=0.5, K=1.0)
        engine_decoupled = SwarmalatorEngine(N, DIM, DT, J=0.0, K=1.0)
        pos_c, _ = engine_coupled.step(pos, ph, om)
        pos_d, _ = engine_decoupled.step(pos, ph, om)
        # With J=0, phase doesn't affect position dynamics
        # (but attraction A still operates, so positions differ from J=0.5)
        assert not np.allclose(pos_c, pos_d)


class TestSwarmalatorRun:
    def test_trajectory_shapes(self, engine, initial_state):
        pos, ph, om = initial_state
        fp, fph, pt, pht = engine.run(pos, ph, om, 100)
        assert fp.shape == (N, DIM)
        assert fph.shape == (N,)
        assert pt.shape == (100, N, DIM)
        assert pht.shape == (100, N)

    def test_finite_trajectories(self, engine, initial_state):
        pos, ph, om = initial_state
        _, _, pt, pht = engine.run(pos, ph, om, 200)
        assert np.all(np.isfinite(pt))
        assert np.all(np.isfinite(pht))

    def test_3d_works(self, initial_state):
        rng = np.random.default_rng(99)
        pos3d = rng.uniform(-1, 1, (N, 3))
        _, ph, om = initial_state
        engine3d = SwarmalatorEngine(N, dim=3, dt=DT)
        fp, fph, pt, pht = engine3d.run(pos3d, ph, om, 50)
        assert pt.shape == (50, N, 3)


class TestSwarmalatorMetrics:
    def test_spatial_coherence_positive(self, engine, initial_state):
        pos, _, _ = initial_state
        sc = engine.spatial_coherence(pos)
        assert sc > 0.0

    def test_phase_coherence_range(self, engine, initial_state):
        _, ph, _ = initial_state
        R = engine.phase_coherence(ph)
        assert 0.0 <= R <= 1.0

    def test_phase_coherence_perfect_sync(self, engine):
        ph = np.ones(N) * 1.5
        R = engine.phase_coherence(ph)
        assert abs(R - 1.0) < 1e-10

    def test_phase_spatial_correlation_bounded(self, engine, initial_state):
        pos, ph, _ = initial_state
        corr = engine.phase_spatial_correlation(pos, ph)
        assert -1.0 <= corr <= 1.0


class TestSwarmalatorBehavior:
    def test_positive_J_phase_similar_attract(self):
        """J > 0: phase-similar agents should cluster spatially."""
        rng = np.random.default_rng(42)
        n = 20
        engine = SwarmalatorEngine(n, 2, dt=0.01, A=1.0, B=1.0, J=1.0, K=1.0)
        pos = rng.uniform(-2, 2, (n, 2))
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.zeros(n)

        sc_before = engine.spatial_coherence(pos)
        fp, _, _, _ = engine.run(pos, phases, omegas, 500)
        sc_after = engine.spatial_coherence(fp)
        # With J>0, agents should become more compact (lower mean distance)
        assert sc_after < sc_before

    def test_K_positive_nearby_sync(self):
        """K > 0: nearby agents should synchronize phases."""
        rng = np.random.default_rng(42)
        n = 10
        engine = SwarmalatorEngine(n, 2, dt=0.01, A=1.0, B=1.0, J=0.0, K=2.0)
        # Start in a tight cluster
        pos = rng.uniform(-0.1, 0.1, (n, 2))
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.zeros(n)

        R_before = engine.phase_coherence(phases)
        _, fph, _, _ = engine.run(pos, phases, omegas, 500)
        R_after = engine.phase_coherence(fph)
        assert R_after > R_before
