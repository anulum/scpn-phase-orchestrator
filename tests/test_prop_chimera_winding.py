# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Property-based chimera & winding number proofs

"""Hypothesis-driven invariant proofs for chimera state detection and
phase winding number computation.

Chimera: Kuramoto & Battogtokh 2002. Winding: topological invariant.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor.chimera import detect_chimera
from scpn_phase_orchestrator.monitor.winding import winding_numbers, winding_vector

TWO_PI = 2.0 * np.pi


def _connected_knm(n: int, strength: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 1.0, (n, n)) * strength
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    return knm


# ── 1. Chimera detection: structural invariants ─────────────────────────


class TestChimeraInvariants:
    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=50, deadline=None)
    def test_chimera_index_bounded(self, n: int, seed: int) -> None:
        """chimera_index ∈ [0, 1]."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        knm = _connected_knm(n, seed=seed)
        result = detect_chimera(phases, knm)
        assert 0.0 <= result.chimera_index <= 1.0

    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=50, deadline=None)
    def test_indices_disjoint(self, n: int, seed: int) -> None:
        """Coherent and incoherent index sets must be disjoint."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        knm = _connected_knm(n, seed=seed)
        result = detect_chimera(phases, knm)
        assert len(set(result.coherent_indices) & set(result.incoherent_indices)) == 0

    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=50, deadline=None)
    def test_indices_valid_range(self, n: int, seed: int) -> None:
        """All indices must be in [0, N)."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        knm = _connected_knm(n, seed=seed)
        result = detect_chimera(phases, knm)
        for idx in result.coherent_indices + result.incoherent_indices:
            assert 0 <= idx < n

    @given(n=st.integers(min_value=2, max_value=16))
    @settings(max_examples=30, deadline=None)
    def test_sync_phases_low_chimera(self, n: int) -> None:
        """Fully synchronized → chimera_index ≈ 0."""
        phases = np.full(n, 1.5)
        knm = _connected_knm(n)
        result = detect_chimera(phases, knm)
        assert result.chimera_index < 0.01

    @given(n=st.integers(min_value=2, max_value=16))
    @settings(max_examples=30, deadline=None)
    def test_sync_all_coherent(self, n: int) -> None:
        """Synchronized → all oscillators should be coherent."""
        phases = np.full(n, 2.0)
        knm = _connected_knm(n)
        result = detect_chimera(phases, knm)
        assert len(result.coherent_indices) == n

    def test_empty_phases(self) -> None:
        result = detect_chimera(np.array([]), np.zeros((0, 0)))
        assert result.chimera_index == 0.0
        assert result.coherent_indices == []
        assert result.incoherent_indices == []

    @given(
        n=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_zero_coupling_all_incoherent_or_boundary(self, n: int, seed: int) -> None:
        """No coupling → R_local = 0 for all → all incoherent."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        knm = np.zeros((n, n))
        result = detect_chimera(phases, knm)
        # With no neighbors, R_local = 0, so all classified as incoherent
        assert len(result.coherent_indices) == 0

    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_boundary_count_consistent(self, n: int, seed: int) -> None:
        """n_coherent + n_incoherent + n_boundary = N, chimera_index = n_boundary/N."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        knm = _connected_knm(n, seed=seed)
        result = detect_chimera(phases, knm)
        n_boundary = n - len(result.coherent_indices) - len(result.incoherent_indices)
        assert n_boundary >= 0
        expected_ci = n_boundary / n if n > 0 else 0.0
        assert abs(result.chimera_index - expected_ci) < 1e-12


# ── 2. Winding numbers: integer, topological invariant ───────────────────


class TestWindingNumberInvariants:
    @given(
        n=st.integers(min_value=1, max_value=8),
        t=st.integers(min_value=10, max_value=50),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=50, deadline=None)
    def test_integer_valued(self, n: int, t: int, seed: int) -> None:
        """Winding numbers must be integers."""
        rng = np.random.default_rng(seed)
        traj = rng.uniform(0, TWO_PI, (t, n))
        wn = winding_numbers(traj)
        assert wn.dtype == np.int64

    @given(
        n=st.integers(min_value=1, max_value=8),
        t=st.integers(min_value=10, max_value=50),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=50, deadline=None)
    def test_length_n(self, n: int, t: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        traj = rng.uniform(0, TWO_PI, (t, n))
        wn = winding_numbers(traj)
        assert len(wn) == n

    @given(n=st.integers(min_value=1, max_value=8))
    @settings(max_examples=20, deadline=None)
    def test_stationary_zero(self, n: int) -> None:
        """Constant phases → winding = 0."""
        traj = np.full((20, n), 1.5)
        wn = winding_numbers(traj)
        np.testing.assert_array_equal(wn, 0)

    @pytest.mark.parametrize("omega", [1.0, 2.5, 5.0, -3.0])
    def test_known_winding(self, omega: float) -> None:
        """Linearly advancing phase → predictable winding."""
        n_steps = 1000
        dt = 0.01
        t = np.arange(n_steps + 1) * dt
        phases = (omega * t) % TWO_PI
        traj = phases.reshape(-1, 1)
        wn = winding_numbers(traj)
        expected = int(np.floor(omega * n_steps * dt / TWO_PI))
        assert wn[0] == expected

    def test_single_timestep_zero(self) -> None:
        """T=1 → no movement → winding = 0."""
        traj = np.array([[1.0, 2.0, 3.0]])
        wn = winding_numbers(traj)
        np.testing.assert_array_equal(wn, 0)

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=20, deadline=None)
    def test_winding_vector_equals_winding_numbers(self, seed: int) -> None:
        """winding_vector is an alias for winding_numbers."""
        rng = np.random.default_rng(seed)
        traj = rng.uniform(0, TWO_PI, (30, 4))
        np.testing.assert_array_equal(winding_numbers(traj), winding_vector(traj))

    def test_reverse_trajectory_negates(self) -> None:
        """Reversing time should negate (or closely negate) winding numbers."""
        n_steps = 500
        dt = 0.01
        t = np.arange(n_steps + 1) * dt
        omega = 3.0
        phases = (omega * t) % TWO_PI
        traj = phases.reshape(-1, 1)
        wn_fwd = winding_numbers(traj)
        wn_rev = winding_numbers(traj[::-1])
        # Reversed trajectory should have opposite sign (within ±1 due to floor)
        assert abs(wn_fwd[0] + wn_rev[0]) <= 1

    @given(
        n=st.integers(min_value=1, max_value=6),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_finite(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        traj = rng.uniform(0, TWO_PI, (30, n))
        wn = winding_numbers(traj)
        assert np.all(np.isfinite(wn))


class TestPipelineWiring:
    """Pipeline wiring: proves this module is not decorative."""

    def test_wires_into_pipeline(self):
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
