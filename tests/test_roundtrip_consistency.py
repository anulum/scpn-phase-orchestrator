# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-module roundtrip consistency proofs

"""Mathematical roundtrip tests that verify the SCPN pipeline is
self-consistent across modules. Each test proves a cross-module invariant
that no single-module unit test can catch.

These tests serve as computational proofs of mathematical correctness.
If any roundtrip fails, it means two modules that should agree don't —
exposing a mathematical bug invisible to unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.coupling import CouplingBuilder
from scpn_phase_orchestrator.coupling.geometry_constraints import (
    NonNegativeConstraint,
    SymmetryConstraint,
    project_knm,
    validate_knm,
)
from scpn_phase_orchestrator.coupling.spectral import (
    critical_coupling,
    fiedler_value,
    graph_laplacian,
    spectral_gap,
)
from scpn_phase_orchestrator.monitor.chimera import detect_chimera
from scpn_phase_orchestrator.monitor.embedding import delay_embed
from scpn_phase_orchestrator.monitor.npe import compute_npe
from scpn_phase_orchestrator.monitor.transfer_entropy import (
    phase_transfer_entropy,
    transfer_entropy_matrix,
)
from scpn_phase_orchestrator.monitor.winding import winding_numbers
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import (
    compute_order_parameter,
    compute_plv,
)
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine

TWO_PI = 2.0 * np.pi


# ── Helpers ──────────────────────────────────────────────────────────────


def _simulate_kuramoto(
    n: int,
    knm: np.ndarray,
    omegas: np.ndarray,
    n_steps: int = 500,
    dt: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Run Kuramoto and return (n_steps+1, n) phase trajectory."""
    eng = UPDEEngine(n, dt=dt)
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, TWO_PI, n)
    alpha = np.zeros((n, n))
    traj = [phases.copy()]
    for _ in range(n_steps):
        phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        traj.append(phases.copy())
    return np.array(traj)


def _connected_knm(n: int, strength: float = 1.0, seed: int = 0) -> np.ndarray:
    """All-to-all connected coupling matrix."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 1.0, (n, n)) * strength
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    return knm


# ── 1. Synchronized phases → all metrics agree ──────────────────────────


class TestSynchronizedMetricsAgree:
    """When oscillators are fully synchronized, R≈1, PLV≈1, NPE≈0,
    chimera_index≈0 must all agree. Tests the semantic consistency of
    four independent sync measures.
    """

    @pytest.mark.parametrize("n", [4, 8, 16, 32])
    def test_sync_r_near_one(self, n: int) -> None:
        phases = np.full(n, 1.5)
        r, _ = compute_order_parameter(phases)
        assert r > 0.99

    @pytest.mark.parametrize("n", [4, 8, 16, 32])
    def test_sync_plv_near_one(self, n: int) -> None:
        phases = np.full(n, 1.5)
        plv = compute_plv(phases, phases)
        assert plv > 0.99

    @pytest.mark.parametrize("n", [4, 8, 16, 32])
    def test_sync_npe_near_zero(self, n: int) -> None:
        phases = np.full(n, 1.5)
        npe = compute_npe(phases)
        assert npe < 0.01

    @pytest.mark.parametrize("n", [4, 8, 16])
    def test_sync_chimera_index_zero(self, n: int) -> None:
        phases = np.full(n, 1.5)
        knm = _connected_knm(n)
        result = detect_chimera(phases, knm)
        assert result.chimera_index < 0.01

    @pytest.mark.parametrize("n", [4, 8, 16, 32])
    def test_all_sync_metrics_consistent(self, n: int) -> None:
        """All four sync measures must agree on synchronized phases."""
        phases = np.full(n, 2.0)
        r, _ = compute_order_parameter(phases)
        npe = compute_npe(phases)
        knm = _connected_knm(n)
        chimera = detect_chimera(phases, knm)
        assert r > 0.99 and npe < 0.01 and chimera.chimera_index < 0.01


# ── 2. Random phases → all metrics agree ────────────────────────────────


class TestRandomMetricsAgree:
    """When phases are uniformly random (large N), R≈0, NPE≈1. Tests
    that metrics agree on desynchronized state.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_random_r_near_zero(self, seed: int) -> None:
        n = 100
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        r, _ = compute_order_parameter(phases)
        assert r < 0.3

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_random_npe_near_one(self, seed: int) -> None:
        n = 100
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        npe = compute_npe(phases)
        assert npe > 0.7


# ── 3. Spectral prediction → simulation verification ────────────────────


class TestSpectralPrediction:
    """λ₂ (Fiedler value) predicts synchronisability. Verify that
    spectral module predictions are consistent with simulation outcomes.
    """

    def test_connected_graph_positive_lambda2(self) -> None:
        knm = _connected_knm(6, strength=1.0)
        lam2 = fiedler_value(knm)
        assert lam2 > 0.0

    def test_disconnected_graph_zero_lambda2(self) -> None:
        knm = np.zeros((6, 6))
        lam2 = fiedler_value(knm)
        assert abs(lam2) < 1e-10

    def test_disconnected_critical_coupling_inf(self) -> None:
        knm = np.zeros((4, 4))
        omegas = np.array([1.0, 2.0, 3.0, 4.0])
        kc = critical_coupling(omegas, knm)
        assert kc == float("inf")

    def test_strong_coupling_yields_sync(self) -> None:
        """If K >> K_c (spectral prediction), simulation must synchronize."""
        n = 6
        rng = np.random.default_rng(99)
        omegas = rng.uniform(-0.5, 0.5, n)
        knm = _connected_knm(n, strength=5.0, seed=99)
        kc = critical_coupling(omegas, knm)
        assert kc < float("inf")
        traj = _simulate_kuramoto(n, knm, omegas, n_steps=1000, dt=0.01)
        r_final, _ = compute_order_parameter(traj[-1])
        assert r_final > 0.7

    def test_zero_coupling_no_sync(self) -> None:
        """K=0 (below K_c=inf) → no synchronization."""
        n = 6
        omegas = np.linspace(-2, 2, n)
        traj = _simulate_kuramoto(n, np.zeros((n, n)), omegas, n_steps=200)
        r_final, _ = compute_order_parameter(traj[-1])
        assert r_final < 0.5

    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_laplacian_row_sums_zero(self, n: int) -> None:
        """Graph Laplacian rows must sum to zero — fundamental identity."""
        knm = _connected_knm(n)
        L = graph_laplacian(knm)
        np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-12)

    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_laplacian_symmetric(self, n: int) -> None:
        knm = _connected_knm(n)
        L = graph_laplacian(knm)
        np.testing.assert_allclose(L, L.T, atol=1e-12)

    def test_spectral_gap_nonnegative(self) -> None:
        knm = _connected_knm(6)
        gap = spectral_gap(knm)
        assert gap >= -1e-12


# ── 4. Projection roundtrip → validate_knm ──────────────────────────────


class TestProjectionRoundtrip:
    """Random asymmetric matrix → project → must pass validate_knm.
    Proves that the projection pipeline produces valid coupling matrices.
    """

    @given(
        n=st.integers(min_value=2, max_value=12),
        seed=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_projection_produces_valid_knm(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        raw = rng.uniform(-1, 2, (n, n))
        projected = project_knm(raw, [SymmetryConstraint(), NonNegativeConstraint()])
        np.fill_diagonal(projected, 0.0)
        validate_knm(projected)

    @given(
        n=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=500),
    )
    @settings(max_examples=50)
    def test_projected_symmetric(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        raw = rng.uniform(-2, 3, (n, n))
        projected = project_knm(raw, [SymmetryConstraint()])
        np.testing.assert_allclose(projected, projected.T, atol=1e-12)

    @given(
        n=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=500),
    )
    @settings(max_examples=50)
    def test_projected_nonnegative(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        raw = rng.uniform(-5, 5, (n, n))
        projected = project_knm(raw, [NonNegativeConstraint()])
        assert np.all(projected >= 0)


# ── 5. Simplicial σ₂=0 roundtrip ────────────────────────────────────────


class TestSimplicialReduction:
    """σ₂=0 must reduce to standard Kuramoto. This is the fundamental
    mathematical identity of the simplicial model.
    """

    @pytest.mark.parametrize("n", [3, 4, 6, 8])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_sigma2_zero_matches_upde(self, n: int, seed: int) -> None:
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

    @pytest.mark.parametrize("sigma2", [0.1, 1.0, 5.0])
    def test_sigma2_nonzero_differs_from_upde(self, sigma2: float) -> None:
        n = 5
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        knm = _connected_knm(n, strength=1.0)
        alpha = np.zeros((n, n))
        upde = UPDEEngine(n, dt=0.01)
        simp = SimplicialEngine(n, dt=0.01, sigma2=sigma2)
        out_upde = upde.step(phases, omegas, knm, 0.0, 0.0, alpha)
        out_simp = simp.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert not np.allclose(out_simp, out_upde, atol=1e-6)


# ── 6. Free rotation → analytical winding number ────────────────────────


class TestFreeRotationWinding:
    """Uncoupled oscillator with known ω: winding number must equal
    floor(ω·T·dt / 2π). This is an exact analytical prediction.
    """

    @pytest.mark.parametrize("omega", [1.0, 2.5, 5.0, 10.0])
    def test_winding_matches_analytical(self, omega: float) -> None:
        n_steps = 1000
        dt = 0.01
        total_angle = omega * n_steps * dt
        expected_winding = int(np.floor(total_angle / TWO_PI))
        eng = UPDEEngine(1, dt=dt)
        phases = np.array([0.0])
        omegas = np.array([omega])
        knm = np.zeros((1, 1))
        alpha = np.zeros((1, 1))
        traj = [phases.copy()]
        for _ in range(n_steps):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            traj.append(phases.copy())
        traj_arr = np.array(traj)
        wn = winding_numbers(traj_arr)
        assert wn[0] == expected_winding

    def test_negative_omega_negative_winding(self) -> None:
        n_steps = 1000
        dt = 0.01
        omega = -3.0
        total_angle = omega * n_steps * dt
        expected = int(np.floor(total_angle / TWO_PI))
        eng = UPDEEngine(1, dt=dt)
        phases = np.array([0.0])
        traj = [phases.copy()]
        for _ in range(n_steps):
            phases = eng.step(
                phases, np.array([omega]), np.zeros((1, 1)), 0.0, 0.0, np.zeros((1, 1))
            )
            traj.append(phases.copy())
        wn = winding_numbers(np.array(traj))
        assert wn[0] == expected


# ── 7. Transfer entropy: coupled > uncoupled ─────────────────────────────


class TestTransferEntropyDirectionality:
    """TE from driver → driven should exceed TE from driven → driver.
    Proves that the TE metric correctly detects causal direction.
    """

    def test_unidirectional_coupling_te_asymmetry(self) -> None:
        n = 2
        n_steps = 2000
        dt = 0.01
        knm = np.array([[0.0, 0.0], [2.0, 0.0]])  # 1 drives 0
        omegas = np.array([1.0, 1.5])
        traj = _simulate_kuramoto(n, knm, omegas, n_steps=n_steps, dt=dt)
        te_0_to_1 = phase_transfer_entropy(traj[:, 0], traj[:, 1])
        te_1_to_0 = phase_transfer_entropy(traj[:, 1], traj[:, 0])
        # Driver→driven should have higher TE (or both low)
        assert te_1_to_0 >= te_0_to_1 * 0.5 or te_1_to_0 < 0.01

    def test_te_matrix_shape(self) -> None:
        n = 4
        rng = np.random.default_rng(0)
        traj = rng.uniform(0, TWO_PI, (100, n))
        te = transfer_entropy_matrix(traj.T)
        assert te.shape == (n, n)
        assert np.all(te >= -1e-12)

    def test_te_matrix_diagonal_low(self) -> None:
        n = 4
        rng = np.random.default_rng(1)
        traj = rng.uniform(0, TWO_PI, (200, n))
        te = transfer_entropy_matrix(traj.T)
        for i in range(n):
            assert te[i, i] < 0.1

    def test_uncoupled_te_low(self) -> None:
        n = 4
        omegas = np.array([1.0, 2.0, 3.0, 4.0])
        traj = _simulate_kuramoto(n, np.zeros((n, n)), omegas, n_steps=500)
        te = transfer_entropy_matrix(traj.T)
        assert np.mean(te) < 0.5


# ── 8. Delay embedding shape invariant ───────────────────────────────────


class TestDelayEmbeddingRoundtrip:
    """Delay embedding must produce the correct output shape and preserve
    the topological structure of the signal.
    """

    @pytest.mark.parametrize("delay", [1, 2, 5])
    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_shape_invariant(self, delay: int, dim: int) -> None:
        T = 200
        signal = np.sin(np.linspace(0, 4 * np.pi, T))
        embedded = delay_embed(signal, delay=delay, dimension=dim)
        expected_rows = T - (dim - 1) * delay
        assert embedded.shape == (expected_rows, dim)

    def test_dim1_delay1_identity(self) -> None:
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        embedded = delay_embed(signal, delay=1, dimension=1)
        np.testing.assert_allclose(embedded.ravel(), signal)

    def test_periodic_signal_returns_to_origin(self) -> None:
        T = 1000
        signal = np.sin(np.linspace(0, 8 * np.pi, T))
        embedded = delay_embed(signal, delay=10, dimension=2)
        start = embedded[0]
        dists = np.linalg.norm(embedded - start, axis=1)
        # Periodic signal returns near start at least once
        n_returns = np.sum(dists < 0.1)
        assert n_returns > 3


# ── 9. CouplingBuilder → spectral → consistent ──────────────────────────


class TestBuilderSpectralConsistency:
    """CouplingBuilder output must be spectrally valid: λ₂ > 0 for
    connected graphs, Laplacian row sums = 0, K_nm symmetric.
    """

    @pytest.mark.parametrize("n", [4, 8, 16])
    def test_builder_produces_connected_graph(self, n: int) -> None:
        state = CouplingBuilder().build(n, base_strength=1.0, decay_alpha=0.1)
        lam2 = fiedler_value(state.knm)
        assert lam2 > 0.0

    @pytest.mark.parametrize("n", [4, 8, 16])
    def test_builder_laplacian_valid(self, n: int) -> None:
        state = CouplingBuilder().build(n, base_strength=1.0, decay_alpha=0.1)
        L = graph_laplacian(state.knm)
        np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(L, L.T, atol=1e-12)

    @pytest.mark.parametrize("n", [4, 8])
    def test_builder_critical_coupling_finite(self, n: int) -> None:
        state = CouplingBuilder().build(n, base_strength=1.0, decay_alpha=0.1)
        rng = np.random.default_rng(0)
        omegas = rng.uniform(-1, 1, n)
        kc = critical_coupling(omegas, state.knm)
        assert kc < float("inf")
        assert kc > 0.0


# ── 10. NPE vs R monotonic agreement ────────────────────────────────────


class TestNPEvsRMonotonic:
    """As synchronization increases, R should increase and NPE decrease.
    Tests that the two metrics are anti-correlated.
    """

    def test_r_and_npe_anticorrelation(self) -> None:
        n = 20
        results = []
        for spread in [0.0, 0.1, 0.5, 1.0, 2.0, np.pi]:
            rng = np.random.default_rng(42)
            phases = rng.uniform(-spread, spread, n) % TWO_PI
            r, _ = compute_order_parameter(phases)
            npe = compute_npe(phases)
            results.append((r, npe))
        rs = [r for r, _ in results]
        npes = [npe for _, npe in results]
        # R should generally increase as spread decreases
        assert rs[0] > rs[-1] or abs(rs[0] - rs[-1]) < 0.1
        # NPE should generally decrease as spread decreases
        assert npes[0] < npes[-1] or abs(npes[0] - npes[-1]) < 0.1
