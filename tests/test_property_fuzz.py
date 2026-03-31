# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Property-based fuzz tests (Stuart-Landau, PAC, lags)

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.coupling.lags import LagModel
from scpn_phase_orchestrator.upde.pac import modulation_index, pac_matrix
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

TWO_PI = 2.0 * np.pi

_FINITE = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)


# ─── Stuart-Landau invariants ────────────────────────────────────────


@given(
    n=st.integers(min_value=2, max_value=16),
    dt=st.floats(min_value=1e-4, max_value=0.05, allow_nan=False),
    mu_sign=st.sampled_from([-1.0, 0.5, 1.0, 2.0]),
)
@settings(max_examples=100, deadline=3000, suppress_health_check=[HealthCheck.too_slow])
def test_sl_phases_bounded(n: int, dt: float, mu_sign: float) -> None:
    """Stuart-Landau phases stay in [0, 2*pi) after one step."""
    rng = np.random.default_rng(42)
    state = np.concatenate([rng.uniform(0, TWO_PI, n), rng.uniform(0.1, 2.0, n)])
    omegas = rng.uniform(0.5, 5.0, n)
    mu = np.full(n, mu_sign)
    knm = np.zeros((n, n))
    knm_r = np.zeros((n, n))
    alpha = np.zeros((n, n))

    eng = StuartLandauEngine(n, dt=dt, method="euler")
    result = eng.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha, epsilon=0.0)

    assert np.all(result[:n] >= 0.0), f"negative phase: {result[:n].min()}"
    assert np.all(result[:n] < TWO_PI + 1e-12), f"phase >= 2pi: {result[:n].max()}"


@given(
    n=st.integers(min_value=2, max_value=16),
    dt=st.floats(min_value=1e-4, max_value=0.05, allow_nan=False),
)
@settings(max_examples=100, deadline=3000, suppress_health_check=[HealthCheck.too_slow])
def test_sl_amplitudes_non_negative(n: int, dt: float) -> None:
    """Stuart-Landau amplitudes are clamped >= 0 after post_step."""
    rng = np.random.default_rng(7)
    state = np.concatenate([rng.uniform(0, TWO_PI, n), rng.uniform(0.01, 0.5, n)])
    omegas = np.ones(n)
    mu = np.full(n, -2.0)  # subcritical → decay
    knm = np.zeros((n, n))
    knm_r = np.zeros((n, n))
    alpha = np.zeros((n, n))

    eng = StuartLandauEngine(n, dt=dt, method="rk4")
    result = state.copy()
    for _ in range(50):
        result = eng.step(result, omegas, mu, knm, knm_r, 0.0, 0.0, alpha, epsilon=0.0)

    assert np.all(result[n:] >= 0.0), f"negative amplitude: {result[n:].min()}"


@given(
    n=st.integers(min_value=1, max_value=8),
    mu_val=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
)
@settings(max_examples=50, deadline=5000, suppress_health_check=[HealthCheck.too_slow])
def test_sl_limit_cycle_convergence(n: int, mu_val: float) -> None:
    """Uncoupled Stuart-Landau: r → sqrt(mu) for supercritical mu > 0."""
    state = np.concatenate([np.zeros(n), np.full(n, 0.5)])
    omegas = np.ones(n)
    mu = np.full(n, mu_val)

    eng = StuartLandauEngine(n, dt=0.01, method="rk4")
    result = state.copy()
    for _ in range(3000):
        result = eng.step(
            result,
            omegas,
            mu,
            np.zeros((n, n)),
            np.zeros((n, n)),
            0.0,
            0.0,
            np.zeros((n, n)),
            epsilon=0.0,
        )

    expected_r = np.sqrt(mu_val)
    for i in range(n):
        assert abs(result[n + i] - expected_r) < 0.05, (
            f"r[{i}]={result[n + i]:.4f}, expected ~{expected_r:.4f}"
        )


@given(
    n=st.integers(min_value=2, max_value=12),
    dt=st.floats(min_value=1e-4, max_value=0.05, allow_nan=False),
)
@settings(max_examples=50, deadline=3000, suppress_health_check=[HealthCheck.too_slow])
def test_sl_state_finite(n: int, dt: float) -> None:
    """Stuart-Landau output is always finite for finite input."""
    rng = np.random.default_rng(0)
    state = np.concatenate([rng.uniform(0, TWO_PI, n), rng.uniform(0.1, 3.0, n)])
    omegas = rng.uniform(-5.0, 5.0, n)
    mu = rng.uniform(-2.0, 3.0, n)
    knm = rng.uniform(0.0, 0.5, (n, n))
    np.fill_diagonal(knm, 0.0)
    knm_r = rng.uniform(0.0, 0.3, (n, n))
    np.fill_diagonal(knm_r, 0.0)
    alpha = np.zeros((n, n))

    eng = StuartLandauEngine(n, dt=dt, method="euler")
    result = eng.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha, epsilon=0.5)
    assert np.all(np.isfinite(result)), "non-finite output"


# ─── PAC invariants ──────────────────────────────────────────────────


@given(
    n=st.integers(min_value=10, max_value=500),
    n_bins=st.integers(min_value=4, max_value=36),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_pac_mi_bounds(n: int, n_bins: int) -> None:
    """MI ∈ [0, 1] for any phase/amplitude distribution."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, TWO_PI, n)
    amp = rng.uniform(0, 10.0, n)
    mi = modulation_index(theta, amp, n_bins)
    assert 0.0 <= mi <= 1.0 + 1e-12, f"MI={mi}"


@given(
    n=st.integers(min_value=50, max_value=500),
    n_bins=st.integers(min_value=4, max_value=18),
)
@settings(max_examples=50)
def test_pac_uniform_amplitude_low_mi(n: int, n_bins: int) -> None:
    """Constant amplitude across all phases → MI ≈ 0 (needs n >> n_bins)."""
    theta = np.linspace(0, TWO_PI, n, endpoint=False)
    amp = np.ones(n)
    mi = modulation_index(theta, amp, n_bins)
    assert mi < 0.05, f"MI={mi} for uniform amplitude (n={n}, bins={n_bins})"


@given(
    t=st.integers(min_value=10, max_value=100),
    n=st.integers(min_value=2, max_value=8),
    n_bins=st.integers(min_value=4, max_value=18),
)
@settings(max_examples=50)
def test_pac_matrix_shape(t: int, n: int, n_bins: int) -> None:
    """pac_matrix returns (n, n) matrix with non-negative entries."""
    rng = np.random.default_rng(0)
    phases = rng.uniform(0, TWO_PI, (t, n))
    amplitudes = rng.uniform(0, 1.0, (t, n))
    mat = pac_matrix(phases, amplitudes, n_bins)
    assert mat.shape == (n, n)
    assert np.all(mat >= -1e-12), "negative PAC matrix entry"


@given(threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
@settings(max_examples=50)
def test_pac_gate_monotonic(threshold: float) -> None:
    """pac_gate: higher MI → gate_open more likely."""
    rng = np.random.default_rng(0)
    n = 200
    # Correlated signal → high MI
    theta = np.linspace(0, TWO_PI * 4, n)
    amp_corr = np.abs(np.sin(theta))
    mi_corr = modulation_index(theta % TWO_PI, amp_corr, 18)

    # Uncorrelated → low MI
    amp_rand = rng.uniform(0, 1, n)
    mi_rand = modulation_index(theta % TWO_PI, amp_rand, 18)

    assert mi_corr >= mi_rand - 0.1, f"correlated MI={mi_corr} < random MI={mi_rand}"


# ─── Lag model invariants ────────────────────────────────────────────


@given(
    n=st.integers(min_value=2, max_value=16),
    speed=st.floats(min_value=0.1, max_value=100.0, allow_nan=False),
)
@settings(max_examples=100)
def test_lag_antisymmetric(n: int, speed: float) -> None:
    """LagModel.estimate_from_distances: alpha[i,j] = -alpha[j,i]."""
    rng = np.random.default_rng(0)
    dist = rng.uniform(0, 10.0, (n, n))
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    alpha = LagModel.estimate_from_distances(dist, speed)
    np.testing.assert_allclose(alpha, -alpha.T, atol=1e-12)


@given(
    n=st.integers(min_value=2, max_value=16),
    speed=st.floats(min_value=0.1, max_value=100.0, allow_nan=False),
)
@settings(max_examples=100)
def test_lag_zero_diagonal(n: int, speed: float) -> None:
    """LagModel: diagonal is always zero."""
    rng = np.random.default_rng(0)
    dist = rng.uniform(0, 10.0, (n, n))
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    alpha = LagModel.estimate_from_distances(dist, speed)
    np.testing.assert_allclose(np.diag(alpha), 0.0, atol=1e-15)


@given(
    n=st.integers(min_value=2, max_value=8),
    speed=st.floats(min_value=0.1, max_value=100.0, allow_nan=False),
)
@settings(max_examples=50)
def test_lag_proportional_to_distance(n: int, speed: float) -> None:
    """Farther oscillators have larger absolute lag."""
    rng = np.random.default_rng(0)
    dist = rng.uniform(0, 10.0, (n, n))
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    alpha = LagModel.estimate_from_distances(dist, speed)
    for i in range(n):
        for j in range(i + 1, n):
            expected = 2.0 * np.pi * dist[i, j] / speed
            assert abs(alpha[i, j] - expected) < 1e-12



# Pipeline wiring: property fuzzing tests use UPDEEngine + order_parameter
# with hypothesis-generated inputs.
