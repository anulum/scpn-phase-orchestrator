# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE engine tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.engine import UPDEEngine

TWO_PI = 2.0 * np.pi


def test_identical_phases_stay_synchronised(sample_knm, sample_omegas):
    n = 8
    engine = UPDEEngine(n_oscillators=n, dt=0.01)
    # All phases identical -> coupling term vanishes, all advance equally
    phases = np.full(n, 1.0)
    # Use identical omegas so they stay together
    omegas = np.full(n, 1.0)
    alpha = np.zeros((n, n))
    for _ in range(100):
        phases = engine.step(phases, omegas, sample_knm, zeta=0.0, psi=0.0, alpha=alpha)
    R, _ = engine.compute_order_parameter(phases)
    np.testing.assert_allclose(R, 1.0, atol=1e-6)


def test_random_phases_converge_under_strong_coupling():
    n = 8
    rng = np.random.default_rng(7)
    phases = rng.uniform(0, TWO_PI, size=n)
    omegas = np.ones(n) * 1.0
    knm = np.full((n, n), 2.0)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    engine = UPDEEngine(n_oscillators=n, dt=0.005)
    R_init, _ = engine.compute_order_parameter(phases)
    for _ in range(500):
        phases = engine.step(phases, omegas, knm, zeta=0.0, psi=0.0, alpha=alpha)
    R_final, _ = engine.compute_order_parameter(phases)
    assert R_final > R_init


def test_rk4_produces_valid_phases():
    n = 4
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, TWO_PI, size=n)
    omegas = rng.uniform(0.5, 2.0, size=n)
    knm = np.full((n, n), 0.3)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    engine = UPDEEngine(n_oscillators=n, dt=0.01, method="rk4")
    for _ in range(200):
        phases = engine.step(phases, omegas, knm, zeta=0.0, psi=0.0, alpha=alpha)
    assert np.all(phases >= 0.0)
    assert np.all(phases < TWO_PI)


def test_zero_coupling_preserves_natural_frequency():
    n = 4
    dt = 0.01
    phases = np.zeros(n)
    omegas = np.array([1.0, 2.0, 3.0, 4.0])
    knm = np.zeros((n, n))
    alpha = np.zeros((n, n))

    engine = UPDEEngine(n_oscillators=n, dt=dt)
    phases = engine.step(phases, omegas, knm, zeta=0.0, psi=0.0, alpha=alpha)
    expected = (omegas * dt) % TWO_PI
    np.testing.assert_allclose(phases, expected, atol=1e-12)


def test_phase_wrapping():
    n = 2
    phases = np.array([TWO_PI - 0.01, TWO_PI - 0.02])
    omegas = np.array([1.0, 1.0])
    knm = np.zeros((n, n))
    alpha = np.zeros((n, n))

    engine = UPDEEngine(n_oscillators=n, dt=0.1)
    new_phases = engine.step(phases, omegas, knm, zeta=0.0, psi=0.0, alpha=alpha)
    assert np.all(new_phases >= 0.0)
    assert np.all(new_phases < TWO_PI)


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        UPDEEngine(n_oscillators=4, dt=0.01, method="adams")


def test_external_drive_zeta_nonzero():
    n = 4
    dt = 0.01
    phases = np.zeros(n)
    omegas = np.zeros(n)
    knm = np.zeros((n, n))
    alpha = np.zeros((n, n))

    engine = UPDEEngine(n_oscillators=n, dt=dt)
    new_phases = engine.step(phases, omegas, knm, zeta=1.0, psi=np.pi / 2, alpha=alpha)
    assert np.all(new_phases > 0.0)


def test_step_shape_mismatch_raises():
    engine = UPDEEngine(n_oscillators=4, dt=0.01)
    ok_phases = np.zeros(4)
    ok_omegas = np.ones(4)
    ok_knm = np.zeros((4, 4))
    ok_alpha = np.zeros((4, 4))

    with pytest.raises(ValueError, match="phases.shape"):
        engine.step(np.zeros(3), ok_omegas, ok_knm, 0.0, 0.0, ok_alpha)
    with pytest.raises(ValueError, match="omegas.shape"):
        engine.step(ok_phases, np.ones(5), ok_knm, 0.0, 0.0, ok_alpha)
    with pytest.raises(ValueError, match="knm.shape"):
        engine.step(ok_phases, ok_omegas, np.zeros((3, 3)), 0.0, 0.0, ok_alpha)
    with pytest.raises(ValueError, match="alpha.shape"):
        engine.step(ok_phases, ok_omegas, ok_knm, 0.0, 0.0, np.zeros((5, 5)))


def test_rk45_produces_valid_phases():
    n = 8
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, TWO_PI, size=n)
    omegas = rng.uniform(0.5, 2.0, size=n)
    knm = np.full((n, n), 0.3)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    engine = UPDEEngine(n_oscillators=n, dt=0.01, method="rk45")
    for _ in range(200):
        phases = engine.step(phases, omegas, knm, zeta=0.0, psi=0.0, alpha=alpha)
    assert np.all(phases >= 0.0)
    assert np.all(phases < TWO_PI)
    assert engine.last_dt > 0.0


def test_rk45_converges_like_rk4():
    """RK45 adaptive stepping grows dt, so we compare synchronization behavior."""
    n = 8
    rng = np.random.default_rng(7)
    phases_init = rng.uniform(0, TWO_PI, size=n)
    omegas = np.ones(n) * 1.0
    knm = np.full((n, n), 2.0)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    engine_rk4 = UPDEEngine(n_oscillators=n, dt=0.005, method="rk4")
    engine_rk45 = UPDEEngine(n_oscillators=n, dt=0.005, method="rk45")

    phases_rk4 = phases_init.copy()
    phases_rk45 = phases_init.copy()
    for _ in range(500):
        phases_rk4 = engine_rk4.step(phases_rk4, omegas, knm, 0.0, 0.0, alpha)
        phases_rk45 = engine_rk45.step(phases_rk45, omegas, knm, 0.0, 0.0, alpha)

    R_rk4, _ = engine_rk4.compute_order_parameter(phases_rk4)
    R_rk45, _ = engine_rk45.compute_order_parameter(phases_rk45)
    assert R_rk4 > 0.8
    assert R_rk45 > 0.8


def test_rk45_accepted_in_method_list():
    UPDEEngine(n_oscillators=4, dt=0.01, method="rk45")
    with pytest.raises(ValueError, match="Unknown method"):
        UPDEEngine(n_oscillators=4, dt=0.01, method="bogus")


def test_rk45_exhausted_retries_returns_valid_phases():
    """When RK45 exhausts all retry attempts, it falls back to the last y5 result."""
    n = 4
    rng = np.random.default_rng(99)
    phases = rng.uniform(0, TWO_PI, size=n)
    omegas = np.ones(n) * 100.0  # large omega → large derivatives → large error
    knm = np.full((n, n), 50.0)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    engine = UPDEEngine(n_oscillators=n, dt=0.01, method="rk45")
    # Set extremely tight tolerances to force rejection
    engine._atol = 1e-15
    engine._rtol = 1e-15
    result = engine.step(phases, omegas, knm, zeta=0.0, psi=0.0, alpha=alpha)
    # Fallback must still produce valid wrapped phases
    assert result.shape == (n,)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)
    assert np.all(result < TWO_PI)
