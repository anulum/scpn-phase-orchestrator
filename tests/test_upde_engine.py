# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

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
    # With zeta=1.0 and psi=pi, derivative = zeta*sin(pi - 0) = 0 at theta=0
    # Use psi=pi/2 so sin(pi/2 - 0) = 1.0 → phases advance
    new_phases = engine.step(phases, omegas, knm, zeta=1.0, psi=np.pi / 2, alpha=alpha)
    assert np.all(new_phases > 0.0)
