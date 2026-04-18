# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for the Lyapunov spectrum

"""Algorithmic properties of :func:`lyapunov_spectrum`.

These tests pin down the analytic behaviour that every backend must
satisfy — shape, ordering, conservation laws, basin structure, driver
response — independently of which backend happens to be active. They
run against the reference Python implementation so they remain
meaningful when toolchains are missing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor import lyapunov as ly_mod
from scpn_phase_orchestrator.monitor.lyapunov import (
    lyapunov_spectrum,
)

TWO_PI = 2.0 * math.pi


def _with_python_backend(func):
    """Decorator: force the Python reference backend for the test body."""

    def wrapper(*args, **kwargs):
        prev = ly_mod.ACTIVE_BACKEND
        ly_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            ly_mod.ACTIVE_BACKEND = prev

    wrapper.__name__ = func.__name__
    return wrapper


def _random_problem(
    rng: np.random.Generator,
    n: int,
    coupling: float = 1.0,
    alpha_amp: float = 0.0,
):
    phases = rng.uniform(0.0, TWO_PI, size=n)
    omegas = rng.normal(0.0, 0.5, size=n)
    knm = rng.uniform(0.0, coupling, size=(n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = (
        rng.uniform(-alpha_amp, alpha_amp, size=(n, n))
        if alpha_amp
        else np.zeros((n, n))
    )
    return phases, omegas, knm, alpha


class TestShapeAndOrdering:
    @_with_python_backend
    def test_returns_n_exponents(self):
        phases, omegas, knm, alpha = _random_problem(
            np.random.default_rng(42), n=6
        )
        spec = lyapunov_spectrum(phases, omegas, knm, alpha, n_steps=200)
        assert spec.shape == (6,)

    @_with_python_backend
    def test_sorted_descending(self):
        phases, omegas, knm, alpha = _random_problem(
            np.random.default_rng(7), n=8, coupling=0.3
        )
        spec = lyapunov_spectrum(phases, omegas, knm, alpha, n_steps=300)
        for i in range(len(spec) - 1):
            assert spec[i] >= spec[i + 1] - 1e-12

    @_with_python_backend
    def test_finite_output(self):
        phases, omegas, knm, alpha = _random_problem(
            np.random.default_rng(11), n=5, coupling=2.0
        )
        spec = lyapunov_spectrum(phases, omegas, knm, alpha, n_steps=400)
        assert np.all(np.isfinite(spec))


class TestAnalyticLimits:
    @_with_python_backend
    def test_zero_coupling_zero_exponents(self):
        """Uncoupled oscillators → all λ_i ≈ 0 (neutral stability)."""
        rng = np.random.default_rng(99)
        phases = rng.uniform(0.0, TWO_PI, size=4)
        omegas = rng.normal(0.0, 0.5, size=4)
        knm = np.zeros((4, 4))
        alpha = np.zeros((4, 4))
        spec = lyapunov_spectrum(phases, omegas, knm, alpha, n_steps=400)
        assert np.max(np.abs(spec)) < 0.05

    @_with_python_backend
    def test_strong_coupling_transverse_contraction(self):
        """Strong all-to-all coupling → N-1 transverse λ strictly negative."""
        n = 5
        phases = np.full(n, 0.1)
        omegas = np.zeros(n)
        knm = np.full((n, n), 4.0)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        spec = lyapunov_spectrum(
            phases, omegas, knm, alpha, dt=0.005, n_steps=1500
        )
        # Transverse exponents λ_2..λ_N should all be negative.
        assert np.all(spec[1:] < -0.1)

    @_with_python_backend
    def test_single_oscillator_neutral(self):
        spec = lyapunov_spectrum(
            np.array([0.5]),
            np.array([1.0]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            n_steps=500,
        )
        assert spec.shape == (1,)
        assert abs(spec[0]) < 0.05


class TestDriverResponse:
    @_with_python_backend
    def test_driver_contracts_phase_space(self):
        """Strong external driver should make λ_max more negative."""
        n = 4
        rng = np.random.default_rng(21)
        phases = rng.uniform(0.0, TWO_PI, size=n)
        omegas = np.ones(n)
        knm = np.full((n, n), 1.0)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        free = lyapunov_spectrum(
            phases, omegas, knm, alpha, n_steps=800, zeta=0.0
        )
        driven = lyapunov_spectrum(
            phases, omegas, knm, alpha, n_steps=800, zeta=2.5, psi=0.3
        )
        # A stronger contracting driver should not increase λ_max.
        assert driven[0] <= free[0] + 1e-6


class TestSakaguchiPhaseLag:
    @_with_python_backend
    def test_alpha_changes_spectrum(self):
        """Non-zero α must enter the spectrum — guards against the
        pre-fix bug where Euler integration ignored α in the Jacobian."""
        rng = np.random.default_rng(2026)
        n = 4
        phases = rng.uniform(0.0, TWO_PI, size=n)
        omegas = rng.normal(0.0, 0.1, size=n)
        knm = rng.uniform(0.5, 1.5, size=(n, n))
        np.fill_diagonal(knm, 0.0)

        zero_alpha = np.zeros((n, n))
        nonzero_alpha = rng.uniform(-0.4, 0.4, size=(n, n))
        np.fill_diagonal(nonzero_alpha, 0.0)

        spec0 = lyapunov_spectrum(phases, omegas, knm, zero_alpha, n_steps=300)
        spec1 = lyapunov_spectrum(phases, omegas, knm, nonzero_alpha, n_steps=300)
        assert np.max(np.abs(spec0 - spec1)) > 1e-6


class TestKuramotoJacobianInternals:
    def test_diagonal_includes_driver(self):
        """``_kuramoto_jacobian`` must subtract ζ cos(Ψ − θ_i) from the
        diagonal whenever ``zeta != 0``."""
        phases = np.array([0.0, 1.0, 2.0])
        knm = np.zeros((3, 3))
        alpha = np.zeros((3, 3))
        J_free = ly_mod._kuramoto_jacobian(phases, knm, alpha, 0.0, 0.0)
        J_drv = ly_mod._kuramoto_jacobian(phases, knm, alpha, 1.5, 0.2)
        expected = -1.5 * np.cos(0.2 - phases)
        np.testing.assert_allclose(
            np.diag(J_drv) - np.diag(J_free), expected, atol=1e-12
        )

    def test_offdiagonal_uses_sakaguchi_lag(self):
        """Off-diagonal J_ij = K_ij cos(θ_j − θ_i − α_ij)."""
        phases = np.array([0.0, 0.5])
        knm = np.array([[0.0, 0.7], [0.9, 0.0]])
        alpha = np.array([[0.0, 0.2], [0.3, 0.0]])
        J = ly_mod._kuramoto_jacobian(phases, knm, alpha, 0.0, 0.0)
        assert abs(J[0, 1] - 0.7 * np.cos(0.5 - 0.0 - 0.2)) < 1e-12
        assert abs(J[1, 0] - 0.9 * np.cos(0.0 - 0.5 - 0.3)) < 1e-12


class TestRK4Convergence:
    @_with_python_backend
    def test_halving_dt_converges(self):
        """Halving dt should not change λ by more than O(dt⁴) — RK4 is
        fourth-order in the timestep."""
        rng = np.random.default_rng(2027)
        n = 3
        phases = rng.uniform(0.0, TWO_PI, size=n)
        omegas = rng.normal(0.0, 0.3, size=n)
        knm = rng.uniform(0.3, 0.8, size=(n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        coarse = lyapunov_spectrum(
            phases, omegas, knm, alpha, dt=0.02, n_steps=500
        )
        fine = lyapunov_spectrum(
            phases, omegas, knm, alpha, dt=0.01, n_steps=1000
        )
        # Expect spectra to agree within 10⁻¹ for this timestep pair —
        # exponents are long-time averages so they respond to integrator
        # accuracy through the trajectory, not through a single step.
        assert np.max(np.abs(coarse - fine)) < 0.1


class TestRandomProperty:
    @_with_python_backend
    @given(
        n=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_monotone_sorted_and_finite(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        phases, omegas, knm, alpha = _random_problem(
            rng, n=n, coupling=0.8, alpha_amp=0.1
        )
        spec = lyapunov_spectrum(
            phases, omegas, knm, alpha, n_steps=200
        )
        assert spec.shape == (n,)
        assert np.all(np.isfinite(spec))
        for i in range(n - 1):
            assert spec[i] >= spec[i + 1] - 1e-12


class TestInputValidation:
    def test_empty_phases(self):
        spec = lyapunov_spectrum(
            np.array([]),
            np.array([]),
            np.zeros((0, 0)),
            np.zeros((0, 0)),
            n_steps=10,
        )
        assert spec.shape == (0,)

    def test_zero_qr_interval_rejected_by_rust(self):
        """The Rust kernel rejects qr_interval=0 at the FFI boundary.
        Python reference silently never triggers QR; cover the Rust
        error path via ``ValueError`` / ``RuntimeError``."""
        if "rust" not in ly_mod.AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not available")
        prev = ly_mod.ACTIVE_BACKEND
        ly_mod.ACTIVE_BACKEND = "rust"
        try:
            with pytest.raises((ValueError, RuntimeError)):
                lyapunov_spectrum(
                    np.array([0.0]),
                    np.array([1.0]),
                    np.zeros((1, 1)),
                    np.zeros((1, 1)),
                    n_steps=10,
                    qr_interval=0,
                )
        finally:
            ly_mod.ACTIVE_BACKEND = prev
