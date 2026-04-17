# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — FFI parity tests

"""Cross-validate Python engine against Rust FFI when spo_kernel is available."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator._compat import HAS_RUST
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

_HAS_STEPPER = HAS_RUST and hasattr(
    __import__("spo_kernel") if HAS_RUST else None, "PyUPDEStepper"
)
pytestmark = pytest.mark.skipif(
    not _HAS_STEPPER,
    reason="spo_kernel.PyUPDEStepper not available",
)


@pytest.fixture()
def spo():
    import spo_kernel

    return spo_kernel


def test_euler_parity(spo):
    n = 8
    rng = np.random.default_rng(0)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)
    knm = np.zeros((n, n))
    alpha = np.zeros((n, n))
    dt = 0.01

    py_engine = UPDEEngine(n, dt=dt, method="euler")
    py_result = py_engine.step(phases.copy(), omegas, knm, 0.0, 0.0, alpha)

    rust = spo.PyUPDEStepper(n, dt=dt, method="euler")
    rust_result = np.asarray(
        rust.step(phases.copy(), omegas, knm.ravel(), 0.0, 0.0, alpha.ravel())
    )

    np.testing.assert_allclose(py_result, rust_result, atol=1e-10)


def test_rk4_parity(spo):
    n = 8
    rng = np.random.default_rng(1)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n) + rng.normal(0, 0.1, n)
    base = 0.3
    dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
    knm = base * np.exp(-0.3 * dist)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    dt = 0.01

    py_engine = UPDEEngine(n, dt=dt, method="rk4")
    py_phases = phases.copy()
    for _ in range(50):
        py_phases = py_engine.step(py_phases, omegas, knm, 0.0, 0.0, alpha)

    rust = spo.PyUPDEStepper(n, dt=dt, method="rk4")
    rust_phases = phases.copy()
    knm_flat = knm.ravel()
    alpha_flat = alpha.ravel()
    for _ in range(50):
        rust_phases = np.asarray(
            rust.step(rust_phases, omegas, knm_flat, 0.0, 0.0, alpha_flat)
        )

    np.testing.assert_allclose(py_phases, rust_phases, atol=1e-8)


def test_rk45_parity(spo):
    n = 8
    rng = np.random.default_rng(7)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n) + rng.normal(0, 0.1, n)
    base = 0.3
    dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
    knm = base * np.exp(-0.3 * dist)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    dt = 0.01

    py_engine = UPDEEngine(n, dt=dt, method="rk45")
    py_phases = phases.copy()
    for _ in range(50):
        py_phases = py_engine.step(py_phases, omegas, knm, 0.0, 0.0, alpha)

    rust = spo.PyUPDEStepper(n, dt=dt, method="rk45")
    rust_phases = phases.copy()
    knm_flat = knm.ravel()
    alpha_flat = alpha.ravel()
    for _ in range(50):
        rust_phases = np.asarray(
            rust.step(rust_phases, omegas, knm_flat, 0.0, 0.0, alpha_flat)
        )

    np.testing.assert_allclose(py_phases, rust_phases, atol=1e-6)


def test_order_parameter_parity(spo):
    phases = [0.1, 0.2, 0.15, 0.12]
    r_py, _ = compute_order_parameter(np.array(phases))
    r_rust, _ = spo.order_parameter(np.array(phases))
    assert abs(r_py - r_rust) < 1e-10


# ---------------------------------------------------------------------------
# Edge cases and extended parity coverage (S6 strengthening).
# The pre-existing four tests exercise the happy path; the cases below
# cover lag / drive / scale / sign / ordering regimes that diverge between
# integrators when the Rust and Python derivatives drift apart.
# ---------------------------------------------------------------------------


def test_sakaguchi_lag_parity(spo):
    """Non-zero α (Sakaguchi phase lag) must be honoured by both paths."""
    n = 6
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)
    knm = 0.5 * np.ones((n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = np.full((n, n), 0.3)  # constant 0.3 rad lag
    dt = 0.01

    py_engine = UPDEEngine(n, dt=dt, method="rk4")
    py = phases.copy()
    for _ in range(20):
        py = py_engine.step(py, omegas, knm, 0.0, 0.0, alpha)

    rust = spo.PyUPDEStepper(n, dt=dt, method="rk4")
    ru = phases.copy()
    kf, af = knm.ravel(), alpha.ravel()
    for _ in range(20):
        ru = np.asarray(rust.step(ru, omegas, kf, 0.0, 0.0, af))
    np.testing.assert_allclose(py, ru, atol=1e-8)


def test_external_drive_parity(spo):
    """External drive ζ·sin(Ψ − θ) must be applied identically on both paths."""
    n = 4
    rng = np.random.default_rng(99)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.zeros(n)  # isolate the drive contribution
    knm = np.zeros((n, n))
    alpha = np.zeros((n, n))
    dt = 0.01
    zeta, psi = 1.5, np.pi / 3

    py_engine = UPDEEngine(n, dt=dt, method="rk4")
    py = phases.copy()
    for _ in range(30):
        py = py_engine.step(py, omegas, knm, zeta, psi, alpha)

    rust = spo.PyUPDEStepper(n, dt=dt, method="rk4")
    ru = phases.copy()
    for _ in range(30):
        ru = np.asarray(
            rust.step(ru, omegas, knm.ravel(), zeta, psi, alpha.ravel())
        )
    np.testing.assert_allclose(py, ru, atol=1e-8)


def test_negative_coupling_parity(spo):
    """Inhibitory (negative K_ij) coupling drives anti-phase; both paths match."""
    n = 6
    rng = np.random.default_rng(3)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)
    knm = -0.2 * np.ones((n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    dt = 0.01

    py_engine = UPDEEngine(n, dt=dt, method="rk4")
    py = phases.copy()
    for _ in range(30):
        py = py_engine.step(py, omegas, knm, 0.0, 0.0, alpha)

    rust = spo.PyUPDEStepper(n, dt=dt, method="rk4")
    ru = phases.copy()
    for _ in range(30):
        ru = np.asarray(
            rust.step(ru, omegas, knm.ravel(), 0.0, 0.0, alpha.ravel())
        )
    np.testing.assert_allclose(py, ru, atol=1e-8)


def test_asymmetric_coupling_parity(spo):
    """K_ij ≠ K_ji — directed interactions must integrate identically."""
    n = 5
    rng = np.random.default_rng(17)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)
    knm = rng.uniform(0.1, 0.5, (n, n))
    np.fill_diagonal(knm, 0.0)  # no self-coupling; off-diagonal asymmetric
    alpha = np.zeros((n, n))
    dt = 0.005

    py_engine = UPDEEngine(n, dt=dt, method="rk4")
    py = phases.copy()
    for _ in range(40):
        py = py_engine.step(py, omegas, knm, 0.0, 0.0, alpha)

    rust = spo.PyUPDEStepper(n, dt=dt, method="rk4")
    ru = phases.copy()
    for _ in range(40):
        ru = np.asarray(
            rust.step(ru, omegas, knm.ravel(), 0.0, 0.0, alpha.ravel())
        )
    np.testing.assert_allclose(py, ru, atol=1e-8)


def test_large_network_parity(spo):
    """N=64: scale-out stress test beyond the happy-path N=8 cases."""
    n = 64
    rng = np.random.default_rng(2026)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = rng.normal(1.0, 0.05, n)
    dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
    knm = 0.2 * np.exp(-0.15 * dist)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    dt = 0.01

    py_engine = UPDEEngine(n, dt=dt, method="rk4")
    py = phases.copy()
    rust = spo.PyUPDEStepper(n, dt=dt, method="rk4")
    ru = phases.copy()
    kf, af = knm.ravel(), alpha.ravel()
    for _ in range(20):
        py = py_engine.step(py, omegas, knm, 0.0, 0.0, alpha)
        ru = np.asarray(rust.step(ru, omegas, kf, 0.0, 0.0, af))
    np.testing.assert_allclose(py, ru, atol=1e-7)


def test_zero_phases_remain_near_zero_parity(spo):
    """θ = 0, ω = 0, K = 0 → both paths must stay at zero (degenerate case)."""
    n = 4
    phases = np.zeros(n)
    omegas = np.zeros(n)
    knm = np.zeros((n, n))
    alpha = np.zeros((n, n))
    dt = 0.01

    py_engine = UPDEEngine(n, dt=dt, method="rk4")
    py = py_engine.step(phases.copy(), omegas, knm, 0.0, 0.0, alpha)

    rust = spo.PyUPDEStepper(n, dt=dt, method="rk4")
    ru = np.asarray(
        rust.step(phases.copy(), omegas, knm.ravel(), 0.0, 0.0, alpha.ravel())
    )

    np.testing.assert_allclose(py, np.zeros(n), atol=1e-12)
    np.testing.assert_allclose(ru, np.zeros(n), atol=1e-12)


def test_run_batch_parity(spo):
    """UPDEEngine.run() must match Rust PyUPDEStepper.run() at arbitrary length."""
    n = 8
    rng = np.random.default_rng(11)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)
    knm = 0.3 * np.ones((n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    dt = 0.01
    n_steps = 100

    py_engine = UPDEEngine(n, dt=dt, method="rk4")
    py = py_engine.run(phases.copy(), omegas, knm, 0.0, 0.0, alpha, n_steps)

    rust = spo.PyUPDEStepper(n, dt=dt, method="rk4")
    ru = np.asarray(
        rust.run(
            phases.copy(),
            omegas,
            knm.ravel(),
            0.0,
            0.0,
            alpha.ravel(),
            n_steps,
        )
    )
    np.testing.assert_allclose(py, ru, atol=1e-7)


def test_determinism_same_seed(spo):
    """Two independent Rust stepper instances must produce byte-identical
    output for the same inputs — guards against hidden global state."""
    n = 8
    rng = np.random.default_rng(55)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)
    knm = 0.4 * np.ones((n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    dt = 0.01

    def run_once() -> np.ndarray:
        rust = spo.PyUPDEStepper(n, dt=dt, method="rk4")
        p = phases.copy()
        for _ in range(50):
            p = np.asarray(
                rust.step(p, omegas, knm.ravel(), 0.0, 0.0, alpha.ravel())
            )
        return p

    a = run_once()
    b = run_once()
    np.testing.assert_array_equal(a, b)


def test_order_parameter_edge_single_oscillator(spo):
    """N=1: order parameter is trivially 1.0 regardless of phase."""
    phases = np.array([1.23])
    r_py, _ = compute_order_parameter(phases)
    r_rust, _ = spo.order_parameter(phases)
    assert abs(r_py - 1.0) < 1e-12
    assert abs(r_rust - 1.0) < 1e-12


def test_order_parameter_edge_antiphase(spo):
    """Two anti-phase oscillators produce R = 0 on both paths."""
    phases = np.array([0.0, np.pi])
    r_py, _ = compute_order_parameter(phases)
    r_rust, _ = spo.order_parameter(phases)
    assert abs(r_py - r_rust) < 1e-12
    assert r_py < 1e-12


def test_order_parameter_edge_fully_synchronous(spo):
    """All-equal phases produce R = 1 on both paths."""
    phases = np.full(16, 0.42)
    r_py, _ = compute_order_parameter(phases)
    r_rust, _ = spo.order_parameter(phases)
    assert abs(r_py - 1.0) < 1e-12
    assert abs(r_rust - 1.0) < 1e-12


# Pipeline wiring: this suite is the gate that guarantees the Rust FFI
# path is substitutable for the Python reference. Every case above exercises
# a failure mode that would silently diverge the two backends: lag,
# drive, sign, asymmetry, scale, degeneracy, run-batch, determinism and
# extreme order-parameter edges. If this file turns green, the _HAS_RUST
# fast path is safe to ship.
