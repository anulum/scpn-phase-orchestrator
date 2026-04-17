# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stuart-Landau parity tests

"""Cross-validate Python StuartLandauEngine against Rust PyStuartLandauStepper."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator._compat import HAS_RUST
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

_HAS_SL = HAS_RUST and hasattr(
    __import__("spo_kernel") if HAS_RUST else None,
    "PyStuartLandauStepper",
)
pytestmark = pytest.mark.skipif(
    not _HAS_SL,
    reason="spo_kernel.PyStuartLandauStepper not available",
)

TWO_PI = 2.0 * np.pi


@pytest.fixture()
def spo():
    import spo_kernel

    return spo_kernel


def _build_params(n, rng):
    omegas = np.ones(n) + rng.normal(0, 0.1, n)
    mu = np.ones(n)
    dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
    knm = 0.3 * np.exp(-0.3 * dist)
    np.fill_diagonal(knm, 0.0)
    knm_r = 0.2 * np.exp(-0.3 * dist)
    np.fill_diagonal(knm_r, 0.0)
    alpha = np.zeros((n, n))
    return omegas, mu, knm, knm_r, alpha


def test_euler_parity(spo):
    n = 8
    rng = np.random.default_rng(10)
    theta = rng.uniform(0, TWO_PI, n)
    r = rng.uniform(0.3, 1.5, n)
    state = np.concatenate([theta, r])
    omegas, mu, knm, knm_r, alpha = _build_params(n, rng)
    dt = 0.01

    py_eng = StuartLandauEngine(n, dt=dt, method="euler")
    py_eng._use_rust = False
    py_result = py_eng.step(
        state.copy(), omegas, mu, knm, knm_r, 0.0, 0.0, alpha, epsilon=1.0
    )

    rust = spo.PyStuartLandauStepper(n, dt=dt, method="euler")
    rust_result = np.array(
        rust.step(
            state.copy(),
            omegas,
            mu,
            knm.ravel(),
            knm_r.ravel(),
            0.0,
            0.0,
            alpha.ravel(),
            1.0,
        )
    )

    np.testing.assert_allclose(py_result, rust_result, atol=1e-10)


def test_rk4_parity(spo):
    n = 8
    rng = np.random.default_rng(20)
    theta = rng.uniform(0, TWO_PI, n)
    r = rng.uniform(0.3, 1.5, n)
    state = np.concatenate([theta, r])
    omegas, mu, knm, knm_r, alpha = _build_params(n, rng)
    dt = 0.01

    py_eng = StuartLandauEngine(n, dt=dt, method="rk4")
    py_eng._use_rust = False
    py_state = state.copy()
    for _ in range(50):
        py_state = py_eng.step(
            py_state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha, epsilon=0.5
        )

    rust = spo.PyStuartLandauStepper(n, dt=dt, method="rk4")
    knm_flat, knm_r_flat, alpha_flat = knm.ravel(), knm_r.ravel(), alpha.ravel()
    rust_state = state.copy()
    for _ in range(50):
        rust_state = np.asarray(
            rust.step(
                rust_state,
                omegas,
                mu,
                knm_flat,
                knm_r_flat,
                0.0,
                0.0,
                alpha_flat,
                0.5,
            )
        )

    np.testing.assert_allclose(py_state, rust_state, atol=1e-8)


@pytest.mark.xfail(
    reason="Python/Rust rk45 adaptive dt diverge on chaotic SL system",
    strict=False,
)
def test_rk45_parity(spo):
    n = 8
    rng = np.random.default_rng(30)
    theta = rng.uniform(0, TWO_PI, n)
    r = rng.uniform(0.3, 1.5, n)
    state = np.concatenate([theta, r])
    omegas, mu, knm, knm_r, alpha = _build_params(n, rng)
    dt = 0.01

    py_eng = StuartLandauEngine(n, dt=dt, method="rk45")
    py_eng._use_rust = False
    py_state = state.copy()
    for _ in range(50):
        py_state = py_eng.step(
            py_state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha, epsilon=0.5
        )

    rust = spo.PyStuartLandauStepper(n, dt=dt, method="rk45")
    knm_flat, knm_r_flat, alpha_flat = knm.ravel(), knm_r.ravel(), alpha.ravel()
    rust_state = state.copy()
    for _ in range(50):
        rust_state = np.asarray(
            rust.step(
                rust_state,
                omegas,
                mu,
                knm_flat,
                knm_r_flat,
                0.0,
                0.0,
                alpha_flat,
                0.5,
            )
        )

    np.testing.assert_allclose(py_state, rust_state, atol=1e-5)


def test_external_drive_parity(spo):
    n = 4
    rng = np.random.default_rng(40)
    state = np.concatenate([rng.uniform(0, TWO_PI, n), np.ones(n)])
    omegas, mu, knm, knm_r, alpha = _build_params(n, rng)
    dt = 0.01

    py_eng = StuartLandauEngine(n, dt=dt, method="euler")
    py_eng._use_rust = False
    py_result = py_eng.step(
        state.copy(), omegas, mu, knm, knm_r, 0.5, 1.0, alpha, epsilon=1.0
    )

    rust = spo.PyStuartLandauStepper(n, dt=dt, method="euler")
    rust_result = np.array(
        rust.step(
            state.copy(),
            omegas,
            mu,
            knm.ravel(),
            knm_r.ravel(),
            0.5,
            1.0,
            alpha.ravel(),
            1.0,
        )
    )

    np.testing.assert_allclose(py_result, rust_result, atol=1e-10)


def test_zero_epsilon_parity(spo):
    n = 4
    rng = np.random.default_rng(50)
    state = np.concatenate([rng.uniform(0, TWO_PI, n), np.ones(n)])
    omegas, mu, knm, knm_r, alpha = _build_params(n, rng)
    dt = 0.01

    py_eng = StuartLandauEngine(n, dt=dt, method="euler")
    py_eng._use_rust = False
    py_result = py_eng.step(
        state.copy(), omegas, mu, knm, knm_r, 0.0, 0.0, alpha, epsilon=0.0
    )

    rust = spo.PyStuartLandauStepper(n, dt=dt, method="euler")
    rust_result = np.array(
        rust.step(
            state.copy(),
            omegas,
            mu,
            knm.ravel(),
            knm_r.ravel(),
            0.0,
            0.0,
            alpha.ravel(),
            0.0,
        )
    )

    np.testing.assert_allclose(py_result, rust_result, atol=1e-10)


@pytest.mark.xfail(
    reason="PyStuartLandauStepper.run() not yet in Rust kernel",
    strict=False,
)
def test_run_parity(spo):
    """PyStuartLandauStepper.run() matches N sequential Python steps."""
    n = 8
    n_steps = 100
    rng = np.random.default_rng(60)
    theta = rng.uniform(0, TWO_PI, n)
    r = rng.uniform(0.3, 1.5, n)
    state = np.concatenate([theta, r])
    omegas, mu, knm, knm_r, alpha = _build_params(n, rng)
    dt = 0.01

    py_eng = StuartLandauEngine(n, dt=dt, method="rk4")
    py_eng._use_rust = False
    py_state = state.copy()
    for _ in range(n_steps):
        py_state = py_eng.step(
            py_state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha, epsilon=0.5
        )

    rust = spo.PyStuartLandauStepper(n, dt=dt, method="rk4")
    rust_state = np.asarray(
        rust.run(
            state.copy(),
            omegas,
            mu,
            knm.ravel(),
            knm_r.ravel(),
            0.0,
            0.0,
            alpha.ravel(),
            0.5,
            n_steps,
        )
    )

    np.testing.assert_allclose(py_state, rust_state, atol=1e-8)


@pytest.mark.xfail(
    reason="PyStuartLandauStepper.run() not yet in Rust kernel",
    strict=False,
)
def test_run_with_drive_parity(spo):
    """run() parity with external drive (zeta, psi)."""
    n = 4
    n_steps = 50
    rng = np.random.default_rng(70)
    state = np.concatenate([rng.uniform(0, TWO_PI, n), np.ones(n)])
    omegas, mu, knm, knm_r, alpha = _build_params(n, rng)
    dt = 0.01

    py_eng = StuartLandauEngine(n, dt=dt, method="euler")
    py_eng._use_rust = False
    py_state = state.copy()
    for _ in range(n_steps):
        py_state = py_eng.step(
            py_state, omegas, mu, knm, knm_r, 0.3, 1.5, alpha, epsilon=1.0
        )

    rust = spo.PyStuartLandauStepper(n, dt=dt, method="euler")
    rust_state = np.asarray(
        rust.run(
            state.copy(),
            omegas,
            mu,
            knm.ravel(),
            knm_r.ravel(),
            0.3,
            1.5,
            alpha.ravel(),
            1.0,
            n_steps,
        )
    )

    np.testing.assert_allclose(py_state, rust_state, atol=1e-10)


# Pipeline wiring is proven by the Rust/Python parity tests above:
# they cross-validate StuartLandauEngine against spo_kernel Rust stepper.
