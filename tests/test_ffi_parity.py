# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

"""Cross-validate Python engine against Rust FFI when spo_kernel is available."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator._compat import HAS_RUST
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="spo_kernel not installed")


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
    rust_result = np.array(
        rust.step(phases.tolist(), omegas, knm.ravel(), 0.0, 0.0, alpha.ravel())
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
    rust_phases = phases.tolist()
    knm_flat = knm.ravel()
    alpha_flat = alpha.ravel()
    for _ in range(50):
        rust_phases = rust.step(rust_phases, omegas, knm_flat, 0.0, 0.0, alpha_flat)
    rust_phases = np.array(rust_phases)

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
    rust_phases = phases.tolist()
    knm_flat = knm.ravel()
    alpha_flat = alpha.ravel()
    for _ in range(50):
        rust_phases = rust.step(rust_phases, omegas, knm_flat, 0.0, 0.0, alpha_flat)
    rust_phases = np.array(rust_phases)

    np.testing.assert_allclose(py_phases, rust_phases, atol=1e-6)


def test_order_parameter_parity(spo):
    phases = [0.1, 0.2, 0.15, 0.12]
    r_py, _ = compute_order_parameter(np.array(phases))
    r_rust, _ = spo.order_parameter(np.array(phases))
    assert abs(r_py - r_rust) < 1e-10
