# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — JAX engine tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.jax_engine import HAS_JAX

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")

TWO_PI = 2.0 * np.pi


@pytest.fixture()
def jax_upde():
    from scpn_phase_orchestrator.upde.jax_engine import JaxUPDEEngine

    return JaxUPDEEngine(8, dt=0.01, method="rk4")


@pytest.fixture()
def jax_sl():
    from scpn_phase_orchestrator.upde.jax_engine import JaxStuartLandauEngine

    return JaxStuartLandauEngine(4, dt=0.01)


def test_jax_upde_phases_bounded(jax_upde):
    rng = np.random.default_rng(0)
    phases = rng.uniform(0, TWO_PI, 8)
    omegas = np.ones(8)
    knm = np.zeros((8, 8))
    alpha = np.zeros((8, 8))
    result = jax_upde.step(phases, omegas, knm, 0.0, 0.0, alpha)
    assert result.shape == (8,)
    assert np.all(result >= 0.0)
    assert np.all(result < TWO_PI + 1e-6)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n": 0}, "n"),
        ({"n": True}, "n"),
        ({"n": 8, "dt": "0.01"}, "dt"),
        ({"n": 8, "dt": np.inf}, "dt"),
        ({"n": 8, "method": "verlet"}, "unsupported method"),
    ],
)
def test_jax_upde_rejects_invalid_constructor_inputs(kwargs, match):
    from scpn_phase_orchestrator.upde.jax_engine import JaxUPDEEngine

    with pytest.raises(ValueError, match=match):
        JaxUPDEEngine(**kwargs)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("phases", np.zeros(7), "phases"),
        ("omegas", np.zeros(7), "omegas"),
        ("knm", np.zeros((8, 7)), "knm"),
        ("alpha", np.zeros((7, 8)), "alpha"),
        ("phases", np.full(8, np.nan), "phases"),
        ("omegas", np.full(8, np.inf), "omegas"),
        ("knm", np.full((8, 8), np.nan), "knm"),
        ("alpha", np.full((8, 8), np.inf), "alpha"),
        ("alpha", object(), "alpha"),
        ("zeta", np.nan, "zeta"),
        ("psi", "0.0", "psi"),
    ],
)
def test_jax_upde_rejects_invalid_step_inputs(jax_upde, field, value, match):
    values = {
        "phases": np.zeros(8),
        "omegas": np.ones(8),
        "knm": np.zeros((8, 8)),
        "zeta": 0.0,
        "psi": 0.0,
        "alpha": np.zeros((8, 8)),
    }
    values[field] = value
    with pytest.raises(ValueError, match=match):
        jax_upde.step(
            values["phases"],
            values["omegas"],
            values["knm"],
            values["zeta"],
            values["psi"],
            values["alpha"],
        )


def test_jax_upde_parity_with_numpy():
    from scpn_phase_orchestrator.upde.engine import UPDEEngine
    from scpn_phase_orchestrator.upde.jax_engine import JaxUPDEEngine

    n = 8
    rng = np.random.default_rng(1)
    phases = rng.uniform(0, TWO_PI, n)
    omegas = np.ones(n) + rng.normal(0, 0.1, n)
    knm = np.full((n, n), 0.3)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    py = UPDEEngine(n, dt=0.01, method="rk4")
    jx = JaxUPDEEngine(n, dt=0.01, method="rk4")

    py_result = py.step(phases.copy(), omegas, knm, 0.0, 0.0, alpha)
    jx_result = jx.step(phases.copy(), omegas, knm, 0.0, 0.0, alpha)

    np.testing.assert_allclose(py_result, jx_result, atol=1e-10)


def test_jax_upde_sync():
    from scpn_phase_orchestrator.upde.jax_engine import JaxUPDEEngine
    from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

    n = 16
    rng = np.random.default_rng(2)
    phases = rng.uniform(0, TWO_PI, n)
    omegas = np.ones(n)
    knm = np.full((n, n), 0.5)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    eng = JaxUPDEEngine(n, dt=0.01, method="rk4")
    for _ in range(500):
        phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)

    r, _ = compute_order_parameter(np.array(phases, dtype=np.float64))
    assert r > 0.9


def test_jax_sl_phases_and_amplitudes(jax_sl):
    n = 4
    rng = np.random.default_rng(0)
    state = np.concatenate([rng.uniform(0, TWO_PI, n), np.ones(n)])
    omegas = np.ones(n)
    mu = np.ones(n)
    knm = np.zeros((n, n))
    knm_r = np.zeros((n, n))
    alpha = np.zeros((n, n))

    result = jax_sl.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha, 0.0)
    assert result.shape == (2 * n,)
    assert np.all(result[:n] >= 0.0)
    assert np.all(result[:n] < TWO_PI + 1e-6)
    assert np.all(result[n:] >= 0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n": 0}, "n"),
        ({"n": False}, "n"),
        ({"n": 4, "dt": 0.0}, "dt"),
        ({"n": 4, "dt": np.nan}, "dt"),
    ],
)
def test_jax_stuart_landau_rejects_invalid_constructor_inputs(kwargs, match):
    from scpn_phase_orchestrator.upde.jax_engine import JaxStuartLandauEngine

    with pytest.raises(ValueError, match=match):
        JaxStuartLandauEngine(**kwargs)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("state", np.zeros(7), "state"),
        ("omegas", np.zeros(3), "omegas"),
        ("mu", np.zeros(3), "mu"),
        ("knm", np.zeros((4, 3)), "knm"),
        ("knm_r", np.zeros((3, 4)), "knm_r"),
        ("alpha", np.zeros((3, 4)), "alpha"),
        ("state", np.full(8, np.nan), "state"),
        ("mu", np.full(4, np.inf), "mu"),
        ("zeta", True, "zeta"),
        ("psi", np.inf, "psi"),
        ("epsilon", "1.0", "epsilon"),
    ],
)
def test_jax_stuart_landau_rejects_invalid_step_inputs(jax_sl, field, value, match):
    values = {
        "state": np.concatenate([np.zeros(4), np.ones(4)]),
        "omegas": np.ones(4),
        "mu": np.ones(4),
        "knm": np.zeros((4, 4)),
        "knm_r": np.zeros((4, 4)),
        "zeta": 0.0,
        "psi": 0.0,
        "alpha": np.zeros((4, 4)),
        "epsilon": 1.0,
    }
    values[field] = value
    with pytest.raises(ValueError, match=match):
        jax_sl.step(
            values["state"],
            values["omegas"],
            values["mu"],
            values["knm"],
            values["knm_r"],
            values["zeta"],
            values["psi"],
            values["alpha"],
            values["epsilon"],
        )


def test_jax_sl_parity():
    from scpn_phase_orchestrator.upde.jax_engine import JaxStuartLandauEngine
    from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

    n = 4
    rng = np.random.default_rng(10)
    state = np.concatenate([rng.uniform(0, TWO_PI, n), rng.uniform(0.5, 1.5, n)])
    omegas = np.ones(n)
    mu = np.ones(n)
    knm = np.full((n, n), 0.3)
    np.fill_diagonal(knm, 0.0)
    knm_r = np.full((n, n), 0.2)
    np.fill_diagonal(knm_r, 0.0)
    alpha = np.zeros((n, n))

    py = StuartLandauEngine(n, dt=0.01, method="rk4")
    py._use_rust = False
    jx = JaxStuartLandauEngine(n, dt=0.01)

    py_result = py.step(
        state.copy(),
        omegas,
        mu,
        knm,
        knm_r,
        0.0,
        0.0,
        alpha,
        epsilon=0.5,
    )
    jx_result = jx.step(
        state.copy(),
        omegas,
        mu,
        knm,
        knm_r,
        0.0,
        0.0,
        alpha,
        epsilon=0.5,
    )

    np.testing.assert_allclose(py_result, jx_result, atol=1e-8)
