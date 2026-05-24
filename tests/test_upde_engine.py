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


def _py_engine(n: int, dt: float = 0.01, method: str = "euler", **kwargs):
    engine = UPDEEngine(n_oscillators=n, dt=dt, method=method, **kwargs)
    engine._use_rust = False
    return engine


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


def test_step_rejects_self_coupling_diagonal():
    engine = UPDEEngine(n_oscillators=3, dt=0.01)
    phases = np.array([0.0, 0.5, 1.0])
    omegas = np.ones(3)
    knm = np.zeros((3, 3))
    knm[1, 1] = 0.2
    alpha = np.zeros((3, 3))

    with pytest.raises(ValueError, match="self-coupling"):
        engine.step(phases, omegas, knm, 0.0, 0.0, alpha)


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


# Salvaged module-specific behavioural contracts from deleted broad tests.
class TestUPDEEngineInputValidation:
    """Verify that UPDEEngine rejects invalid inputs with clear error messages
    and that RK45 handles extreme conditions gracefully."""

    def test_alpha_nan_raises_with_message(self):
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        eng = UPDEEngine(2, dt=0.01)
        with pytest.raises(ValueError, match="alpha contains NaN"):
            eng.step(
                np.array([0.1, 0.2]),
                np.array([1.0, 1.0]),
                np.zeros((2, 2)),
                0.0,
                0.0,
                np.array([[0.0, float("nan")], [0.0, 0.0]]),
            )

    def test_alpha_inf_also_rejected(self):
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        eng = UPDEEngine(2, dt=0.01)
        with pytest.raises(ValueError, match="alpha contains NaN"):
            eng.step(
                np.array([0.1, 0.2]),
                np.array([1.0, 1.0]),
                np.zeros((2, 2)),
                0.0,
                0.0,
                np.array([[0.0, float("inf")], [0.0, 0.0]]),
            )

    def test_rk45_extreme_coupling_remains_finite(self):
        """RK45 with extremely tight tolerances and large coupling must
        exhaust retries gracefully (fallback to Euler) and return finite phases."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        eng = UPDEEngine(4, dt=1.0, method="rk45", atol=1e-15, rtol=1e-15)
        phases = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        omegas = np.array([100.0, 200.0, 300.0, 400.0])
        knm = np.full((4, 4), 1000.0)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((4, 4))

        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert result.shape == (4,)
        assert np.all(np.isfinite(result)), (
            f"RK45 fallback must return finite phases: {result}"
        )
        # Phases must have changed (omegas are non-zero)
        assert not np.allclose(result, phases), (
            "Extreme conditions should still advance phases"
        )


# Salvaged module-specific behavioural contracts from deleted mixed tests.


class TestUPDEEnginePythonPath:
    def test_euler_step(self):
        n = 4
        engine = _py_engine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert result.shape == (n,)
        assert np.all(result >= 0.0)
        assert np.all(result < TWO_PI)

    def test_euler_with_zeta(self):
        n = 4
        engine = _py_engine(n, dt=0.01)
        phases = np.zeros(n)
        omegas = np.zeros(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        result = engine.step(phases, omegas, knm, 1.0, np.pi / 2, alpha)
        assert np.all(result > 0.0)

    def test_rk4_step(self):
        n = 4
        engine = _py_engine(n, dt=0.01, method="rk4")
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(0.5, 2.0, n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert result.shape == (n,)
        assert np.all(np.isfinite(result))
        assert not np.allclose(result, phases), "RK4 must advance phases"

    def test_rk45_step(self):
        n = 4
        engine = _py_engine(n, dt=0.01, method="rk45")
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(0.5, 2.0, n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert result.shape == (n,)
        assert np.all(np.isfinite(result))
        assert engine.last_dt > 0.0
        assert not np.allclose(result, phases), "RK45 must advance phases"

    def test_rk45_with_zeta(self):
        n = 4
        engine = _py_engine(n, dt=0.01, method="rk45")
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(0.5, 2.0, n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = engine.step(phases, omegas, knm, 0.5, 1.0, alpha)
        assert np.all(np.isfinite(result))

    def test_run_euler(self):
        n = 4
        engine = _py_engine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = engine.run(phases, omegas, knm, 0.0, 0.0, alpha, 10)
        assert result.shape == (n,)
        assert np.all(result >= 0.0)
        assert np.all(result < TWO_PI)

    def test_run_rk4(self):
        n = 4
        engine = _py_engine(n, dt=0.01, method="rk4")
        phases = np.zeros(n)
        omegas = np.ones(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        result = engine.run(phases, omegas, knm, 0.0, 0.0, alpha, 5)
        assert result.shape == (n,)
        # With ω=1 and no coupling, phases should advance by ~5*dt*ω = 0.05
        expected = (phases + 5 * 0.01 * omegas) % TWO_PI
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_run_rk45(self):
        """RK45 adaptive: phases must advance and stay finite."""
        n = 4
        engine = _py_engine(n, dt=0.01, method="rk45")
        phases = np.zeros(n)
        omegas = np.ones(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        result = engine.run(phases, omegas, knm, 0.0, 0.0, alpha, 5)
        assert result.shape == (n,)
        assert np.all(np.isfinite(result))
        assert not np.allclose(result, phases), "RK45 run must advance phases"

    def test_python_euler_matches_analytical_free_rotation(self):
        """Without coupling, Euler: θ(t) = θ₀ + ω·dt·n_steps (analytical)."""
        n = 4
        engine = _py_engine(n, dt=0.01)
        phases = np.array([0.0, 1.0, 2.0, 3.0])
        omegas = np.array([1.0, 2.0, 3.0, 4.0])
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        result = engine.run(phases, omegas, knm, 0.0, 0.0, alpha, 100)
        expected = (phases + 100 * 0.01 * omegas) % TWO_PI
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_python_rk4_euler_agree_for_zero_coupling(self):
        """RK4 and Euler must agree for free rotation (no coupling)."""
        n = 4
        phases = np.array([0.5, 1.5, 2.5, 3.5])
        omegas = np.array([1.0, 1.5, 2.0, 0.5])
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        r_euler = _py_engine(n, dt=0.01).run(phases, omegas, knm, 0.0, 0.0, alpha, 50)
        r_rk4 = _py_engine(n, dt=0.01, method="rk4").run(
            phases, omegas, knm, 0.0, 0.0, alpha, 50
        )
        np.testing.assert_allclose(r_euler, r_rk4, atol=1e-6)


class TestUPDEEngineValidation:
    def test_nan_zeta_raises(self):
        engine = _py_engine(4, dt=0.01)
        with pytest.raises(ValueError, match="finite"):
            engine.step(
                np.zeros(4),
                np.ones(4),
                np.zeros((4, 4)),
                float("nan"),
                0.0,
                np.zeros((4, 4)),
            )

    def test_inf_psi_raises(self):
        engine = _py_engine(4, dt=0.01)
        with pytest.raises(ValueError, match="finite"):
            engine.step(
                np.zeros(4),
                np.ones(4),
                np.zeros((4, 4)),
                0.0,
                float("inf"),
                np.zeros((4, 4)),
            )


# ──────────────────────────────────────────────────────────────────────
# physical.py: force Python fallback path for extract()
# ──────────────────────────────────────────────────────────────────────
