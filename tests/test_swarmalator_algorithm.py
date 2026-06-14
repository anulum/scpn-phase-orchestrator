# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for swarmalator stepper

"""Algorithmic properties of :class:`SwarmalatorEngine`.

Covered: constructor validation; step output shape; phase wrap
within ``[0, 2π)``; zero-coupling limit reduces to pure-ω rotation;
empty-/single-agent edge cases; run trajectory shapes;
order-parameter helper; Hypothesis invariants.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import swarmalator as sw_mod
from scpn_phase_orchestrator.upde.swarmalator import SwarmalatorEngine

TWO_PI = 2.0 * math.pi


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = sw_mod.ACTIVE_BACKEND
        sw_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            sw_mod.ACTIVE_BACKEND = prev

    return wrapper


def _problem(seed: int, n: int = 16, dim: int = 2):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, (n, dim))
    phases = rng.uniform(0, TWO_PI, n)
    omegas = rng.normal(0.5, 0.2, n)
    return pos, phases, omegas


class TestConstructor:
    def test_rejects_zero_agents(self):
        with pytest.raises(ValueError, match="n_agents"):
            SwarmalatorEngine(n_agents=0, dim=2, dt=0.01)

    def test_rejects_zero_dim(self):
        with pytest.raises(ValueError, match="dim"):
            SwarmalatorEngine(n_agents=4, dim=0, dt=0.01)

    def test_rejects_non_positive_dt(self):
        with pytest.raises(ValueError, match="dt"):
            SwarmalatorEngine(n_agents=4, dim=2, dt=0.0)

    def test_rejects_non_real_dt(self):
        with pytest.raises(ValueError, match="dt"):
            SwarmalatorEngine(n_agents=4, dim=2, dt=object())


class TestStep:
    @_python
    def test_invalid_state_shapes_are_rejected(self):
        pos, phases, omegas = _problem(4)
        eng = SwarmalatorEngine(16, 2, 0.01)
        with pytest.raises(ValueError, match="shape"):
            eng.step(pos[:15], phases, omegas, 1.0, 1.0, 0.8, 1.0)
        with pytest.raises(ValueError, match="shape"):
            eng.step(pos, phases[:15], omegas, 1.0, 1.0, 0.8, 1.0)
        with pytest.raises(ValueError, match="shape"):
            eng.step(pos, phases, omegas[:15], 1.0, 1.0, 0.8, 1.0)

        bad_pos = pos.copy()
        with pytest.raises(ValueError, match="shape"):
            eng.step(bad_pos[:, :1], phases, omegas, 1.0, 1.0, 0.8, 1.0)

    @_python
    def test_nonfinite_step_inputs_are_rejected(self):
        pos, phases, omegas = _problem(5)
        eng = SwarmalatorEngine(16, 2, 0.01)

        bad_pos = pos.copy()
        bad_pos[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            eng.step(bad_pos, phases, omegas, 1.0, 1.0, 0.8, 1.0)

        bad_phases = phases.copy()
        bad_phases[0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            eng.step(pos, bad_phases, omegas, 1.0, 1.0, 0.8, 1.0)

        bad_omegas = omegas.copy()
        bad_omegas[0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            eng.step(pos, phases, bad_omegas, 1.0, 1.0, 0.8, 1.0)

    @_python
    @pytest.mark.parametrize(
        ("pos", "phases", "omegas", "match"),
        [
            (
                np.zeros((2, 2), dtype=bool),
                np.zeros(2, dtype=np.float64),
                np.zeros(2, dtype=np.float64),
                "pos",
            ),
            (
                np.zeros((2, 2), dtype=np.float64),
                np.zeros(2, dtype=bool),
                np.zeros(2, dtype=np.float64),
                "phases",
            ),
            (
                np.zeros((2, 2), dtype=np.float64),
                np.zeros(2, dtype=np.float64),
                np.zeros(2, dtype=bool),
                "omegas",
            ),
        ],
    )
    def test_boolean_state_aliases_are_rejected(self, pos, phases, omegas, match):
        eng = SwarmalatorEngine(2, 2, 0.01)

        with pytest.raises(ValueError, match=match):
            eng.step(pos, phases, omegas)

    @_python
    def test_single_agent_kinematic_boundary(self):
        eng = SwarmalatorEngine(1, dim=2, dt=0.01)
        pos = np.array([[0.2, -0.1]], dtype=float)
        phases = np.array([0.8], dtype=float)
        omegas = np.array([1.2], dtype=float)
        new_pos, new_phases = eng.step(
            pos,
            phases,
            omegas,
            a=1.0,
            b=1.0,
            j=0.3,
            k=0.7,
        )
        np.testing.assert_allclose(new_pos, pos)
        np.testing.assert_allclose(new_phases, (phases + 0.01 * omegas) % (2.0 * np.pi))

    @_python
    def test_output_shape(self):
        pos, phases, omegas = _problem(0)
        eng = SwarmalatorEngine(16, 2, 0.01)
        p, ph = eng.step(pos, phases, omegas, 1.0, 1.0, 0.8, 1.2)
        assert p.shape == (16, 2)
        assert ph.shape == (16,)

    @_python
    def test_phases_wrap_in_two_pi(self):
        pos, phases, omegas = _problem(1)
        eng = SwarmalatorEngine(16, 2, 0.1)
        _, ph = eng.step(pos, phases, omegas, 1.0, 1.0, 1.0, 1.0)
        assert np.all(ph >= 0.0)
        assert np.all(ph < TWO_PI + 1e-12)

    @_python
    def test_zero_couplings_pure_rotation(self):
        """``a = b = j = k = 0`` → position frozen, phases evolve
        purely by ω."""
        pos, phases, omegas = _problem(2)
        eng = SwarmalatorEngine(16, 2, 0.01)
        new_pos, new_ph = eng.step(pos, phases, omegas, 0.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(new_pos, pos, atol=1e-12)
        expected = (phases + 0.01 * omegas) % TWO_PI
        np.testing.assert_allclose(new_ph, expected, atol=1e-12)

    @_python
    def test_two_body_sync_equilibrium_matches_ohs(self):
        """OHS canonical inverse-distance repulsion: two phase-synced agents
        balance at separation ``r = b / (A + J)`` (O'Keeffe-Hong-Strogatz,
        Nat. Commun. 2017). With ``A = b = J = 1`` that is ``r = 0.5``; the
        spurious ``|dx|**3`` core would instead balance at
        ``sqrt(0.5) ~ 0.707``, so this pins the repulsion exponent.
        """
        eng = SwarmalatorEngine(2, dim=2, dt=0.01)
        phases = np.array([0.3, 0.3], dtype=np.float64)  # synced -> cos=1
        omegas = np.zeros(2, dtype=np.float64)

        # At r = b / (A + J) = 0.5 the net radial velocity vanishes.
        pos_eq = np.array([[0.0, 0.0], [0.5, 0.0]], dtype=np.float64)
        new_eq, _ = eng.step(pos_eq, phases, omegas, a=1.0, b=1.0, j=1.0, k=0.0)
        np.testing.assert_allclose(new_eq, pos_eq, atol=1e-4)

        # At the old |dx|**3 equilibrium sqrt(0.5) attraction must now win,
        # so the pair closes in. This assertion fails for the |dx|**3 core.
        r0 = math.sqrt(0.5)
        pos_far = np.array([[0.0, 0.0], [r0, 0.0]], dtype=np.float64)
        new_far, _ = eng.step(pos_far, phases, omegas, a=1.0, b=1.0, j=1.0, k=0.0)
        sep_after = float(np.linalg.norm(new_far[1] - new_far[0]))
        assert sep_after < r0

    def test_selected_backend_output_must_preserve_finite_torus_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        pos = np.array([[0.0, 0.5], [1.0, -0.25]], dtype=np.float64)
        phases = np.array([0.2, 0.4], dtype=np.float64)
        omegas = np.array([0.1, -0.2], dtype=np.float64)

        def invalid_backend(*_args, **_kwargs):
            return pos.copy(), np.array([0.1, TWO_PI], dtype=np.float64)

        monkeypatch.setattr(sw_mod, "ACTIVE_BACKEND", "go")
        monkeypatch.setattr(sw_mod, "_load_backend", lambda _name: invalid_backend)
        eng = SwarmalatorEngine(2, 2, 0.01)

        with pytest.raises(ValueError, match="backend output phases"):
            eng.step(pos, phases, omegas)


class TestRun:
    @_python
    def test_trajectory_shapes(self):
        pos, phases, omegas = _problem(3)
        eng = SwarmalatorEngine(16, 2, 0.01)
        final_pos, final_ph, pos_traj, phase_traj = eng.run(
            pos,
            phases,
            omegas,
            n_steps=5,
        )
        assert final_pos.shape == (16, 2)
        assert final_ph.shape == (16,)
        assert pos_traj.shape == (5, 16, 2)
        assert phase_traj.shape == (5, 16)

    @_python
    def test_zero_steps_rejected(self):
        pos, phases, omegas = _problem(4)
        eng = SwarmalatorEngine(16, 2, 0.01)
        with pytest.raises(ValueError, match="n_steps"):
            eng.run(
                pos,
                phases,
                omegas,
                n_steps=0,
            )

    @_python
    def test_one_step_run_matches_single_step(self):
        pos, phases, omegas = _problem(5)
        eng = SwarmalatorEngine(16, 2, 0.01)
        step_pos, step_ph = eng.step(pos, phases, omegas)
        final_pos, final_ph, pos_traj, phase_traj = eng.run(
            pos,
            phases,
            omegas,
            n_steps=1,
        )
        np.testing.assert_allclose(final_pos, step_pos)
        np.testing.assert_allclose(final_ph, step_ph)
        assert pos_traj.shape == (1, 16, 2)
        assert phase_traj.shape == (1, 16)
        np.testing.assert_allclose(pos_traj[0], step_pos)
        np.testing.assert_allclose(phase_traj[0], step_ph)

    def test_negative_step_count_rejected(self):
        pos, phases, omegas = _problem(6)
        eng = SwarmalatorEngine(16, 2, 0.01)
        with pytest.raises(ValueError):
            eng.run(pos, phases, omegas, n_steps=-1)


class TestOrderParameter:
    @_python
    def test_perfectly_locked(self):
        phases = np.full(10, 0.5)
        eng = SwarmalatorEngine(10, 2, 0.01)
        assert eng.order_parameter(phases) == pytest.approx(1.0, abs=1e-12)

    @_python
    def test_uniform_approaches_zero(self):
        phases = np.linspace(0, TWO_PI, 1000, endpoint=False)
        eng = SwarmalatorEngine(1000, 2, 0.01)
        r = eng.order_parameter(phases)
        assert r < 1e-10


class TestHypothesis:
    @_python
    @given(
        n=st.integers(min_value=2, max_value=16),
        dim=st.integers(min_value=1, max_value=3),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_finite_output(self, n: int, dim: int, seed: int):
        pos, phases, omegas = _problem(seed, n, dim)
        eng = SwarmalatorEngine(n, dim, 0.01)
        p, ph = eng.step(pos, phases, omegas)
        assert np.all(np.isfinite(p))
        assert np.all(np.isfinite(ph))


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert sw_mod.AVAILABLE_BACKENDS
        assert "python" in sw_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert sw_mod.AVAILABLE_BACKENDS[0] == sw_mod.ACTIVE_BACKEND
