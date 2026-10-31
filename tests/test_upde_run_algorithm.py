# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for upde_run

"""Algorithmic properties of ``upde/engine.upde_run``.

Covers: output shape, phase wrapping, method dispatch, n_substeps
monotonicity, adaptive-dt bounds, synchronisation attractor, NaN/Inf
propagation rules, and behaviour under degenerate inputs.

All tests force the Python reference backend so they stay
deterministic regardless of which toolchains are installed.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde import _ref_kernel as ref_kernel_mod
from scpn_phase_orchestrator.upde import engine as eng_mod
from scpn_phase_orchestrator.upde.engine import upde_run

TWO_PI = 2.0 * math.pi
FloatArray = NDArray[np.float64]


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _python(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Run ``func`` with the pure-Python backend forced."""

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        prev = eng_mod.ACTIVE_BACKEND
        eng_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            eng_mod.ACTIVE_BACKEND = prev

    return wrapper


def _problem(
    rng: np.random.Generator, n: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    phases = rng.uniform(0.0, TWO_PI, size=n)
    omegas = rng.normal(0.0, 0.3, size=n)
    knm = rng.uniform(0.3, 1.0, size=(n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = rng.uniform(-0.2, 0.2, size=(n, n))
    np.fill_diagonal(alpha, 0.0)
    return phases, omegas, knm, alpha


class TestShape:
    @_python
    def test_output_shape(self) -> None:
        phases, omegas, knm, alpha = _problem(np.random.default_rng(0), n=5)
        out = upde_run(
            phases,
            omegas,
            knm,
            alpha,
            zeta=0.1,
            psi=0.2,
            dt=0.01,
            n_steps=30,
        )
        assert out.shape == (5,)

    @_python
    def test_wrapped_in_two_pi(self) -> None:
        phases, omegas, knm, alpha = _problem(np.random.default_rng(1), n=4)
        out = upde_run(
            phases,
            omegas,
            knm,
            alpha,
            zeta=0.0,
            psi=0.0,
            dt=0.02,
            n_steps=100,
        )
        assert np.all(out >= 0.0)
        assert np.all(out < TWO_PI + 1e-12)


class TestMethodDispatch:
    @_python
    def test_unknown_method_rejected(self) -> None:
        phases, omegas, knm, alpha = _problem(np.random.default_rng(0), n=3)
        with pytest.raises(ValueError, match="unknown method"):
            upde_run(
                phases,
                omegas,
                knm,
                alpha,
                zeta=0.0,
                psi=0.0,
                dt=0.01,
                n_steps=10,
                method="heun",
            )

    @_python
    def test_n_substeps_floor(self) -> None:
        phases, omegas, knm, alpha = _problem(np.random.default_rng(0), n=3)
        with pytest.raises(ValueError, match="n_substeps"):
            upde_run(
                phases,
                omegas,
                knm,
                alpha,
                zeta=0.0,
                psi=0.0,
                dt=0.01,
                n_steps=5,
                method="euler",
                n_substeps=0,
            )

    @pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
    @_python
    def test_three_methods_all_finite(self, method: str) -> None:
        phases, omegas, knm, alpha = _problem(np.random.default_rng(7), n=4)
        out = upde_run(
            phases,
            omegas,
            knm,
            alpha,
            zeta=0.2,
            psi=0.1,
            dt=0.01,
            n_steps=50,
            method=method,
        )
        assert np.all(np.isfinite(out))


def test_rk45_step_falls_back_after_failed_adaptation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bad_dp_stages(*args: object) -> tuple[FloatArray, FloatArray]:
        phases = np.asarray(args[0], dtype=np.float64)
        y5 = phases + 0.5
        y4 = phases
        return y5, y4

    monkeypatch.setattr(ref_kernel_mod, "_dp_stages", bad_dp_stages)
    phases = np.array([0.1, 0.2], dtype=np.float64)
    omegas = np.array([1.0, 1.2], dtype=np.float64)
    knm = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
    alpha = np.zeros((2, 2), dtype=np.float64)
    out, last_dt = ref_kernel_mod._rk45_step(
        phases,
        omegas,
        knm,
        alpha,
        zeta=0.0,
        psi=0.0,
        atol=1e-6,
        rtol=1e-3,
        dt_config=0.1,
        last_dt=0.1,
    )

    assert np.allclose(out, phases + 0.5)
    assert np.isclose(last_dt, 0.00016, rtol=1e-12, atol=0.0)


class TestSubstepping:
    @_python
    def test_euler_more_substeps_closer_to_rk4(self) -> None:
        """More Euler substeps approach the RK4 solution (higher order)."""
        phases, omegas, knm, alpha = _problem(np.random.default_rng(11), n=4)
        ref = upde_run(
            phases,
            omegas,
            knm,
            alpha,
            zeta=0.0,
            psi=0.0,
            dt=0.01,
            n_steps=50,
            method="rk4",
        )
        coarse = upde_run(
            phases,
            omegas,
            knm,
            alpha,
            zeta=0.0,
            psi=0.0,
            dt=0.01,
            n_steps=50,
            method="euler",
            n_substeps=1,
        )
        fine = upde_run(
            phases,
            omegas,
            knm,
            alpha,
            zeta=0.0,
            psi=0.0,
            dt=0.01,
            n_steps=50,
            method="euler",
            n_substeps=8,
        )
        d_coarse = float(np.max(np.abs(coarse - ref)))
        d_fine = float(np.max(np.abs(fine - ref)))
        assert d_fine <= d_coarse


class TestAttractor:
    @_python
    def test_strong_coupling_synchronises(self) -> None:
        """Strong all-to-all coupling with zero ω pulls phases close
        together: final phase spread < initial spread."""
        rng = np.random.default_rng(21)
        n = 6
        phases = rng.uniform(0.0, TWO_PI, size=n)
        omegas = np.zeros(n)
        knm = np.full((n, n), 5.0)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        # Account for wrap: compare complex-mean magnitudes.
        init_R = float(np.abs(np.mean(np.exp(1j * phases))))
        out = upde_run(
            phases,
            omegas,
            knm,
            alpha,
            zeta=0.0,
            psi=0.0,
            dt=0.005,
            n_steps=800,
            method="rk4",
        )
        final_R = float(np.abs(np.mean(np.exp(1j * out))))
        assert final_R > init_R


class TestDriverPulling:
    @_python
    def test_driver_pulls_phases_to_psi(self) -> None:
        """A strong driver at ψ pulls phases towards ψ."""
        rng = np.random.default_rng(31)
        n = 4
        phases = rng.uniform(0.0, TWO_PI, size=n)
        omegas = np.zeros(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        psi = 1.0
        out = upde_run(
            phases,
            omegas,
            knm,
            alpha,
            zeta=3.0,
            psi=psi,
            dt=0.005,
            n_steps=500,
            method="rk4",
        )
        # Distance to ψ in the circular metric.
        diff = np.abs(((out - psi + math.pi) % TWO_PI) - math.pi)
        assert np.max(diff) < 0.05


class TestInputValidation:
    @_python
    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_random_seed_no_nan(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases, omegas, knm, alpha = _problem(rng, n=5)
        out = upde_run(
            phases,
            omegas,
            knm,
            alpha,
            zeta=0.2,
            psi=0.3,
            dt=0.01,
            n_steps=20,
            method="rk4",
        )
        assert np.all(np.isfinite(out))

    @_python
    def test_zero_n_steps(self) -> None:
        phases, omegas, knm, alpha = _problem(np.random.default_rng(3), n=3)
        out = upde_run(
            phases,
            omegas,
            knm,
            alpha,
            zeta=0.0,
            psi=0.0,
            dt=0.01,
            n_steps=0,
        )
        # With n_steps=0 nothing is integrated; output is a wrap-only
        # copy of the input.
        np.testing.assert_allclose(out, phases % TWO_PI, atol=1e-15)


class TestRK45Tolerances:
    @_python
    def test_tighter_tolerance_smaller_error(self) -> None:
        """Tightening ``atol/rtol`` should narrow the gap between
        RK45 and a high-sub-step RK4 reference."""
        rng = np.random.default_rng(41)
        n = 3
        phases, omegas, knm, alpha = _problem(rng, n=n)
        ref = upde_run(
            phases,
            omegas,
            knm,
            alpha,
            zeta=0.1,
            psi=0.0,
            dt=0.001,
            n_steps=500,
            method="rk4",
        )
        loose = upde_run(
            phases,
            omegas,
            knm,
            alpha,
            zeta=0.1,
            psi=0.0,
            dt=0.01,
            n_steps=50,
            method="rk45",
            atol=1e-2,
            rtol=1e-2,
        )
        tight = upde_run(
            phases,
            omegas,
            knm,
            alpha,
            zeta=0.1,
            psi=0.0,
            dt=0.01,
            n_steps=50,
            method="rk45",
            atol=1e-10,
            rtol=1e-10,
        )
        d_loose = float(np.max(np.abs(loose - ref)))
        d_tight = float(np.max(np.abs(tight - ref)))
        assert d_tight <= d_loose + 1e-12
