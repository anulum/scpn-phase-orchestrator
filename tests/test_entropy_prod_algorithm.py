# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for entropy production rate

"""Algorithmic properties of :func:`entropy_production_rate`.

Covers: non-negativity, zero at synchronised fixed points, analytical
identity against the explicit sum form, scaling with ``dt`` and
``alpha``, input-edge cases (``n=0``, ``dt≤0``), and Hypothesis
property testing.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor import entropy_prod as ep_mod
from scpn_phase_orchestrator.monitor.entropy_prod import (
    entropy_production_rate,
)

TWO_PI = 2.0 * math.pi


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = ep_mod.ACTIVE_BACKEND
        ep_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            ep_mod.ACTIVE_BACKEND = prev

    return wrapper


def _random_problem(rng: np.random.Generator, n: int):
    phases = rng.uniform(0.0, TWO_PI, size=n)
    omegas = rng.normal(0.0, 0.3, size=n)
    knm = rng.uniform(0.2, 1.0, size=(n, n))
    np.fill_diagonal(knm, 0.0)
    return phases, omegas, knm


class TestNonNegativity:
    @_python
    def test_random_inputs_non_negative(self):
        for seed in range(10):
            phases, omegas, knm = _random_problem(np.random.default_rng(seed), n=6)
            sigma = entropy_production_rate(phases, omegas, knm, alpha=0.5, dt=0.01)
            assert sigma >= 0.0


class TestFixedPoint:
    @_python
    def test_zero_frequency_sync_is_zero(self):
        """Synchronised phases + ω = 0 → Σ = 0."""
        phases = np.full(5, 0.3)
        omegas = np.zeros(5)
        knm = np.full((5, 5), 0.7)
        np.fill_diagonal(knm, 0.0)
        sigma = entropy_production_rate(phases, omegas, knm, 0.6, 0.01)
        assert sigma < 1e-20

    @_python
    def test_uncoupled_equal_omegas_positive(self):
        """Uncoupled oscillators with non-zero ω → Σ = n · ω² · dt."""
        n = 4
        phases = np.zeros(n)
        omegas = np.full(n, 0.5)
        knm = np.zeros((n, n))
        sigma = entropy_production_rate(phases, omegas, knm, 0.0, 0.01)
        expected = n * (0.5**2) * 0.01
        assert abs(sigma - expected) < 1e-15


class TestAnalyticalIdentity:
    @_python
    def test_matches_explicit_sum(self):
        """``Σ == Σ_i (ω_i + α/N Σ_j K_ij sin(θ_j − θ_i))² · dt``."""
        rng = np.random.default_rng(7)
        n = 5
        phases, omegas, knm = _random_problem(rng, n)
        alpha, dt = 0.4, 0.02

        expected = 0.0
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += knm[i, j] * math.sin(phases[j] - phases[i])
            d = omegas[i] + (alpha / n) * s
            expected += d * d
        expected *= dt

        got = entropy_production_rate(phases, omegas, knm, alpha, dt)
        assert abs(got - expected) < 1e-12


class TestScaling:
    @_python
    def test_dt_scales_linearly(self):
        rng = np.random.default_rng(11)
        phases, omegas, knm = _random_problem(rng, n=5)
        s1 = entropy_production_rate(phases, omegas, knm, 0.5, 0.01)
        s2 = entropy_production_rate(phases, omegas, knm, 0.5, 0.02)
        # Σ is linear in dt by construction.
        assert abs(s2 / s1 - 2.0) < 1e-10

    @_python
    def test_alpha_zero_depends_only_on_omegas(self):
        rng = np.random.default_rng(17)
        phases, omegas, knm = _random_problem(rng, n=6)
        # With α = 0, Σ = Σ_i ω_i² · dt — independent of knm / phases.
        s_with_knm = entropy_production_rate(phases, omegas, knm, 0.0, 0.02)
        s_no_knm = entropy_production_rate(
            phases, omegas, np.zeros_like(knm), 0.0, 0.02
        )
        assert abs(s_with_knm - s_no_knm) < 1e-15


class TestEdgeCases:
    @_python
    def test_zero_n_returns_zero(self):
        assert (
            entropy_production_rate(
                np.array([]), np.array([]), np.zeros((0, 0)), 0.5, 0.01
            )
            == 0.0
        )

    @_python
    def test_non_positive_dt_returns_zero(self):
        phases = np.zeros(3)
        omegas = np.ones(3)
        knm = np.zeros((3, 3))
        assert entropy_production_rate(phases, omegas, knm, 0.5, 0.0) == 0.0
        assert (
            entropy_production_rate(
                phases,
                omegas,
                knm,
                0.5,
                -0.1,
            )
            == 0.0
        )


class TestInputValidation:
    @pytest.mark.parametrize(
        ("field", "bad_value", "match"),
        [
            ("phases", np.zeros((3, 1), dtype=np.float64), "phases shape"),
            ("omegas", np.zeros((4,), dtype=np.float64), "omegas shape"),
            ("knm", np.zeros((3, 2), dtype=np.float64), "knm shape"),
        ],
    )
    def test_rejects_shape_mismatch(
        self,
        field: str,
        bad_value: np.ndarray,
        match: str,
    ) -> None:
        values = {
            "phases": np.zeros(3, dtype=np.float64),
            "omegas": np.ones(3, dtype=np.float64),
            "knm": np.zeros((3, 3), dtype=np.float64),
        }
        values[field] = bad_value

        with pytest.raises(ValueError, match=match):
            entropy_production_rate(
                values["phases"],
                values["omegas"],
                values["knm"],
                alpha=1.0,
                dt=0.01,
            )

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("phases", np.nan),
            ("omegas", np.inf),
            ("knm", np.nan),
        ],
    )
    def test_rejects_non_finite_arrays(
        self,
        field: str,
        bad_value: float,
    ) -> None:
        phases = np.zeros(3, dtype=np.float64)
        omegas = np.ones(3, dtype=np.float64)
        knm = np.zeros((3, 3), dtype=np.float64)
        if field == "knm":
            knm[0, 1] = bad_value
        else:
            locals()[field][0] = bad_value

        with pytest.raises(ValueError, match=field):
            entropy_production_rate(phases, omegas, knm, alpha=1.0, dt=0.01)

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("alpha", False),
            ("alpha", np.nan),
            ("alpha", np.inf),
            ("alpha", "1.0"),
            ("dt", False),
            ("dt", np.nan),
            ("dt", np.inf),
            ("dt", "0.01"),
        ],
    )
    def test_rejects_invalid_scalar_inputs(
        self,
        field: str,
        bad_value: Any,
    ) -> None:
        kwargs = {"alpha": 1.0, "dt": 0.01}
        kwargs[field] = bad_value

        with pytest.raises(ValueError, match=field):
            entropy_production_rate(
                np.zeros(3, dtype=np.float64),
                np.ones(3, dtype=np.float64),
                np.zeros((3, 3), dtype=np.float64),
                alpha=kwargs["alpha"],
                dt=kwargs["dt"],
            )

    def test_accepts_array_like_float_buffers(self) -> None:
        rate = entropy_production_rate(
            [0.0, 0.1],
            [1.0, -1.0],
            [[0.0, 0.2], [0.2, 0.0]],
            alpha=1.0,
            dt=0.01,
        )

        assert math.isfinite(rate)
        assert rate >= 0.0


class TestHypothesis:
    @_python
    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_random_input_finite_non_negative(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases, omegas, knm = _random_problem(rng, n=n)
        sigma = entropy_production_rate(phases, omegas, knm, alpha=0.5, dt=0.01)
        assert math.isfinite(sigma)
        assert sigma >= 0.0


class TestDispatcherSurface:
    def test_python_always_available(self):
        assert "python" in ep_mod.AVAILABLE_BACKENDS

    def test_active_backend_first(self):
        assert ep_mod.AVAILABLE_BACKENDS[0] == ep_mod.ACTIVE_BACKEND
