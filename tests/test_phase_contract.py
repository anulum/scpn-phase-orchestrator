# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase contract tests

"""Phase-contract invariants: wrapping, order parameter bounds, PLV bounds."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import (
    compute_layer_coherence,
    compute_order_parameter,
    compute_plv,
)

N = 8
RNG = np.random.default_rng(99)


def _make_engine(method="euler"):
    return UPDEEngine(N, dt=0.01, method=method)


def _step_once(engine, phases, omegas=None, knm=None, alpha=None):
    if omegas is None:
        omegas = np.ones(N)
    if knm is None:
        knm = 0.3 * np.ones((N, N))
        np.fill_diagonal(knm, 0.0)
    if alpha is None:
        alpha = np.zeros((N, N))
    return engine.step(phases, omegas, knm, 0.0, 0.0, alpha)


class TestPhaseWrapping:
    """theta in [0, 2*pi) after every step, for all integrators."""

    @pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
    def test_wrapping_after_single_step(self, method):
        engine = _make_engine(method)
        phases = RNG.uniform(0, TWO_PI, N)
        out = _step_once(engine, phases)
        assert np.all(out >= 0.0)
        assert np.all(out < TWO_PI)

    @pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
    def test_wrapping_after_many_steps(self, method):
        engine = _make_engine(method)
        phases = RNG.uniform(0, TWO_PI, N)
        for _ in range(200):
            phases = _step_once(engine, phases)
        assert np.all(phases >= 0.0)
        assert np.all(phases < TWO_PI)

    def test_wrapping_with_large_omega(self):
        engine = _make_engine("euler")
        phases = np.zeros(N)
        omegas = np.full(N, 500.0)
        out = _step_once(engine, phases, omegas=omegas)
        assert np.all(out >= 0.0)
        assert np.all(out < TWO_PI)


class TestOrderParameterBounds:
    """R in [0, 1], psi in [0, 2*pi)."""

    def test_synchronised_phases_R_near_one(self):
        phases = np.full(N, 1.5)
        r, psi = compute_order_parameter(phases)
        assert r == pytest.approx(1.0, abs=1e-12)
        assert 0.0 <= psi < TWO_PI

    def test_uniform_phases_R_near_zero(self):
        phases = np.linspace(0, TWO_PI, N, endpoint=False)
        r, _ = compute_order_parameter(phases)
        assert r < 0.15

    def test_random_phases_R_in_range(self):
        for _ in range(20):
            phases = RNG.uniform(0, TWO_PI, N)
            r, psi = compute_order_parameter(phases)
            assert 0.0 <= r <= 1.0 + 1e-12
            assert 0.0 <= psi < TWO_PI

    def test_single_oscillator_R_one(self):
        r, _ = compute_order_parameter(np.array([3.0]))
        assert r == pytest.approx(1.0, abs=1e-12)


class TestPLVBounds:
    """PLV in [0, 1]."""

    def test_identical_phases_plv_one(self):
        a = RNG.uniform(0, TWO_PI, 50)
        assert compute_plv(a, a) == pytest.approx(1.0, abs=1e-12)

    def test_random_phases_plv_in_range(self):
        for _ in range(20):
            a = RNG.uniform(0, TWO_PI, 50)
            b = RNG.uniform(0, TWO_PI, 50)
            plv = compute_plv(a, b)
            assert 0.0 <= plv <= 1.0 + 1e-12

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="equal-length"):
            compute_plv(np.array([1.0, 2.0]), np.array([1.0]))


class TestLayerCoherence:
    def test_empty_mask_returns_zero(self):
        phases = RNG.uniform(0, TWO_PI, N)
        mask = np.zeros(N, dtype=bool)
        assert compute_layer_coherence(phases, mask) == 0.0

    def test_full_mask_matches_order_param(self):
        phases = RNG.uniform(0, TWO_PI, N)
        mask = np.ones(N, dtype=bool)
        r_layer = compute_layer_coherence(phases, mask)
        r_global, _ = compute_order_parameter(phases)
        assert r_layer == pytest.approx(r_global, abs=1e-12)


class TestRK45AdaptiveDt:
    def test_last_dt_property(self):
        engine = _make_engine("rk45")
        phases = RNG.uniform(0, TWO_PI, N)
        _step_once(engine, phases)
        assert engine.last_dt > 0.0

    def test_rk45_finite_output(self):
        engine = _make_engine("rk45")
        phases = RNG.uniform(0, TWO_PI, N)
        for _ in range(50):
            phases = _step_once(engine, phases)
        assert np.all(np.isfinite(phases))
