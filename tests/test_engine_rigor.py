# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Comprehensive rigor tests for auxiliary engines

"""Dedicated tests for HypergraphEngine, market module, envelope solver,
adjoint gradients, and DelayedEngine. These modules had scattered test
coverage — this file provides focused, structured validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.adjoint import cost_R, gradient_knm_fd
from scpn_phase_orchestrator.upde.delay import DelayBuffer, DelayedEngine
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.envelope import (
    envelope_modulation_depth,
    extract_envelope,
)
from scpn_phase_orchestrator.upde.hypergraph import Hyperedge, HypergraphEngine
from scpn_phase_orchestrator.upde.market import (
    detect_regimes,
    extract_phase,
    market_order_parameter,
)
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


# ── HypergraphEngine ────────────────────────────────────────────────────


class TestHypergraphEngine:
    def test_pairwise_via_knm(self) -> None:
        """Pairwise coupling via knm parameter works."""
        n = 4
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-1, 1, n)
        knm = np.ones((n, n)) * 0.5
        np.fill_diagonal(knm, 0.0)
        eng = HypergraphEngine(n, dt=0.01)
        out = eng.step(phases, omegas, pairwise_knm=knm)
        assert len(out) == n
        assert np.all(np.isfinite(out))

    def test_3body_hyperedges(self) -> None:
        n = 4
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-1, 1, n)
        edges = [Hyperedge((0, 1, 2), 1.0)]
        eng = HypergraphEngine(n, dt=0.01, hyperedges=edges)
        out = eng.step(phases, omegas)
        assert np.all(np.isfinite(out))
        assert len(out) == n

    def test_no_edges_free_rotation(self) -> None:
        n = 3
        omegas = np.array([1.0, 2.0, 3.0])
        phases = np.zeros(n)
        eng = HypergraphEngine(n, dt=0.01)
        out = eng.step(phases, omegas)
        expected = (omegas * 0.01) % TWO_PI
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_output_in_0_2pi(self) -> None:
        n = 5
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-5, 5, n)
        edges = [Hyperedge((0, 1, 2), 2.0), Hyperedge((2, 3, 4), 2.0)]
        eng = HypergraphEngine(n, dt=0.01, hyperedges=edges)
        out = eng.step(phases, omegas)
        assert np.all(out >= 0)
        assert np.all(out < TWO_PI + 1e-10)

    def test_hyperedge_order(self) -> None:
        e2 = Hyperedge((0, 1))
        e3 = Hyperedge((0, 1, 2))
        e4 = Hyperedge((0, 1, 2, 3))
        assert e2.order == 2
        assert e3.order == 3
        assert e4.order == 4


# ── Market module ────────────────────────────────────────────────────────


class TestMarketModule:
    def test_extract_phase_shape(self) -> None:
        series = np.sin(np.linspace(0, 10 * np.pi, 200))
        phase = extract_phase(series)
        assert phase.shape == series.shape
        assert np.all(phase >= 0)
        assert np.all(phase < TWO_PI + 1e-10)

    def test_extract_phase_2d(self) -> None:
        rng = np.random.default_rng(0)
        series = rng.standard_normal((100, 3))
        phase = extract_phase(series)
        assert phase.shape == (100, 3)

    def test_market_order_parameter_shape(self) -> None:
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, (50, 4))
        R = market_order_parameter(phases)
        assert R.shape == (50,)
        assert np.all(R >= 0)
        assert np.all(R <= 1.0 + 1e-6)

    def test_sync_phases_r_one(self) -> None:
        phases = np.full((20, 5), 1.0)
        R = market_order_parameter(phases)
        np.testing.assert_allclose(R, 1.0, atol=1e-10)

    def test_detect_regimes_shape(self) -> None:
        R = np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.5, 0.7, 0.9])
        regimes = detect_regimes(R)
        assert len(regimes) == len(R)


# ── Envelope solver ──────────────────────────────────────────────────────


class TestEnvelopeSolver:
    def test_extract_envelope_shape(self) -> None:
        amps = np.abs(np.sin(np.linspace(0, 4 * np.pi, 100)))
        env = extract_envelope(amps, window=5)
        assert env.shape == amps.shape

    def test_extract_envelope_nonneg(self) -> None:
        rng = np.random.default_rng(0)
        amps = np.abs(rng.standard_normal(100))
        env = extract_envelope(amps, window=10)
        assert np.all(env >= 0)

    def test_extract_envelope_empty(self) -> None:
        env = extract_envelope(np.array([]))
        assert env.size == 0

    def test_window_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            extract_envelope(np.ones(10), window=0)

    def test_modulation_depth_bounded(self) -> None:
        amps = np.abs(np.sin(np.linspace(0, 4 * np.pi, 200))) + 0.5
        depth = envelope_modulation_depth(amps)
        assert 0.0 <= depth <= 1.0

    def test_constant_signal_zero_depth(self) -> None:
        amps = np.ones(50)
        depth = envelope_modulation_depth(amps)
        assert depth < 0.01


# ── Adjoint gradient ────────────────────────────────────────────────────


class TestAdjointGradient:
    def test_cost_r_sync_zero(self) -> None:
        phases = np.full(5, 1.0)
        assert cost_R(phases) < 0.01

    def test_cost_r_random_positive(self) -> None:
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, 50)
        c = cost_R(phases)
        assert 0.0 <= c <= 1.0

    def test_gradient_shape(self) -> None:
        n = 3
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-1, 1, n)
        knm = np.ones((n, n)) * 0.5
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        grad = gradient_knm_fd(eng, phases, omegas, knm, alpha, n_steps=20)
        assert grad.shape == (n, n)
        assert np.all(np.isfinite(grad))

    def test_gradient_zero_diagonal(self) -> None:
        n = 3
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-1, 1, n)
        knm = np.ones((n, n)) * 0.5
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        grad = gradient_knm_fd(eng, phases, omegas, knm, alpha, n_steps=20)
        np.testing.assert_array_equal(np.diag(grad), 0.0)


# ── DelayBuffer + DelayedEngine ──────────────────────────────────────────


class TestDelayBuffer:
    def test_push_and_get(self) -> None:
        buf = DelayBuffer(3, max_delay_steps=5)
        for i in range(5):
            buf.push(np.full(3, float(i)))
        delayed = buf.get_delayed(2)
        assert delayed is not None
        np.testing.assert_array_equal(delayed, [3.0, 3.0, 3.0])

    def test_get_before_fill_returns_none(self) -> None:
        buf = DelayBuffer(2, max_delay_steps=10)
        buf.push(np.zeros(2))
        assert buf.get_delayed(5) is None

    def test_length(self) -> None:
        buf = DelayBuffer(2, max_delay_steps=10)
        assert buf.length == 0
        buf.push(np.zeros(2))
        assert buf.length == 1

    def test_clear(self) -> None:
        buf = DelayBuffer(2, max_delay_steps=5)
        buf.push(np.ones(2))
        buf.clear()
        assert buf.length == 0

    def test_invalid_max_delay(self) -> None:
        with pytest.raises(ValueError):
            DelayBuffer(2, max_delay_steps=0)


class TestDelayedEngine:
    def test_step_finite(self) -> None:
        n = 4
        eng = DelayedEngine(n, dt=0.01, delay_steps=3)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-1, 1, n)
        knm = np.ones((n, n)) * 0.5
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(10):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(np.isfinite(phases))

    def test_zero_delay_like_standard(self) -> None:
        """delay_steps=1 (minimal) should behave near standard."""
        n = 4
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.zeros(n)
        knm = np.ones((n, n)) * 2.0
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        eng_d = DelayedEngine(n, dt=0.01, delay_steps=1)
        eng_s = UPDEEngine(n, dt=0.01)
        p_d, p_s = phases.copy(), phases.copy()
        # Prime delay buffer
        for _ in range(5):
            p_d = eng_d.step(p_d, omegas, knm, 0.0, 0.0, alpha)
            p_s = eng_s.step(p_s, omegas, knm, 0.0, 0.0, alpha)
        # After priming, both should converge similarly
        for _ in range(200):
            p_d = eng_d.step(p_d, omegas, knm, 0.0, 0.0, alpha)
            p_s = eng_s.step(p_s, omegas, knm, 0.0, 0.0, alpha)
        R_d, _ = compute_order_parameter(p_d)
        R_s, _ = compute_order_parameter(p_s)
        assert abs(R_d - R_s) < 0.2
