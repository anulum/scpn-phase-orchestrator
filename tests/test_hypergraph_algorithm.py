# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for hypergraph Kuramoto

"""Algorithmic properties of :class:`HypergraphEngine`.

Covered: edge-management API (``add_edge``, ``add_all_to_all``,
``n_edges``); phase wrap in ``[0, 2π)``; zero-coupling limit
reduces to pure-ω rotation; ``k = 2`` hyperedges reproduce
standard Kuramoto pairwise terms; Skardal-Arenas 2019 three-body
all-to-all limit drives sync from asymmetric ICs; empty hypergraph
matches standalone pairwise Kuramoto; order-parameter helper;
Hypothesis invariants.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import hypergraph as h_mod
from scpn_phase_orchestrator.upde.hypergraph import (
    Hyperedge,
    HypergraphEngine,
)

TWO_PI = 2.0 * math.pi


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = h_mod.ACTIVE_BACKEND
        h_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            h_mod.ACTIVE_BACKEND = prev

    return wrapper


class TestHyperedge:
    def test_order_reports_len(self):
        e = Hyperedge(nodes=(0, 1, 2, 3), strength=0.7)
        assert e.order == 4
        assert e.strength == 0.7


class TestEngineAPI:
    def test_n_edges_starts_zero(self):
        eng = HypergraphEngine(8, 0.01)
        assert eng.n_edges == 0

    def test_add_edge_grows_count(self):
        eng = HypergraphEngine(8, 0.01)
        eng.add_edge((0, 1, 2))
        eng.add_edge((3, 4, 5, 6), strength=0.2)
        assert eng.n_edges == 2

    def test_add_all_to_all_order_3_count(self):
        """C(5, 3) = 10 triadic edges on N=5."""
        eng = HypergraphEngine(5, 0.01)
        eng.add_all_to_all(order=3, strength=0.1)
        assert eng.n_edges == 10


class TestStep:
    @_python
    def test_phases_wrap_in_two_pi(self):
        rng = np.random.default_rng(0)
        n = 8
        theta = rng.uniform(0, TWO_PI, n)
        omegas = rng.normal(1.0, 0.2, n)
        eng = HypergraphEngine(n, 0.1)
        eng.add_edge((0, 1, 2), strength=0.5)
        new_ph = eng.step(theta, omegas)
        assert np.all(new_ph >= 0.0)
        assert np.all(new_ph < TWO_PI + 1e-12)

    @_python
    def test_zero_coupling_pure_rotation(self):
        """No edges + no pairwise + no drive → θ(t+dt) =
        (θ + dt·ω) mod 2π."""
        n = 4
        theta = np.array([0.3, 1.1, 2.0, 3.0])
        omega = np.array([0.1, -0.2, 0.05, 0.0])
        eng = HypergraphEngine(n, 0.01)
        new_ph = eng.step(theta, omega)
        expected = (theta + 0.01 * omega) % TWO_PI
        np.testing.assert_allclose(new_ph, expected, atol=1e-14)


class TestKuramotoLimit:
    @_python
    def test_pairwise_only_reproduces_standard_kuramoto(self):
        """When no hyperedges are registered and only
        ``pairwise_knm`` is supplied, the engine must reproduce
        the first-order Kuramoto Euler step exactly."""
        n = 5
        rng = np.random.default_rng(2)
        theta = rng.uniform(0, TWO_PI, n)
        omega = rng.normal(1.0, 0.2, n)
        knm = rng.uniform(0, 0.3, (n, n))
        np.fill_diagonal(knm, 0.0)
        dt = 0.01
        eng = HypergraphEngine(n, dt)
        new_ph = eng.step(theta, omega, pairwise_knm=knm)
        # Direct Kuramoto reference
        diff = theta[np.newaxis, :] - theta[:, np.newaxis]
        coupling = np.sum(knm * np.sin(diff), axis=1)
        expected = (theta + dt * (omega + coupling)) % TWO_PI
        np.testing.assert_allclose(new_ph, expected, atol=1e-12)


class TestTriadicCoupling:
    @_python
    def test_triadic_preserves_synchronised_state(self):
        """For σ > 0, the fully-synchronised state
        ``θ_i = θ_j = …`` is a fixed point of the 3-body
        Skardal-Arenas dynamics — ``phase_sum − k·θ_m = 0`` for
        every node, so the derivative vanishes and the state is
        preserved exactly over any number of Euler steps."""
        n = 8
        theta = np.full(n, 1.3)  # perfectly synced
        omegas = np.zeros(n)
        dt = 0.01
        eng = HypergraphEngine(n, dt)
        eng.add_all_to_all(order=3, strength=0.8)
        final = eng.run(theta, omegas, n_steps=200)
        # The fixed-point value of the derivative is exactly zero, so
        # the Euler update keeps the ensemble fully synchronised.
        np.testing.assert_allclose(final, theta, atol=1e-12)
        assert eng.order_parameter(final) == pytest.approx(1.0, abs=1e-12)

    @_python
    def test_triadic_from_small_noise_stays_coherent(self):
        """Starting from ``θ_i ≈ θ₀`` with a small perturbation,
        σ > 0 triadic all-to-all coupling must keep the ensemble
        near-synchronised over a few-hundred-step horizon (the
        synced fixed point is locally stable). No claim is made
        about random-IC convergence: from fully random phases
        triadic dynamics can settle into ``k``-cluster states
        (Skardal & Arenas 2019) that keep ``R`` low, so the
        test constrains only the local-stability direction."""
        rng = np.random.default_rng(3)
        n = 8
        theta = 1.3 + 0.05 * rng.standard_normal(n)
        omegas = np.zeros(n)
        dt = 0.01
        eng = HypergraphEngine(n, dt)
        eng.add_all_to_all(order=3, strength=0.8)
        r0 = eng.order_parameter(theta)
        final = eng.run(theta, omegas, n_steps=200)
        assert eng.order_parameter(final) >= r0 - 1e-3


class TestExternalDrive:
    @_python
    def test_zeta_psi_acts_as_forcing(self):
        """Nonzero ``ζ, ψ`` contributes ``ζ·sin(ψ − θ)`` to the
        derivative. With no coupling, no edges and matching
        ``ψ = θ₀``, the forcing vanishes at t=0; with a ψ offset
        the trajectory diverges from the free rotation."""
        n = 3
        theta = np.array([0.0, 0.0, 0.0])
        omega = np.zeros(n)
        dt = 0.01
        eng = HypergraphEngine(n, dt)
        # No forcing baseline.
        no_drive = eng.step(theta, omega)
        # Nonzero drive.
        drive = eng.step(theta, omega, zeta=1.0, psi=np.pi / 2)
        # The drive must move the phases by dt·ζ·sin(ψ−0) = dt · 1.
        assert np.all(np.abs(drive - no_drive) > 1e-6)


class TestHypothesis:
    @_python
    @given(
        n=st.integers(min_value=3, max_value=6),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8, deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_run_finite_output(self, n, seed):
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0, TWO_PI, n)
        omega = rng.normal(0.5, 0.2, n)
        knm = rng.uniform(0, 0.2, (n, n))
        np.fill_diagonal(knm, 0.0)
        eng = HypergraphEngine(n, 0.01)
        eng.add_edge((0, 1, 2), strength=0.3)
        fin = eng.run(theta, omega, n_steps=50, pairwise_knm=knm)
        assert np.all(np.isfinite(fin))
        assert np.all(fin >= 0.0)
        assert np.all(fin < TWO_PI + 1e-12)


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert h_mod.AVAILABLE_BACKENDS
        assert "python" in h_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert h_mod.AVAILABLE_BACKENDS[0] == h_mod.ACTIVE_BACKEND
