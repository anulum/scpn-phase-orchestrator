# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for the Hodge decomposition

"""Algebraic and topological properties of :func:`hodge_decomposition`.

Covered: reconstruction of the coupling current from the three flow
components; antisymmetry and L²-orthogonality; the gradient being a
node-potential difference; divergence-free curl and harmonic flows; the
first Betti number on canonical complexes (filled triangle, triangle-free
cycle, tree, complete graph); flow built from the symmetric coupling
part; global-phase-shift invariance; scale covariance; explicit triangle
sets; and Hypothesis invariants.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.coupling import hodge as h_mod
from scpn_phase_orchestrator.coupling.hodge import hodge_decomposition

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


def _cycle_knm() -> tuple[np.ndarray, np.ndarray]:
    """Triangle-free 4-cycle 0-1-2-3-0 (no diagonals)."""
    knm = np.zeros((4, 4), dtype=np.float64)
    for i, j in ((0, 1), (1, 2), (2, 3), (0, 3)):
        knm[i, j] = knm[j, i] = 1.0
    return knm, np.array([0.0, 0.7, 1.9, 2.8], dtype=np.float64)


def _flow_inner(a: np.ndarray, b: np.ndarray) -> float:
    upper = np.triu_indices(a.shape[0], k=1)
    return float(np.sum(a[upper] * b[upper]))


class TestHodge:
    @_python
    def test_components_reconstruct_flow(self):
        rng = np.random.default_rng(2)
        n = 8
        k = rng.normal(0, 1, (n, n))
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases)
        np.testing.assert_allclose(
            res.gradient + res.curl + res.harmonic, res.flow, atol=1e-12
        )

    @_python
    def test_flow_uses_symmetric_coupling_part(self):
        rng = np.random.default_rng(5)
        n = 6
        k = rng.normal(0, 1, (n, n))
        phases = rng.uniform(0, TWO_PI, n)
        k_sym = 0.5 * (k + k.T)
        expected = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                expected[i, j] = k_sym[i, j] * np.sin(phases[j] - phases[i])
        np.fill_diagonal(expected, 0.0)
        res = hodge_decomposition(k, phases)
        np.testing.assert_allclose(res.flow, expected, atol=1e-12)

    @_python
    def test_pure_antisymmetric_coupling_has_zero_flow(self):
        rng = np.random.default_rng(1)
        n = 5
        k = rng.normal(0, 1, (n, n))
        k = 0.5 * (k - k.T)  # K_sym = 0 → no edges, zero current
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases)
        np.testing.assert_allclose(res.flow, 0.0, atol=1e-15)
        np.testing.assert_allclose(res.gradient, 0.0, atol=1e-15)
        np.testing.assert_allclose(res.curl, 0.0, atol=1e-15)
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-15)

    @_python
    def test_components_are_antisymmetric(self):
        rng = np.random.default_rng(3)
        n = 7
        k = rng.normal(0, 1, (n, n))
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases)
        for component in (res.gradient, res.curl, res.harmonic, res.flow):
            np.testing.assert_allclose(component, -component.T, atol=1e-12)

    @_python
    def test_components_are_l2_orthogonal(self):
        rng = np.random.default_rng(11)
        n = 8
        k = rng.normal(0, 1, (n, n))
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases)
        assert abs(_flow_inner(res.gradient, res.curl)) < 1e-10
        assert abs(_flow_inner(res.gradient, res.harmonic)) < 1e-10
        assert abs(_flow_inner(res.curl, res.harmonic)) < 1e-10

    @_python
    def test_gradient_is_node_potential_difference(self):
        rng = np.random.default_rng(13)
        n = 6
        k = rng.normal(0, 1, (n, n))
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases)
        for i in range(n):
            for j in range(n):
                if res.flow[i, j] != 0.0:
                    expected = res.potential[j] - res.potential[i]
                    assert abs(res.gradient[i, j] - expected) < 1e-10

    @_python
    def test_curl_and_harmonic_are_divergence_free(self):
        rng = np.random.default_rng(17)
        n = 8
        k = rng.normal(0, 1, (n, n))
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases)
        # Divergence of an antisymmetric flow at node i is its row sum.
        np.testing.assert_allclose(res.curl.sum(axis=1), 0.0, atol=1e-10)
        np.testing.assert_allclose(res.harmonic.sum(axis=1), 0.0, atol=1e-10)

    @_python
    def test_complete_graph_has_zero_betti_and_harmonic(self):
        rng = np.random.default_rng(19)
        n = 6
        k = rng.normal(0, 1, (n, n))
        k = 0.5 * (k + k.T)
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases)
        assert res.betti_one == 0
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-10)

    @_python
    def test_filled_triangle_has_zero_harmonic(self):
        k = np.ones((3, 3), dtype=np.float64) - np.eye(3)
        phases = np.array([0.0, 1.0, 2.3])
        res = hodge_decomposition(k, phases)
        assert res.betti_one == 0
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-9)

    @_python
    def test_triangle_free_cycle_is_purely_harmonic(self):
        """The square cycle has β₁ = 1: zero curl, non-trivial harmonic."""
        knm, phases = _cycle_knm()
        res = hodge_decomposition(knm, phases)
        assert res.betti_one == 1
        np.testing.assert_allclose(res.curl, 0.0, atol=1e-12)
        assert np.max(np.abs(res.harmonic)) > 1e-2
        np.testing.assert_allclose(
            res.gradient + res.curl + res.harmonic, res.flow, atol=1e-12
        )

    @_python
    def test_tree_is_pure_gradient(self):
        knm = np.zeros((4, 4), dtype=np.float64)
        for i, j in ((0, 1), (1, 2), (2, 3)):
            knm[i, j] = knm[j, i] = 1.0
        phases = np.array([0.1, 0.9, 1.7, 2.5])
        res = hodge_decomposition(knm, phases)
        assert res.betti_one == 0
        np.testing.assert_allclose(res.curl, 0.0, atol=1e-12)
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-9)
        np.testing.assert_allclose(res.gradient, res.flow, atol=1e-9)

    @_python
    def test_explicit_empty_triangle_set_exposes_cycle_harmonic(self):
        """Omitting triangles from a complete graph turns filled cycles
        into harmonic content (β₁ counts independent cycles)."""
        rng = np.random.default_rng(23)
        n = 4
        k = rng.normal(0, 1, (n, n))
        k = 0.5 * (k + k.T)
        phases = rng.uniform(0, TWO_PI, n)
        filled = hodge_decomposition(k, phases)
        hollow = hodge_decomposition(k, phases, triangles=[])
        assert filled.betti_one == 0
        # complete graph K4 without triangles: β₁ = 6 - 3 = 3
        assert hollow.betti_one == 3
        np.testing.assert_allclose(hollow.curl, 0.0, atol=1e-12)
        assert np.max(np.abs(hollow.harmonic)) > 1e-3

    @_python
    def test_explicit_triangle_subset_matches_default_on_complete_graph(self):
        rng = np.random.default_rng(29)
        n = 4
        k = rng.normal(0, 1, (n, n))
        k = 0.5 * (k + k.T)
        phases = rng.uniform(0, TWO_PI, n)
        all_triangles = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        explicit = hodge_decomposition(k, phases, triangles=all_triangles)
        default = hodge_decomposition(k, phases)
        np.testing.assert_allclose(explicit.gradient, default.gradient, atol=1e-12)
        np.testing.assert_allclose(explicit.curl, default.curl, atol=1e-12)
        np.testing.assert_allclose(explicit.harmonic, default.harmonic, atol=1e-12)

    @_python
    def test_global_phase_shift_invariance(self):
        rng = np.random.default_rng(31)
        n = 6
        k = rng.normal(0, 1, (n, n))
        phases = rng.uniform(-np.pi, np.pi, n)
        base = hodge_decomposition(k, phases)
        shifted = hodge_decomposition(k, phases + 12.5)
        np.testing.assert_allclose(shifted.gradient, base.gradient, atol=1e-12)
        np.testing.assert_allclose(shifted.curl, base.curl, atol=1e-12)
        np.testing.assert_allclose(shifted.harmonic, base.harmonic, atol=1e-12)

    @_python
    def test_scale_covariance(self):
        rng = np.random.default_rng(37)
        n = 6
        k = rng.normal(0, 1, (n, n))
        phases = rng.uniform(0, TWO_PI, n)
        base = hodge_decomposition(k, phases)
        scaled = hodge_decomposition(2.5 * k, phases)
        np.testing.assert_allclose(scaled.gradient, 2.5 * base.gradient, rtol=1e-10)
        np.testing.assert_allclose(scaled.curl, 2.5 * base.curl, rtol=1e-10)
        np.testing.assert_allclose(scaled.harmonic, 2.5 * base.harmonic, atol=1e-10)

    @_python
    def test_empty_input(self):
        res = hodge_decomposition(np.zeros((0, 0)), np.array([]))
        assert res.gradient.shape == (0, 0)
        assert res.curl.size == 0
        assert res.harmonic.size == 0
        assert res.flow.size == 0
        assert res.potential.size == 0
        assert res.betti_one == 0

    @_python
    def test_single_oscillator_has_no_flow(self):
        res = hodge_decomposition(np.array([[0.0]]), np.array([1.0]))
        assert res.gradient.shape == (1, 1)
        np.testing.assert_allclose(res.flow, 0.0)
        assert res.betti_one == 0

    @_python
    @pytest.mark.parametrize(
        ("knm", "phases", "match"),
        [
            (np.zeros((2, 2)), np.array([True, False]), "phases"),
            (np.array([[True, False], [False, True]]), np.zeros(2), "knm"),
            (np.zeros((2, 3)), np.zeros(2), "knm"),
            (np.zeros((2, 2)), np.array([0.0, np.nan]), "phases"),
            (np.zeros((3, 3)), np.zeros(2), "knm"),
        ],
    )
    def test_rejects_invalid_inputs(self, knm, phases, match):
        with pytest.raises(ValueError, match=match):
            hodge_decomposition(knm, phases)

    @_python
    @pytest.mark.parametrize(
        ("triangles", "match"),
        [
            ([(0, 1)], "exactly three nodes"),
            ([(0, 1, 2, 3)], "exactly three nodes"),
            ([(0, 0, 1)], "distinct"),
            ([(0, 1, 9)], r"\[0, 4\)"),
            ([(0, 1, True)], "integer"),
            ([(0, 1, 2), (2, 1, 0)], "duplicate"),
        ],
    )
    def test_rejects_invalid_triangles(self, triangles, match):
        k = np.ones((4, 4)) - np.eye(4)
        phases = np.zeros(4)
        with pytest.raises(ValueError, match=match):
            hodge_decomposition(k, phases, triangles=triangles)

    @_python
    def test_triangle_referencing_absent_edge_is_rejected(self):
        # Square cycle has no edge (0, 2); a triangle needing it must fail.
        knm, phases = _cycle_knm()
        with pytest.raises(ValueError, match="all three edges"):
            hodge_decomposition(knm, phases, triangles=[(0, 1, 2)])


class TestHypothesis:
    @_python
    @given(
        n=st.integers(min_value=2, max_value=18),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=12,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_shape_finite_and_reconstruction(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        k = rng.normal(0, 1, (n, n))
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases)
        for component in (res.gradient, res.curl, res.harmonic, res.flow):
            assert component.shape == (n, n)
            assert np.all(np.isfinite(component))
        np.testing.assert_allclose(
            res.gradient + res.curl + res.harmonic, res.flow, atol=1e-9
        )
        assert res.betti_one >= 0


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert h_mod.AVAILABLE_BACKENDS
        assert "python" in h_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert h_mod.AVAILABLE_BACKENDS[0] == h_mod.ACTIVE_BACKEND
