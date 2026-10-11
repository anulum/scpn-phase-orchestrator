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

import math
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling import hodge as h_mod
from scpn_phase_orchestrator.coupling.hodge import hodge_decomposition

FloatArray = NDArray[np.float64]
# Invalid input tests deliberately cross the annotated array boundary.
invalid_input_call = cast(Callable[[object, object], object], hodge_decomposition)

TWO_PI = 2.0 * math.pi


def _cycle_knm() -> tuple[FloatArray, FloatArray]:
    """Triangle-free 4-cycle 0-1-2-3-0 (no diagonals)."""
    knm = np.zeros((4, 4), dtype=np.float64)
    for i, j in ((0, 1), (1, 2), (2, 3), (0, 3)):
        knm[i, j] = knm[j, i] = 1.0
    return knm, np.array([0.0, 0.7, 1.9, 2.8], dtype=np.float64)


def _flow_inner(a: FloatArray, b: FloatArray) -> float:
    upper = np.triu_indices(a.shape[0], k=1)
    return float(np.sum(a[upper] * b[upper]))


class TestHodge:
    """Algebraic projections and topology through explicit public NumPy calls."""

    def test_components_reconstruct_flow(self) -> None:
        """Orthogonal projections reconstruct the original directed current."""
        rng = np.random.default_rng(2)
        n = 8
        k = rng.normal(0, 1, (n, n))
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases, backend="python")
        np.testing.assert_allclose(
            res.gradient + res.curl + res.harmonic, res.flow, atol=1e-12
        )

    def test_flow_uses_symmetric_coupling_part(self) -> None:
        """Directed weights produce the independently calculated alternating flow."""
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
        res = hodge_decomposition(k, phases, backend="python")
        np.testing.assert_allclose(res.flow, expected, atol=1e-12)

    def test_pure_antisymmetric_coupling_has_zero_flow(self) -> None:
        """Purely antisymmetric weights carry no symmetric Kuramoto current."""
        rng = np.random.default_rng(1)
        n = 5
        k = rng.normal(0, 1, (n, n))
        k = 0.5 * (k - k.T)  # K_sym = 0 → no edges, zero current
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases, backend="python")
        np.testing.assert_allclose(res.flow, 0.0, atol=1e-15)
        np.testing.assert_allclose(res.gradient, 0.0, atol=1e-15)
        np.testing.assert_allclose(res.curl, 0.0, atol=1e-15)
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-15)

    def test_components_are_antisymmetric(self) -> None:
        """Every public edge-flow component reverses sign with orientation."""
        rng = np.random.default_rng(3)
        n = 7
        k = rng.normal(0, 1, (n, n))
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases, backend="python")
        for component in (res.gradient, res.curl, res.harmonic, res.flow):
            np.testing.assert_allclose(component, -component.T, atol=1e-12)

    def test_components_are_l2_orthogonal(self) -> None:
        """Gradient, curl and harmonic projections have zero pairwise edge products."""
        rng = np.random.default_rng(11)
        n = 8
        k = rng.normal(0, 1, (n, n))
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases, backend="python")
        assert abs(_flow_inner(res.gradient, res.curl)) < 1e-10
        assert abs(_flow_inner(res.gradient, res.harmonic)) < 1e-10
        assert abs(_flow_inner(res.curl, res.harmonic)) < 1e-10

    def test_gradient_is_node_potential_difference(self) -> None:
        """Every active gradient edge equals its node-potential difference."""
        rng = np.random.default_rng(13)
        n = 6
        k = rng.normal(0, 1, (n, n))
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases, backend="python")
        for i in range(n):
            for j in range(n):
                if res.flow[i, j] != 0.0:
                    expected = res.potential[j] - res.potential[i]
                    assert abs(res.gradient[i, j] - expected) < 1e-10

    def test_curl_and_harmonic_are_divergence_free(self) -> None:
        """Curl and harmonic currents conserve flow at each node."""
        rng = np.random.default_rng(17)
        n = 8
        k = rng.normal(0, 1, (n, n))
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases, backend="python")
        # Divergence of an antisymmetric flow at node i is its row sum.
        np.testing.assert_allclose(res.curl.sum(axis=1), 0.0, atol=1e-10)
        np.testing.assert_allclose(res.harmonic.sum(axis=1), 0.0, atol=1e-10)

    def test_complete_graph_has_zero_betti_and_harmonic(self) -> None:
        """A triangle-filled complete graph has no unfilled cycle."""
        rng = np.random.default_rng(19)
        n = 6
        k = rng.normal(0, 1, (n, n))
        k = 0.5 * (k + k.T)
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases, backend="python")
        assert res.betti_one == 0
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-10)

    def test_filled_triangle_has_zero_harmonic(self) -> None:
        """A filled triangle has no harmonic current."""
        k = np.ones((3, 3), dtype=np.float64) - np.eye(3)
        phases = np.array([0.0, 1.0, 2.3])
        res = hodge_decomposition(k, phases, backend="python")
        assert res.betti_one == 0
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-9)

    def test_triangle_free_cycle_is_purely_harmonic(self) -> None:
        """The square cycle has β₁ = 1: zero curl, non-trivial harmonic."""
        knm, phases = _cycle_knm()
        res = hodge_decomposition(knm, phases, backend="python")
        assert res.betti_one == 1
        np.testing.assert_allclose(res.curl, 0.0, atol=1e-12)
        assert np.max(np.abs(res.harmonic)) > 1e-2
        np.testing.assert_allclose(
            res.gradient + res.curl + res.harmonic, res.flow, atol=1e-12
        )

    def test_tree_is_pure_gradient(self) -> None:
        """A tree has no cycle current beyond its node-potential gradient."""
        knm = np.zeros((4, 4), dtype=np.float64)
        for i, j in ((0, 1), (1, 2), (2, 3)):
            knm[i, j] = knm[j, i] = 1.0
        phases = np.array([0.1, 0.9, 1.7, 2.5])
        res = hodge_decomposition(knm, phases, backend="python")
        assert res.betti_one == 0
        np.testing.assert_allclose(res.curl, 0.0, atol=1e-12)
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-9)
        np.testing.assert_allclose(res.gradient, res.flow, atol=1e-9)

    def test_explicit_empty_triangle_set_exposes_cycle_harmonic(self) -> None:
        """Removing triangle fills exposes the complete graph's unfilled cycles."""
        rng = np.random.default_rng(23)
        n = 4
        k = rng.normal(0, 1, (n, n))
        k = 0.5 * (k + k.T)
        phases = rng.uniform(0, TWO_PI, n)
        filled = hodge_decomposition(k, phases, backend="python")
        hollow = hodge_decomposition(k, phases, triangles=[], backend="python")
        assert filled.betti_one == 0
        # complete graph K4 without triangles: β₁ = 6 - 3 = 3
        assert hollow.betti_one == 3
        np.testing.assert_allclose(hollow.curl, 0.0, atol=1e-12)
        assert np.max(np.abs(hollow.harmonic)) > 1e-3

    def test_explicit_triangle_subset_matches_default_on_complete_graph(self) -> None:
        """An explicit complete triangle list matches automatic simplex discovery."""
        rng = np.random.default_rng(29)
        n = 4
        k = rng.normal(0, 1, (n, n))
        k = 0.5 * (k + k.T)
        phases = rng.uniform(0, TWO_PI, n)
        all_triangles = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        explicit = hodge_decomposition(
            k, phases, triangles=all_triangles, backend="python"
        )
        default = hodge_decomposition(k, phases, backend="python")
        np.testing.assert_allclose(explicit.gradient, default.gradient, atol=1e-12)
        np.testing.assert_allclose(explicit.curl, default.curl, atol=1e-12)
        np.testing.assert_allclose(explicit.harmonic, default.harmonic, atol=1e-12)

    def test_global_phase_shift_invariance(self) -> None:
        """A common phase offset preserves all public NumPy projections."""
        rng = np.random.default_rng(31)
        n = 6
        k = rng.normal(0, 1, (n, n))
        phases = rng.uniform(-np.pi, np.pi, n)
        base = hodge_decomposition(k, phases, backend="python")
        shifted = hodge_decomposition(k, phases + 12.5, backend="python")
        np.testing.assert_allclose(shifted.gradient, base.gradient, atol=1e-12)
        np.testing.assert_allclose(shifted.curl, base.curl, atol=1e-12)
        np.testing.assert_allclose(shifted.harmonic, base.harmonic, atol=1e-12)

    def test_scale_covariance(self) -> None:
        """Scaling weights scales the three projections by the same factor."""
        rng = np.random.default_rng(37)
        n = 6
        k = rng.normal(0, 1, (n, n))
        phases = rng.uniform(0, TWO_PI, n)
        base = hodge_decomposition(k, phases, backend="python")
        scaled = hodge_decomposition(2.5 * k, phases, backend="python")
        np.testing.assert_allclose(scaled.gradient, 2.5 * base.gradient, rtol=1e-10)
        np.testing.assert_allclose(scaled.curl, 2.5 * base.curl, rtol=1e-10)
        np.testing.assert_allclose(scaled.harmonic, 2.5 * base.harmonic, atol=1e-10)

    def test_empty_input(self) -> None:
        """An empty public complex has empty flow and no first homology."""
        res = hodge_decomposition(np.zeros((0, 0)), np.array([]), backend="python")
        assert res.gradient.shape == (0, 0)
        assert res.curl.size == 0
        assert res.harmonic.size == 0
        assert res.flow.size == 0
        assert res.potential.size == 0
        assert res.betti_one == 0

    def test_single_oscillator_has_no_flow(self) -> None:
        """One vertex has no edge current or first homology."""
        res = hodge_decomposition(np.array([[0.0]]), np.array([1.0]), backend="python")
        assert res.gradient.shape == (1, 1)
        np.testing.assert_allclose(res.flow, 0.0)
        assert res.betti_one == 0

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
    def test_rejects_invalid_inputs(
        self, knm: object, phases: object, match: str
    ) -> None:
        """Malformed measured arrays refuse through the public input boundary."""
        with pytest.raises(ValueError, match=match):
            invalid_input_call(knm, phases)

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
    def test_rejects_invalid_triangles(
        self, triangles: list[tuple[int, ...]], match: str
    ) -> None:
        """Malformed simplices refuse through the public topology boundary."""
        k = np.ones((4, 4)) - np.eye(4)
        phases = np.zeros(4)
        with pytest.raises(ValueError, match=match):
            hodge_decomposition(k, phases, triangles=triangles, backend="python")

    def test_triangle_referencing_absent_edge_is_rejected(self) -> None:
        # Square cycle has no edge (0, 2); a triangle needing it must fail.
        """A simplex cannot fill a triangle whose physical edge is absent."""
        knm, phases = _cycle_knm()
        with pytest.raises(ValueError, match="all three edges"):
            hodge_decomposition(knm, phases, triangles=[(0, 1, 2)], backend="python")


class TestHypothesis:
    """Seeded directed graph reconstruction across varying dimensions."""

    @given(
        n=st.integers(min_value=2, max_value=18),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=12,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_shape_finite_and_reconstruction(self, n: int, seed: int) -> None:
        """Seeded directed networks preserve dimensions, finite values and flow."""
        rng = np.random.default_rng(seed)
        k = rng.normal(0, 1, (n, n))
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(k, phases, backend="python")
        for component in (res.gradient, res.curl, res.harmonic, res.flow):
            assert component.shape == (n, n)
            assert np.all(np.isfinite(component))
        np.testing.assert_allclose(
            res.gradient + res.curl + res.harmonic, res.flow, atol=1e-9
        )
        assert res.betti_one >= 0


class TestDispatcherSurface:
    """Runtime declarations verified by actual public computation."""

    def test_available_non_empty(self) -> None:
        """Advertised owners execute the public single-edge decomposition."""
        assert h_mod.AVAILABLE_BACKENDS
        assert "python" in h_mod.AVAILABLE_BACKENDS
        coupling = np.array([[0.0, 1.0], [1.0, 0.0]])
        phases = np.array([0.0, 0.5])
        for owner in h_mod.AVAILABLE_BACKENDS:
            result = hodge_decomposition(coupling, phases, backend=owner)
            assert result.gradient[0, 1] == pytest.approx(np.sin(0.5), abs=1e-12)

    def test_active_is_first(self) -> None:
        """The preferred declared owner executes the public decomposition."""
        assert h_mod.AVAILABLE_BACKENDS[0] == h_mod.ACTIVE_BACKEND
        result = hodge_decomposition(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([0.0, 0.5]),
            backend=h_mod.ACTIVE_BACKEND,
        )
        assert result.gradient[0, 1] == pytest.approx(np.sin(0.5), abs=1e-12)
