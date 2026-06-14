# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for the combinatorial Hodge decomposition

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.hodge import HodgeResult, hodge_decomposition


def _symmetric_coupling(n: int, k: float = 1.0) -> np.ndarray:
    knm = np.full((n, n), k)
    np.fill_diagonal(knm, 0.0)
    return knm


class TestHodgeDecomposition:
    def test_returns_hodge_result_dataclass(self):
        res = hodge_decomposition(_symmetric_coupling(3), np.zeros(3))
        assert isinstance(res, HodgeResult)
        assert res.gradient.shape == (3, 3)
        assert res.curl.shape == (3, 3)
        assert res.harmonic.shape == (3, 3)
        assert res.flow.shape == (3, 3)
        assert res.potential.shape == (3,)
        assert isinstance(res.betti_one, int)

    def test_reconstruction(self):
        """gradient + curl + harmonic = input coupling current."""
        rng = np.random.default_rng(42)
        knm = rng.standard_normal((5, 5))
        phases = rng.uniform(0, 2 * np.pi, 5)
        res = hodge_decomposition(knm, phases)
        np.testing.assert_allclose(
            res.gradient + res.curl + res.harmonic, res.flow, atol=1e-10
        )

    def test_zero_coupling_is_all_zero(self):
        res = hodge_decomposition(np.zeros((4, 4)), np.linspace(0, 1, 4))
        np.testing.assert_allclose(res.flow, 0.0, atol=1e-15)
        np.testing.assert_allclose(res.gradient, 0.0, atol=1e-15)
        np.testing.assert_allclose(res.curl, 0.0, atol=1e-15)
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-15)
        assert res.betti_one == 0

    def test_complete_graph_curl_free_and_harmonic_free(self):
        """A triangle-filled complete graph carries no harmonic flow."""
        knm = _symmetric_coupling(4)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        res = hodge_decomposition(knm, phases)
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-10)
        assert res.betti_one == 0

    def test_two_node_flow_closed_form(self):
        """Two coupled oscillators: a single edge, pure gradient flow."""
        a = 1.7
        phases = np.array([0.25, 1.1])
        knm = np.array([[0.0, a], [a, 0.0]])
        res = hodge_decomposition(knm, phases)
        expected = a * np.sin(phases[1] - phases[0])
        assert res.flow[0, 1] == pytest.approx(expected)
        np.testing.assert_allclose(res.gradient, res.flow, atol=1e-12)
        np.testing.assert_allclose(res.curl, 0.0, atol=1e-12)
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-12)
        assert res.betti_one == 0

    def test_global_phase_shift_invariance(self):
        rng = np.random.default_rng(7)
        knm = rng.standard_normal((5, 5))
        knm = 0.5 * (knm + knm.T)
        phases = rng.uniform(-np.pi, np.pi, 5)
        base = hodge_decomposition(knm, phases)
        shifted = hodge_decomposition(knm, phases + 17.25)
        np.testing.assert_allclose(shifted.gradient, base.gradient, atol=1e-12)
        np.testing.assert_allclose(shifted.curl, base.curl, atol=1e-12)
        np.testing.assert_allclose(shifted.harmonic, base.harmonic, atol=1e-12)

    def test_empty_phases(self):
        res = hodge_decomposition(np.zeros((0, 0)), np.array([]))
        assert res.gradient.size == 0
        assert res.curl.size == 0
        assert res.harmonic.size == 0
        assert res.betti_one == 0

    def test_single_oscillator(self):
        res = hodge_decomposition(np.array([[0.0]]), np.array([1.0]))
        np.testing.assert_allclose(res.flow, 0.0)
        np.testing.assert_allclose(res.gradient, 0.0)
        np.testing.assert_allclose(res.curl, 0.0)

    def test_uniform_scale_scales_all_components(self, monkeypatch):
        """Scaling K_nm by a scalar scales every flow component linearly."""
        import scpn_phase_orchestrator.coupling.hodge as hodge_mod

        monkeypatch.setattr(hodge_mod, "ACTIVE_BACKEND", "python")
        monkeypatch.setattr(hodge_mod, "AVAILABLE_BACKENDS", ["python"])
        rng = np.random.default_rng(123)
        knm = rng.standard_normal((5, 5))
        phases = rng.uniform(0.0, 2 * np.pi, 5)
        base = hodge_decomposition(knm, phases)
        doubled = hodge_decomposition(2.5 * knm, phases)
        np.testing.assert_allclose(doubled.gradient, 2.5 * base.gradient, rtol=1e-10)
        np.testing.assert_allclose(doubled.curl, 2.5 * base.curl, rtol=1e-10)
        np.testing.assert_allclose(doubled.harmonic, 2.5 * base.harmonic, atol=1e-10)

    def test_boolean_phase_alias_is_rejected(self):
        knm = np.zeros((2, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="phases must not contain boolean"):
            hodge_decomposition(knm, [True, 0.5])

    def test_boolean_coupling_alias_is_rejected(self):
        phases = np.array([0.0, 0.5], dtype=np.float64)
        knm = [[0.0, True], [0.0, 0.0]]
        with pytest.raises(ValueError, match="knm must not contain boolean"):
            hodge_decomposition(knm, phases)


class TestHodgePipelineWiring:
    """Pipeline: engine phases → Hodge decomposition → coupling analysis."""

    def test_engine_phases_to_hodge(self):
        """UPDEEngine → phases → hodge_decomposition: the three flow
        components reconstruct the coupling current and the complete
        graph carries no harmonic part."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 6
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = rng.uniform(0.1, 0.5, (n, n))
        knm = (knm + knm.T) / 2
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)

        res = hodge_decomposition(knm, phases)
        assert np.all(np.isfinite(res.gradient))
        assert np.all(np.isfinite(res.curl))
        np.testing.assert_allclose(
            res.gradient + res.curl + res.harmonic, res.flow, atol=1e-10
        )
        # Complete graph (all triangles present) → no harmonic content.
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-10)
        assert res.betti_one == 0
