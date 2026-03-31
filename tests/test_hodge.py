# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for Hodge decomposition

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.hodge import HodgeResult, hodge_decomposition


def _symmetric_coupling(n: int, k: float = 1.0) -> np.ndarray:
    knm = np.full((n, n), k)
    np.fill_diagonal(knm, 0.0)
    return knm


class TestHodgeDecomposition:
    def test_symmetric_coupling_no_curl(self):
        """Symmetric K → all coupling in gradient component, zero curl."""
        knm = _symmetric_coupling(4)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        result = hodge_decomposition(knm, phases)
        np.testing.assert_allclose(result.curl, 0.0, atol=1e-12)

    def test_antisymmetric_coupling_no_gradient(self):
        """Antisymmetric K → all coupling in curl component, zero gradient."""
        knm = np.array([[0.0, 1.0, -0.5], [-1.0, 0.0, 2.0], [0.5, -2.0, 0.0]])
        phases = np.array([0.0, 0.5, 1.0])
        result = hodge_decomposition(knm, phases)
        np.testing.assert_allclose(result.gradient, 0.0, atol=1e-12)

    def test_reconstruction(self):
        """gradient + curl + harmonic = total coupling force."""
        rng = np.random.default_rng(42)
        knm = rng.standard_normal((5, 5))
        np.fill_diagonal(knm, 0.0)
        phases = rng.uniform(0, 2 * np.pi, 5)
        result = hodge_decomposition(knm, phases)
        total = np.sum(
            knm * np.cos(phases[np.newaxis, :] - phases[:, np.newaxis]),
            axis=1,
        )
        np.testing.assert_allclose(
            result.gradient + result.curl + result.harmonic, total, atol=1e-10
        )

    def test_harmonic_near_zero(self):
        """For any K, sym+anti decomposition is exact → harmonic ≈ 0."""
        rng = np.random.default_rng(99)
        knm = rng.standard_normal((6, 6))
        phases = rng.uniform(0, 2 * np.pi, 6)
        result = hodge_decomposition(knm, phases)
        np.testing.assert_allclose(result.harmonic, 0.0, atol=1e-10)

    def test_empty_phases(self):
        result = hodge_decomposition(np.zeros((0, 0)), np.array([]))
        assert len(result.gradient) == 0
        assert len(result.curl) == 0
        assert len(result.harmonic) == 0

    def test_returns_hodge_result_dataclass(self):
        knm = _symmetric_coupling(3)
        phases = np.zeros(3)
        result = hodge_decomposition(knm, phases)
        assert isinstance(result, HodgeResult)

    def test_single_oscillator(self):
        """N=1 → no coupling, all components zero."""
        knm = np.array([[0.0]])
        phases = np.array([1.0])
        result = hodge_decomposition(knm, phases)
        assert result.gradient[0] == pytest.approx(0.0)
        assert result.curl[0] == pytest.approx(0.0)


class TestPipelineWiring:
    """Pipeline wiring: proves this module is not decorative."""

    def test_wires_into_pipeline(self):
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
