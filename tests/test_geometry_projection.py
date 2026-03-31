# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Geometry projection tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.coupling.geometry_constraints import (
    NonNegativeConstraint,
    SymmetryConstraint,
    project_knm,
)


def test_symmetry_constraint():
    asymmetric = np.array([[0.0, 0.3], [0.7, 0.0]])
    sym = SymmetryConstraint().project(asymmetric)
    np.testing.assert_allclose(sym, sym.T, atol=1e-14)
    np.testing.assert_allclose(sym[0, 1], 0.5, atol=1e-14)


def test_non_negative_constraint():
    m = np.array([[0.0, -0.5, 0.3], [0.2, 0.0, -0.1], [-0.3, 0.4, 0.0]])
    clipped = NonNegativeConstraint().project(m)
    assert np.all(clipped >= 0.0)
    assert clipped[0, 1] == 0.0
    assert clipped[0, 2] == 0.3


def test_project_knm_preserves_both():
    rng = np.random.default_rng(42)
    raw = rng.uniform(-1.0, 1.0, size=(6, 6))
    np.fill_diagonal(raw, 0.0)
    constraints = [SymmetryConstraint(), NonNegativeConstraint()]
    result = project_knm(raw, constraints)
    np.testing.assert_allclose(result, result.T, atol=1e-14)
    assert np.all(result >= 0.0)


def test_diagonal_preserved_after_projection():
    m = np.zeros((4, 4))
    m[0, 1] = -0.2
    m[1, 0] = 0.5
    constraints = [SymmetryConstraint(), NonNegativeConstraint()]
    result = project_knm(m, constraints)
    np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-15)


def test_already_valid_unchanged():
    m = np.array([[0.0, 0.5, 0.2], [0.5, 0.0, 0.3], [0.2, 0.3, 0.0]])
    result = project_knm(m, [SymmetryConstraint(), NonNegativeConstraint()])
    np.testing.assert_allclose(result, m, atol=1e-14)


class TestGeometryPipelineWiring:
    """Pipeline: project_knm enforces constraints on K_nm → engine."""

    def test_projected_knm_drives_engine(self):
        """project_knm → symmetric + non-negative K_nm → engine → R∈[0,1]."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 6
        rng = np.random.default_rng(0)
        raw = rng.standard_normal((n, n))
        np.fill_diagonal(raw, 0.0)
        projected = project_knm(raw, [SymmetryConstraint(), NonNegativeConstraint()])
        assert np.allclose(projected, projected.T)
        assert np.all(projected >= 0.0)

        eng = UPDEEngine(n, dt=0.01)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, projected, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
