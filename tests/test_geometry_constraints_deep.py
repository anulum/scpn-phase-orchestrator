# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Geometry constraints tests

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from scpn_phase_orchestrator.coupling.geometry_constraints import (
    GeometryConstraint,
    NonNegativeConstraint,
    SymmetryConstraint,
    project_knm,
    validate_knm,
)

# ── validate_knm: square check ─────────────────────────────────────────


class TestValidateKnmShape:
    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            validate_knm(np.zeros((3, 4)))

    def test_one_dimensional_raises(self):
        with pytest.raises(ValueError, match="square"):
            validate_knm(np.zeros((5,)))

    def test_three_dimensional_raises(self):
        with pytest.raises(ValueError, match="square"):
            validate_knm(np.zeros((2, 2, 2)))

    def test_1x1_valid(self):
        """1x1 zero matrix is trivially valid."""
        validate_knm(np.zeros((1, 1)))

    def test_2x2_valid(self):
        m = np.array([[0.0, 0.5], [0.5, 0.0]])
        validate_knm(m)  # Should not raise


# ── validate_knm: symmetry check ───────────────────────────────────────


class TestValidateKnmSymmetry:
    def test_asymmetric_raises(self):
        m = np.array([[0.0, 0.3], [0.7, 0.0]])
        with pytest.raises(ValueError, match="symmetric"):
            validate_knm(m)

    def test_nearly_symmetric_within_atol(self):
        """Difference smaller than atol should pass."""
        m = np.array([[0.0, 0.5], [0.5 + 1e-14, 0.0]])
        validate_knm(m, atol=1e-12)  # Should not raise

    def test_nearly_symmetric_outside_atol(self):
        """Difference larger than both atol and rtol should fail."""
        # np.allclose uses atol + rtol*|b|, so we need a difference that exceeds both.
        # With atol=1e-12, rtol=1e-5 (default), and values ~0.5: threshold ≈ 5e-6.
        # Use a difference of 0.01 which clearly exceeds any tolerance.
        m = np.array([[0.0, 0.5], [0.5 + 0.01, 0.0]])
        with pytest.raises(ValueError, match="symmetric"):
            validate_knm(m, atol=1e-12)


# ── validate_knm: non-negative check ───────────────────────────────────


class TestValidateKnmNonNegative:
    def test_negative_entry_raises(self):
        m = np.array([[0.0, -0.1], [-0.1, 0.0]])
        with pytest.raises(ValueError, match="negative"):
            validate_knm(m)

    def test_small_negative_within_atol_ok(self):
        """Tiny negative from floating-point noise should pass."""
        m = np.array([[0.0, 0.5], [0.5, 0.0]])
        m[0, 1] = -1e-15  # noise
        m[1, 0] = -1e-15
        validate_knm(m, atol=1e-12)  # Should not raise

    def test_all_zeros_valid(self):
        validate_knm(np.zeros((4, 4)))


# ── validate_knm: zero diagonal check ──────────────────────────────────


class TestValidateKnmDiagonal:
    def test_nonzero_diagonal_raises(self):
        m = np.array([[0.1, 0.5], [0.5, 0.0]])
        with pytest.raises(ValueError, match="diagonal"):
            validate_knm(m)

    def test_tiny_diagonal_within_atol_ok(self):
        m = np.array([[1e-14, 0.5], [0.5, 1e-14]])
        validate_knm(m, atol=1e-12)

    def test_large_diagonal_raises(self):
        m = np.eye(3)
        with pytest.raises(ValueError, match="diagonal"):
            validate_knm(m)


# ── validate_knm: order of checks ──────────────────────────────────────


class TestValidateKnmOrder:
    def test_non_square_checked_before_symmetry(self):
        """Shape checked first; rectangular reports shape."""
        with pytest.raises(ValueError, match="square"):
            validate_knm(np.ones((2, 3)))


# ── SymmetryConstraint ──────────────────────────────────────────────────


class TestSymmetryConstraint:
    def test_already_symmetric_unchanged(self):
        m = np.array([[0.0, 0.5], [0.5, 0.0]])
        result = SymmetryConstraint().project(m)
        assert_allclose(result, m)

    def test_asymmetric_averaged(self):
        m = np.array([[0.0, 1.0], [3.0, 0.0]])
        result = SymmetryConstraint().project(m)
        assert_allclose(result[0, 1], 2.0)
        assert_allclose(result[1, 0], 2.0)

    def test_idempotent(self):
        """Projecting twice gives same result as projecting once."""
        rng = np.random.default_rng(42)
        m = rng.standard_normal((5, 5))
        s = SymmetryConstraint()
        once = s.project(m)
        twice = s.project(once)
        assert_allclose(twice, once, atol=1e-14)

    def test_preserves_trace(self):
        """Symmetrisation preserves the diagonal."""
        rng = np.random.default_rng(7)
        m = rng.standard_normal((4, 4))
        result = SymmetryConstraint().project(m)
        assert_allclose(np.diag(result), np.diag(m))

    def test_1x1(self):
        m = np.array([[3.0]])
        result = SymmetryConstraint().project(m)
        assert_allclose(result, m)

    def test_large_matrix(self):
        n = 50
        rng = np.random.default_rng(99)
        m = rng.standard_normal((n, n))
        result = SymmetryConstraint().project(m)
        assert_allclose(result, result.T, atol=1e-14)


# ── NonNegativeConstraint ──────────────────────────────────────────────


class TestNonNegativeConstraint:
    def test_all_positive_unchanged(self):
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = NonNegativeConstraint().project(m)
        assert_allclose(result, m)

    def test_all_negative_becomes_zero(self):
        m = -np.ones((3, 3))
        result = NonNegativeConstraint().project(m)
        assert_allclose(result, np.zeros((3, 3)))

    def test_mixed_clipped(self):
        m = np.array([[1.0, -0.5], [-2.0, 3.0]])
        result = NonNegativeConstraint().project(m)
        expected = np.array([[1.0, 0.0], [0.0, 3.0]])
        assert_allclose(result, expected)

    def test_idempotent(self):
        rng = np.random.default_rng(42)
        m = rng.standard_normal((5, 5))
        nn = NonNegativeConstraint()
        once = nn.project(m)
        twice = nn.project(once)
        assert_allclose(twice, once)

    def test_zero_unchanged(self):
        m = np.zeros((4, 4))
        result = NonNegativeConstraint().project(m)
        assert_allclose(result, m)


# ── project_knm ─────────────────────────────────────────────────────────


class TestProjectKnm:
    def test_empty_constraints_only_zeros_diagonal(self):
        """With no constraints, only the diagonal is zeroed."""
        m = np.array([[5.0, 0.3, -0.1], [0.7, 2.0, 0.5], [-0.2, 0.4, 3.0]])
        result = project_knm(m, [])
        # Off-diagonal should be unchanged
        assert_allclose(result[0, 1], 0.3)
        assert_allclose(result[0, 2], -0.1)
        # Diagonal should be zero
        assert_allclose(np.diag(result), 0.0)

    def test_symmetry_then_nonneg_order_matters(self):
        """Order of constraints matters: sym then clip ≠ clip then sym."""
        m = np.array([[0.0, 1.0], [-1.0, 0.0]])
        sym_first = project_knm(m, [SymmetryConstraint(), NonNegativeConstraint()])
        clip_first = project_knm(m, [NonNegativeConstraint(), SymmetryConstraint()])
        # sym_first: (m+m^T)/2 = [[0,0],[0,0]] → clip → [[0,0],[0,0]] → diag → same
        # clip_first: max(m,0) = [[0,1],[0,0]] → sym → [[0,0.5],[0.5,0]] → diag → same
        # They differ in off-diagonal
        assert_allclose(sym_first[0, 1], 0.0)
        assert_allclose(clip_first[0, 1], 0.5)

    def test_diagonal_always_zero(self):
        """Regardless of constraints, diagonal is always zeroed at the end."""
        rng = np.random.default_rng(42)
        m = rng.uniform(0, 1, (6, 6))
        result = project_knm(m, [SymmetryConstraint(), NonNegativeConstraint()])
        assert_allclose(np.diag(result), 0.0, atol=1e-15)

    def test_does_not_mutate_input(self):
        """project_knm must not modify the input array."""
        m = np.array([[1.0, 0.3], [0.7, 2.0]])
        m_orig = m.copy()
        project_knm(m, [SymmetryConstraint()])
        assert_allclose(m, m_orig)

    def test_single_constraint(self):
        """Works with a single constraint."""
        m = np.array([[0.0, -0.5], [-0.3, 0.0]])
        result = project_knm(m, [NonNegativeConstraint()])
        assert np.all(result >= 0.0)
        assert_allclose(np.diag(result), 0.0)


# ── Custom constraint (abstract base test) ─────────────────────────────


class TestGeometryConstraintABC:
    def test_cannot_instantiate_abstract(self):
        """GeometryConstraint is abstract — direct instantiation should fail."""
        with pytest.raises(TypeError):
            GeometryConstraint()

    def test_custom_constraint(self):
        """A custom constraint implementing project() works with project_knm."""

        class ScaleConstraint(GeometryConstraint):
            def project(self, knm):
                return knm * 0.5

        m = np.array([[0.0, 2.0], [2.0, 0.0]])
        result = project_knm(m, [ScaleConstraint()])
        assert_allclose(result[0, 1], 1.0)
        assert_allclose(np.diag(result), 0.0)


# ── Roundtrip: project then validate ───────────────────────────────────


class TestProjectThenValidate:
    def test_projected_matrix_passes_validation(self):
        """A random matrix projected through sym + nonneg should pass validate_knm."""
        rng = np.random.default_rng(123)
        raw = rng.standard_normal((8, 8))
        projected = project_knm(raw, [SymmetryConstraint(), NonNegativeConstraint()])
        validate_knm(projected)  # Should not raise

    def test_large_random_matrix_roundtrip(self):
        """N=50 random matrix: project → validate succeeds."""
        rng = np.random.default_rng(456)
        raw = rng.uniform(-5, 5, (50, 50))
        projected = project_knm(raw, [SymmetryConstraint(), NonNegativeConstraint()])
        validate_knm(projected)
